"""医生审核服务：校验人工动作、保存原建议与采用结果，不修改 Agent 原输出。"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.app.core.exceptions import InvalidBusinessRequestError, InvalidStateTransitionError
from backend.app.graph.main_graph import HumanGraphCommand
from backend.app.schemas.consultation import ConsultationSnapshot, ConsultationStatus
from backend.app.schemas.review import DoctorReviewRecord, DoctorReviewRequest
from backend.app.services.consultation_service import ConsultationService


class ReviewService:
    """各审核动作有不同合法条件，备注与排序不自动终结会话。"""

    def __init__(self, consultations: ConsultationService) -> None:
        """复用会话的锁、Repository 和轮次校验边界。"""
        self.consultations = consultations

    async def review(
        self, consultation_id: UUID, request: DoctorReviewRequest
    ) -> ConsultationSnapshot:
        """业务审核先原子提交，再补偿 checkpoint，重复响应不重复医生动作。"""
        result = await self._review(consultation_id, request)
        await self.consultations.flush_actions(result)
        return result

    async def _review(
        self, consultation_id: UUID, request: DoctorReviewRequest
    ) -> ConsultationSnapshot:
        """核验当前路径和状态，事务记录人工采用，不伪装成 Agent 输出。"""
        async with self.consultations.lock(consultation_id):
            snapshot = await self.consultations.get_snapshot(consultation_id)
            self.consultations.check_turn(snapshot, request.expected_turn)
            if snapshot.reasoning is None or snapshot.status not in {
                "needs_evidence",
                "awaiting_doctor_review",
            }:
                raise InvalidStateTransitionError("当前会话没有可审核的推理结果")
            available = [item.path_id for item in snapshot.reasoning.reasoning_paths]
            if request.rerun:
                raise InvalidBusinessRequestError("需要重新推理时请使用 rerun 或补充接口")
            if (
                request.action in {"accept", "modify", "reject"}
                and request.selected_path_id not in available
            ):
                raise InvalidBusinessRequestError("此审核动作必须选择当前快照中的路径")
            if request.selected_path_id is not None and request.selected_path_id not in available:
                raise InvalidBusinessRequestError("路径不属于当前快照")
            if (
                request.action in {"accept", "modify"}
                and request.selected_path_id in snapshot.excluded_path_ids
            ):
                raise InvalidBusinessRequestError("该路径已被医生驳回，请补充或重跑后重新审核")
            if request.action == "reorder" and (
                len(request.ordered_path_ids) != len(available)
                or set(request.ordered_path_ids) != set(available)
            ):
                raise InvalidBusinessRequestError("重排必须完整、无重复地包含当前路径")
            if request.action == "note" and not request.note:
                raise InvalidBusinessRequestError("备注动作需要备注内容")
            if request.action == "modify" and not (
                request.modified_six_channels
                or request.modified_syndrome
                or request.modified_formula_pattern
            ):
                raise InvalidBusinessRequestError("修改动作至少需要一个医生修订字段")
            if not set(request.modified_six_channels).issubset(
                {"太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"}
            ):
                raise InvalidBusinessRequestError("医生修订六经包含未知名称")
            now = datetime.now(UTC)
            record = DoctorReviewRecord(
                review_id=uuid4(),
                consultation_id=consultation_id,
                action=request.action,
                selected_path_id=request.selected_path_id,
                ordered_path_ids=request.ordered_path_ids,
                modified_six_channels=request.modified_six_channels,
                modified_syndrome=request.modified_syndrome,
                modified_formula_pattern=request.modified_formula_pattern,
                note=request.note,
                agent_path_ids=available,
                doctor_adopted=request.action in {"accept", "modify"},
                created_at=now,
            )
            status: ConsultationStatus = (
                "completed"
                if request.action in {"accept", "modify"}
                else "needs_evidence" if request.action == "add_evidence" else snapshot.status
            )
            return await self.consultations.save_action(
                snapshot.model_copy(
                    update={
                        "doctor_review": record,
                        "review_history": snapshot.review_history + [record],
                        "status": status,
                        "turn": snapshot.turn + 1,
                        "completed_at": now if status == "completed" else None,
                        "excluded_path_ids": (
                            list(
                                dict.fromkeys(
                                    snapshot.excluded_path_ids + [request.selected_path_id]
                                )
                            )
                            if request.action == "reject"
                            else snapshot.excluded_path_ids
                        ),
                        "review_order": (
                            request.ordered_path_ids
                            if request.action == "reorder"
                            else snapshot.review_order
                        ),
                    }
                ),
                HumanGraphCommand(
                    kind="finish" if record.doctor_adopted else "review",
                    review=record.model_dump(mode="json"),
                ),
                expected_turn=snapshot.turn,
            )
