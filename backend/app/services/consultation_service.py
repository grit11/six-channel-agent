"""V2 会诊应用服务：事务保存运行意图、人工动作和严格公开快照。

本模块负责业务版本与幂等；医学判断由唯一 Graph 完成，不读取旧规则 DTO。
"""

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import ValidationError

from backend.app.clinical_v2.schemas import HumanGraphCommandV2
from backend.app.core.exceptions import (
    ConfigurationError,
    InvalidBusinessRequestError,
    InvalidStateTransitionError,
)
from backend.app.persistence.repositories import ConsultationRepository
from backend.app.persistence.runtime import receipt_context
from backend.app.schemas.consultation import (
    ConsultationHistoryPage,
    ConsultationHistoryQuery,
    ConsultationSnapshot,
)
from backend.app.schemas.review import DoctorReviewRequest
from backend.app.schemas.runtime import RunJob
from backend.app.services.clinical_v2_service import ClinicalV2Service

if TYPE_CHECKING:
    from backend.app.services.runtime_service import ReliableRuntime


class ConsultationService:
    """Repository 是业务权威，后台运行由应用 lifespan 管理。"""

    def __init__(
        self,
        repository: ConsultationRepository,
        workflow: ClinicalV2Service | None = None,
        *,
        versions: dict[str, str] | None = None,
        run_mode: str = "async",
    ) -> None:
        """注入存储和图能力，不在服务内创建 Provider。"""
        self.repository = repository
        self.workflow = workflow
        self.versions = versions or {}
        self.run_mode = run_mode
        self.runtime: ReliableRuntime | None = None

    def _require_workflow(self) -> None:
        if self.workflow is None or self.runtime is None:
            raise ConfigurationError("V2 临床模型或运行调度未装配")

    async def _dispatch(self, job: RunJob, snapshot: ConsultationSnapshot) -> ConsultationSnapshot:
        assert self.runtime is not None
        if self.run_mode == "sync":
            return await self.runtime.execute(job)
        self.runtime.launch(job)
        return snapshot

    async def create(
        self,
        text: str,
        consultation_id: UUID | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> ConsultationSnapshot:
        """先事务保存会诊和任务，再执行图；响应丢失可安全重试。"""
        self._require_workflow()
        if not text.strip():
            raise InvalidBusinessRequestError("病例原文不能为空")
        identifier = consultation_id or uuid4()
        key = "create:" + (idempotency_key or str(identifier))
        digest = hashlib.sha256(text.encode()).hexdigest()
        async with self.repository.runtime.mutex("request:" + key):
            receipt = await self.repository.runtime.read("receipt", key)
            if receipt:
                if receipt["digest"] != digest:
                    raise InvalidStateTransitionError("幂等键已用于不同的创建请求")
                return ConsultationSnapshot.model_validate(receipt["result"])
            now = datetime.now(UTC)
            snapshot = ConsultationSnapshot(
                consultation_id=identifier,
                thread_id=f"clinical-v2:{identifier}",
                status="running",
                turn=1,
                raw_input=text,
                versions=self.versions,
                created_at=now,
                updated_at=now,
                knowledge_base_version=self.versions.get("knowledge_base_version", "1.2"),
            )
            job = RunJob(consultation_id=identifier, turn=1)
            token = receipt_context.set((key, digest))
            try:
                saved = await self.repository.create(
                    snapshot,
                    runtime_records=[("job", str(job.run_id), job.model_dump(mode="json"))],
                )
            finally:
                receipt_context.reset(token)
        return await self._dispatch(job, saved)

    async def get_snapshot(self, consultation_id: UUID) -> ConsultationSnapshot:
        """读取权威业务状态，不用 checkpoint 冒充业务历史。"""
        return await self.repository.get(consultation_id)

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """返回不包含患者原文的分页摘要。"""
        return await self.repository.list_history(query)

    async def resume(
        self,
        identifier: UUID,
        command: HumanGraphCommandV2,
    ) -> ConsultationSnapshot:
        """核验动作身份、版本、暂停点和收据后原子提交恢复意图。"""
        self._require_workflow()
        if command.consultation_id != str(identifier):
            raise InvalidBusinessRequestError("人工动作会诊身份不匹配")
        key = f"action:{identifier}:{command.action_id}"
        digest = hashlib.sha256(command.model_dump_json().encode()).hexdigest()
        async with self.repository.runtime.mutex("business:" + str(identifier)):
            receipt = await self.repository.runtime.read("receipt", key)
            if receipt:
                if receipt["digest"] != digest:
                    raise InvalidStateTransitionError("人工动作幂等键被重复用于不同内容")
                return ConsultationSnapshot.model_validate(receipt["result"])
            current = await self.repository.get(identifier)
            if command.revision != current.revision or command.action_turn != current.turn:
                raise InvalidStateTransitionError("STALE_REVISION：请刷新后再提交")
            self._validate_command(current, command)
            changes: dict[str, object] = {
                "turn": current.turn + 1,
                "status": "running",
                "paused_at": None,
            }
            if command.clinical_text or command.clinical_evidence:
                changes.update(
                    raw_input=(
                        command.clinical_text
                        or (command.clinical_evidence.raw_text if command.clinical_evidence else "")
                    ),
                    clinical_evidence=None,
                    localization=None,
                    retrieval=None,
                    comparison=None,
                    completed_stages=[],
                    revision=current.revision + 1,
                )
            pending = current.model_copy(update=changes)
            job = RunJob(consultation_id=identifier, turn=pending.turn, command=command)
            token = receipt_context.set((key, digest))
            try:
                saved = await self.repository.save(
                    pending,
                    expected_turn=current.turn,
                    runtime_records=[("job", str(job.run_id), job.model_dump(mode="json"))],
                )
                await self.repository.runtime.emit(
                    identifier,
                    saved.turn,
                    "s4_retry_started",
                    {"retry_count": saved.s4_retry_count, "failed_stage": "s4_compare"},
                )
            finally:
                receipt_context.reset(token)
        return await self._dispatch(job, saved)

    async def retry_s4(
        self, identifier: UUID, *, expected_turn: int, expected_revision: int, idempotency_key: str
    ) -> ConsultationSnapshot:
        """只从合法 S4 失败快照重建运行意图，图状态继续自 S3 Checkpoint。"""
        self._require_workflow()
        if not idempotency_key.strip():
            raise InvalidBusinessRequestError("S4 重试必须提供幂等键")
        key = f"retry-s4:{identifier}:{idempotency_key}"
        digest = hashlib.sha256(
            f"{identifier}:{expected_turn}:{expected_revision}".encode()
        ).hexdigest()
        async with self.repository.runtime.mutex("business:" + str(identifier)):
            receipt = await self.repository.runtime.read("receipt", key)
            if receipt:
                if receipt["digest"] != digest:
                    raise InvalidStateTransitionError("幂等键已用于不同的 S4 重试请求")
                return ConsultationSnapshot.model_validate(receipt["result"])
            current = await self.repository.get(identifier)
            if (
                current.status != "failed"
                or current.failed_stage != "s4_compare"
                or current.result_presentation != "s3_retrieval_only"
                or not current.retryable
                or current.turn != expected_turn
                or current.revision != expected_revision
                or current.retrieval is None
            ):
                raise InvalidStateTransitionError("当前会诊没有可恢复的 S4 Checkpoint")
            assert self.workflow is not None
            await self.workflow.validate_s4_checkpoint(str(identifier))
            pending = current.model_copy(
                update={
                    "status": "running",
                    "turn": current.turn + 1,
                    "paused_at": None,
                    "failure_code": None,
                    "failed_stage": None,
                    "result_presentation": "none",
                    "retryable": False,
                    "s4_retry_count": current.s4_retry_count + 1,
                }
            )
            job = RunJob(
                consultation_id=identifier,
                turn=pending.turn,
                retry_s4=True,
                idempotency_key=idempotency_key,
            )
            token = receipt_context.set((key, digest))
            try:
                saved = await self.repository.save(
                    pending,
                    expected_turn=current.turn,
                    runtime_records=[("job", str(job.run_id), job.model_dump(mode="json"))],
                )
            finally:
                receipt_context.reset(token)
        return await self._dispatch(job, saved)

    async def retry_clinical_stage(
        self,
        identifier: UUID,
        *,
        expected_turn: int,
        expected_revision: int,
        idempotency_key: str,
        owner_id: str | None = None,
    ) -> ConsultationSnapshot:
        """使用权威失败快照、业务锁和收据继续 S1/S2，绝不重新初始化整图。"""
        self._require_workflow()
        key = f"retry-clinical:{identifier}:{idempotency_key}"
        digest = hashlib.sha256(
            f"{identifier}:{expected_turn}:{expected_revision}".encode()
        ).hexdigest()
        async with self.repository.runtime.mutex("business:" + str(identifier)):
            current = await self.repository.get(identifier)
            if (
                owner_id is not None or current.doctor_id is not None
            ) and current.doctor_id != owner_id:
                raise InvalidBusinessRequestError("会诊归属不匹配")
            receipt = await self.repository.runtime.read("receipt", key)
            if receipt:
                if receipt["digest"] != digest:
                    raise InvalidStateTransitionError("幂等键内容冲突")
                return ConsultationSnapshot.model_validate(receipt["result"])
            if (
                current.status != "failed"
                or current.failed_stage not in {"s1_extract", "s2_localize"}
                or not current.retryable
                or current.turn != expected_turn
                or current.revision != expected_revision
            ):
                raise InvalidStateTransitionError("当前会诊没有可恢复的 S1/S2 失败阶段")
            assert self.workflow is not None
            await self.workflow.validate_clinical_checkpoint(
                str(identifier),
                current.failed_stage,
                current.raw_input,
                current.revision,
                current.clinical_runtime_digest or "",
                current.clinical_evidence.model_dump_json() if current.clinical_evidence else None,
            )
            pending = current.model_copy(
                update={
                    "status": "running",
                    "turn": current.turn + 1,
                    "failed_stage": None,
                    "failure_code": None,
                    "retryable": False,
                    "clinical_retry_count": current.clinical_retry_count + 1,
                    "completed_at": None,
                }
            )
            job = RunJob(
                consultation_id=identifier, turn=pending.turn, idempotency_key=idempotency_key
            )
            token = receipt_context.set((key, digest))
            try:
                saved = await self.repository.save(
                    pending,
                    expected_turn=current.turn,
                    runtime_records=[("job", str(job.run_id), job.model_dump(mode="json"))],
                )
            finally:
                receipt_context.reset(token)
        return await self._dispatch(job, saved)

    @staticmethod
    def _validate_command(snapshot: ConsultationSnapshot, command: HumanGraphCommandV2) -> None:
        """软件契约校验；不依据医学词项改变模型候选。"""
        if snapshot.status not in {"needs_evidence", "awaiting_doctor_review"}:
            raise InvalidStateTransitionError("当前会诊未处于人工暂停")
        if command.kind == "cancel":
            return
        expected = "clarify" if snapshot.paused_at == "s2_clarification" else "review"
        if command.kind != expected:
            raise InvalidStateTransitionError("动作与当前暂停阶段不匹配")
        if command.clinical_text is not None and not command.clinical_text.strip():
            raise InvalidBusinessRequestError("补证文本不能为空")
        if (
            command.clinical_evidence
            and command.clinical_text
            and command.clinical_evidence.raw_text != command.clinical_text
        ):
            raise InvalidBusinessRequestError("修订事实与病例原文不一致")
        if command.kind == "review":
            try:
                review = DoctorReviewRequest.model_validate(command.review or {})
            except ValidationError as exc:
                raise InvalidBusinessRequestError("医生审核字段不符合契约") from exc
            paths = (
                {p.path_id for p in snapshot.comparison.ordered_paths}
                if snapshot.comparison
                else set()
            )
            if review.selected_path_id and review.selected_path_id not in paths:
                raise InvalidBusinessRequestError("选择的路径不属于当前轮")
            if review.ordered_path_ids and (
                len(set(review.ordered_path_ids)) != len(review.ordered_path_ids)
                or not set(review.ordered_path_ids).issubset(paths)
            ):
                raise InvalidBusinessRequestError("人工路径顺序引用无效")
            if review.action == "accept" and (
                not paths
                or snapshot.comparison is None
                or snapshot.comparison.status != "ranked"
            ):
                raise InvalidBusinessRequestError("未决结果不能接受为已辨证结论")
            if review.action != "add_evidence":
                return
        if command.no_additional_information and command.kind == "clarify":
            return
        if not command.clinical_text and command.clinical_evidence is None:
            raise InvalidBusinessRequestError("补证没有新事实")
        if command.clinical_text == snapshot.raw_input and command.clinical_evidence is None:
            raise InvalidBusinessRequestError("补证与本轮原文相同")
