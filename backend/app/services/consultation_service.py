"""辨证会话应用服务：管理业务状态与同线程多轮输入，不包含医学规则。

依赖 Repository 与应用级 ReasoningService；通过版本和锁阻止并发写入同一图线程。
补充操作合并事实而非丢弃上一轮；显式修订键可替换当前事实，历史与审核原建议保留。
支持应用级异步任务与测试用同步模式；补证/审核处 interrupt 并通过同线程 Command 恢复。
"""

import asyncio
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.app.core.exceptions import (
    InvalidBusinessRequestError,
    InvalidStateTransitionError,
)
from backend.app.graph.graph_factory import ConsultationWorkflow
from backend.app.graph.main_graph import HumanGraphCommand
from backend.app.graph.six_channel_graph import stage_seed
from backend.app.persistence.repositories import ConsultationRepository
from backend.app.persistence.runtime import RuntimeWrite, operation_context
from backend.app.schemas.clinical import ClinicalEvidence, fact_time_group
from backend.app.schemas.consultation import (
    ConsultationCreateRequest,
    ConsultationEvidenceRequest,
    ConsultationHistoryPage,
    ConsultationHistoryQuery,
    ConsultationReplay,
    ConsultationRerunRequest,
    ConsultationSnapshot,
    ConsultationStatus,
)
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.schemas.runtime import GraphAction, RunJob
from backend.app.services.reasoning_service import ReasoningService
from backend.app.services.runtime_service import ReliableRuntime


class ConsultationService:
    """封装版本化会话、补充及重跑，审核操作由 ReviewService 执行。"""

    def __init__(
        self,
        repository: ConsultationRepository,
        reasoning: ReasoningService,
        *,
        protocol_version: str,
        knowledge_base_version: str,
        scoring_config_version: str,
        max_supplement_rounds: int = 3,
        versions: dict[str, str] | None = None,
        workflow: ConsultationWorkflow | None = None,
        asynchronous: bool = False,
        lease_seconds: float = 30,
        poll_seconds: float = 1,
        timeout_seconds: float = 300,
        max_attempts: int = 3,
        max_concurrent: int = 4,
    ) -> None:
        """注入存储与图，不创建网络或数据库客户端。"""
        self.repository = repository
        self.reasoning = reasoning
        self.protocol_version = protocol_version
        self.knowledge_base_version = knowledge_base_version
        self.scoring_config_version = scoring_config_version
        self.max_supplement_rounds = max_supplement_rounds
        self.versions = versions or {}
        self.workflow = workflow
        self.asynchronous = asynchronous
        self.runtime = ReliableRuntime(
            self,
            lease_seconds=lease_seconds,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            max_concurrent=max_concurrent,
        )
        self._tasks = self.runtime.tasks

    def lock(self, consultation_id: UUID) -> AbstractAsyncContextManager[None]:
        """同库 advisory lock 跨进程串行业务提交，与长时图运行锁分开。"""
        return self.repository.runtime.mutex("business:" + str(consultation_id))

    async def get_snapshot(self, consultation_id: UUID) -> ConsultationSnapshot:
        """返回服务端快照，刷新与 SSE 重连以此为准。"""
        return await self.repository.get(consultation_id)

    async def list_history(self, query: ConsultationHistoryQuery) -> ConsultationHistoryPage:
        """只读业务摘要分页；授权过滤将由 S6 可信身份层补齐。"""
        return await self.repository.list_history(query)

    async def get_replay(self, consultation_id: UUID) -> ConsultationReplay:
        """从权威快照读取本轮公开阶段，不读取其他线程或模型隐藏推理。"""
        snapshot = await self.get_snapshot(consultation_id)
        stages = snapshot.reasoning.stage_snapshots if snapshot.reasoning else []
        return ConsultationReplay(
            consultation_id=snapshot.consultation_id,
            thread_id=snapshot.thread_id,
            turn=snapshot.turn,
            status="available" if stages else "unavailable",
            stages=stages,
            versions=snapshot.versions,
            reason=(
                "当前已完成推理轮次的公开阶段输出"
                if stages
                else "本轮尚未完成或旧快照未记录阶段，未补造回放"
            ),
        )

    @staticmethod
    def check_turn(snapshot: ConsultationSnapshot, expected_turn: int | None) -> None:
        """拒绝提交来自旧快照的请求，不让旧结果覆盖新输入。"""
        if expected_turn is not None and expected_turn != snapshot.turn:
            raise InvalidStateTransitionError("会话版本已更新，请刷新后重试")

    async def create(self, request: ConsultationCreateRequest) -> ConsultationSnapshot:
        """创建记录后进入 running；异步模式立即返回，结果通过快照订阅。"""
        now = datetime.now(UTC)
        initial = ConsultationSnapshot(
            consultation_id=operation_context.get() or uuid4(),
            thread_id=str(uuid4()),
            doctor_id=request.doctor_id,
            patient_ref=request.patient_ref,
            patient_profile=request.patient_profile,
            status="created",
            turn=1,
            top_k=request.top_k,
            raw_input=request.text,
            protocol_version=self.protocol_version,
            knowledge_base_version=self.knowledge_base_version,
            scoring_config_version=self.scoring_config_version,
            created_at=now,
            updated_at=now,
            versions=dict(self.versions),
        )
        running = initial.model_copy(
            update={
                "status": "running",
                "turn": 2,
                "reasoning_run_turn": 2,
                "clinical_evidence": request.clinical_evidence,
            }
        )
        job = RunJob(
            run_id=uuid4(),
            consultation_id=running.consultation_id,
            turn=running.turn,
            snapshot=running,
            evidence=request.clinical_evidence,
        )
        await self.repository.create(
            running,
            runtime_records=[
                ("job", str(job.run_id), job.model_dump(mode="json")),
                ("active", str(job.consultation_id), {"run_id": str(job.run_id)}),
            ],
        )
        if self.asynchronous:
            self.runtime.launch(job)
            return running
        return await self.runtime.execute(job)

    async def _run(
        self, snapshot: ConsultationSnapshot, evidence: ClinicalEvidence | None
    ) -> ConsultationSnapshot:
        """事务提交 running 和精确任务输入，计算在业务锁释放后执行。"""
        running = snapshot.model_copy(
            update={
                "status": "running",
                "turn": snapshot.turn + 1,
                "reasoning_run_turn": snapshot.turn + 1,
                "reasoning": None,
                "clinical_evidence": evidence,
                "versions": dict(self.versions) if self.versions else snapshot.versions,
                "evidence_gap_context": snapshot.evidence_gap_context.model_copy(
                    update={"supplementation_round": snapshot.supplementation_round}
                ),
            }
        )
        execution = (
            running.model_copy(update={"reasoning": snapshot.reasoning})
            if snapshot.rerun_from_stage != "build_evidence"
            else running
        )
        job = RunJob(
            run_id=uuid4(),
            consultation_id=running.consultation_id,
            turn=running.turn,
            snapshot=execution,
            evidence=evidence,
        )
        running = await self.repository.save(
            running,
            expected_turn=snapshot.turn,
            runtime_records=[
                ("job", str(job.run_id), job.model_dump(mode="json")),
                ("active", str(job.consultation_id), {"run_id": str(job.run_id)}),
            ],
        )
        if self.asynchronous:
            self.runtime.launch(job)
            return running
        # 同步模式用于受控调用；调用方的业务锁必须在长时计算之前释放。
        return running

    async def close(self) -> None:
        """应用退出释放在途租约，后续实例从持久任务接管。"""
        await self.runtime.close()

    async def compute(
        self, running: ConsultationSnapshot, evidence: ClinicalEvidence | None
    ) -> ReasoningResponse:
        """只计算公开结果；存储、租约和失败协调由 ReliableRuntime 承担。"""
        if self.workflow is not None:
            return await self.workflow.analyze(
                running,
                (
                    evidence
                    if evidence is not None
                    else await self.reasoning.parse_text(running.raw_input)
                ),
            )
        if evidence is not None:
            return await self.reasoning.analyze_evidence(
                evidence,
                top_k=running.top_k,
                thread_id=running.thread_id,
                evidence_gap_context=running.evidence_gap_context,
                from_stage=running.rerun_from_stage,
                previous_result=running.reasoning,
            )
        return await self.reasoning.analyze_text(
            running.raw_input,
            top_k=running.top_k,
            thread_id=running.thread_id,
            evidence_gap_context=running.evidence_gap_context,
        )

    async def finish_sync(self, snapshot: ConsultationSnapshot) -> ConsultationSnapshot:
        """同步测试/调用在业务锁释放后执行已提交任务，避免互斥嵌套死锁。"""
        if self.asynchronous or snapshot.status != "running":
            return snapshot
        jobs = await self.repository.runtime.scan("job", snapshot.consultation_id)
        for payload in jobs:
            if payload.get("consultation_id") == str(snapshot.consultation_id):
                return await self.runtime.execute(RunJob.model_validate(payload))
        return await self.get_snapshot(snapshot.consultation_id)

    async def _save_result(
        self,
        snapshot: ConsultationSnapshot,
        result: ReasoningResponse,
        *,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """补充次数到上限仍保留缺口，但转交医生审核以免无限循环。"""
        # 兼容直接注入 ReasoningService 的调用方；业务轮数上限仍为权威控制。
        if (
            result.evidence_gap_decision is not None
            and snapshot.supplementation_round >= self.max_supplement_rounds
        ):
            decision = result.evidence_gap_decision
            result = result.model_copy(deep=True)
            result.evidence_gap_decision = decision.model_copy(
                update={"stop": True, "reason": "max_rounds"}
            )
        status: ConsultationStatus = (
            "needs_evidence"
            if result.evidence_gap_decision is not None and not result.evidence_gap_decision.stop
            else "awaiting_doctor_review"
        )
        context = snapshot.evidence_gap_context.model_copy(deep=True)
        if status == "needs_evidence":
            for gap in result.evidence_gaps:
                if gap.status == "pending":
                    context.asked_counts[gap.question_id] = (
                        context.asked_counts.get(gap.question_id, 0) + 1
                    )
        return await self.repository.save(
            snapshot.model_copy(
                update={
                    "status": status,
                    "evidence_gap_context": context,
                    "turn": snapshot.turn + 1,
                    "clinical_evidence": result.clinical_evidence,
                    "reasoning": result,
                    "excluded_path_ids": [],
                    "review_order": [],
                }
            ),
            expected_turn=snapshot.turn,
            runtime_records=runtime_records,
        )

    async def _merge(
        self, snapshot: ConsultationSnapshot, request: ConsultationEvidenceRequest
    ) -> ClinicalEvidence:
        """合并文本/结构化事实；病程整体修订须显式给出字段，允许清空但不隐式删除。"""
        for context_field in request.replace_context_fields:
            if (
                request.clinical_evidence is None
                or context_field not in request.clinical_evidence.model_fields_set
            ):
                raise InvalidBusinessRequestError("病程修订字段必须显式出现在结构化输入中")
        if not request.clinical_text and request.clinical_evidence is None:
            if request.unavailable_question_ids or request.available_question_ids:
                return snapshot.clinical_evidence or ClinicalEvidence()
            raise InvalidBusinessRequestError("补充请求必须包含临床事实或可获取状态变更")
        incoming = request.clinical_evidence
        if request.clinical_text:
            parsed = await self.reasoning.parse_text(request.clinical_text)
            if incoming is None:
                incoming = parsed
            else:
                incoming = incoming.model_copy(
                    update={"findings": incoming.findings + parsed.findings}
                )
        assert incoming is not None
        old = snapshot.clinical_evidence or ClinicalEvidence()
        supplied = [item for item in incoming.findings if item.polarity != "unknown"]
        replacement = set(request.replace_keys)
        groups = {
            (item.key, fact_time_group(item.time_scope)) for item in request.replace_fact_groups
        }
        # 逐项修订允许历史组或删除；必须来自本轮医生明确指定的键/时点，不删除其他时点。
        existing_groups = {(item.key, fact_time_group(item.time_scope)) for item in old.findings}
        incoming_groups = {
            (item.key, fact_time_group(item.time_scope)) for item in incoming.findings
        }
        if not groups.issubset(existing_groups | incoming_groups):
            raise InvalidBusinessRequestError("修订事实组不存在于原事实或本次输入")
        if not replacement.issubset({item.key for item in incoming.findings}):
            raise InvalidBusinessRequestError("修订键必须出现在本次补充事实中")
        known_keys = {item.key for item in supplied} | replacement
        findings = [
            item
            for item in old.findings
            if not (
                (item.polarity == "unknown" and item.key in known_keys)
                or (item.key in replacement and fact_time_group(item.time_scope) == "current")
                or (item.key, fact_time_group(item.time_scope)) in groups
            )
        ]
        additions = [
            item
            for item in incoming.findings
            if item.polarity != "unknown"
            or item.key in replacement
            or (item.key, fact_time_group(item.time_scope)) in groups
        ]
        for item in additions:
            if item not in findings:
                findings.append(item)
        payload = old.model_dump()
        payload["findings"] = [item.model_dump() for item in findings]
        payload["timeline"] = [
            item.model_dump()
            for item in old.timeline
            + [item for item in incoming.timeline if item not in old.timeline]
        ]
        payload["treatment_events"] = [
            item.model_dump()
            for item in old.treatment_events
            + [item for item in incoming.treatment_events if item not in old.treatment_events]
        ]
        # 修改次序/共存状态必须明确整体修订；追加仍保留双方记录供节点识别冲突。
        for context_field in request.replace_context_fields:
            assert request.clinical_evidence is not None
            value = getattr(request.clinical_evidence, context_field)
            payload[context_field] = (
                [item.model_dump() for item in value] if isinstance(value, list) else value
            )
            if context_field == "relationship_history":
                payload[context_field] = request.clinical_evidence.relationship_history.model_dump()
        for field in ("onset", "course"):
            if field not in request.replace_context_fields and getattr(incoming, field) is not None:
                payload[field] = getattr(incoming, field)
        for field in (
            "fever_chills",
            "sweating",
            "thirst_mouth",
            "appetite_digestive",
            "stool",
            "urination",
            "chest_hypochondrium",
            "cough_dyspnea_phlegm",
            "spirit_sleep",
            "limbs_body",
            "tongue",
            "pulse",
        ):
            payload[field] = {**payload[field], **getattr(incoming, field)}
        payload["pain"] = old.pain + incoming.pain
        payload["prior_treatment"] = old.prior_treatment + incoming.prior_treatment
        return ClinicalEvidence.model_validate(payload)

    async def add_evidence(
        self, consultation_id: UUID, request: ConsultationEvidenceRequest
    ) -> ConsultationSnapshot:
        """提交事实后按同步/异步运行策略继续，长时计算不持有业务锁。"""
        result = await self.finish_sync(await self._add_evidence(consultation_id, request))
        await self.flush_actions(result)
        return result

    async def rerun(
        self, consultation_id: UUID, request: ConsultationRerunRequest
    ) -> ConsultationSnapshot:
        """提交同线程任务后执行，跨进程图锁负责执行互斥。"""
        return await self.finish_sync(await self._rerun(consultation_id, request))

    async def _add_evidence(
        self, consultation_id: UUID, request: ConsultationEvidenceRequest
    ) -> ConsultationSnapshot:
        """保存补充并按原 thread 重算；resume=false 仅保存事实，等待医生继续。"""
        async with self.lock(consultation_id):
            snapshot = await self.get_snapshot(consultation_id)
            self.check_turn(snapshot, request.expected_turn)
            if snapshot.status not in {"needs_evidence", "awaiting_doctor_review"}:
                raise InvalidStateTransitionError("当前会话状态不允许补充证据")
            context = snapshot.evidence_gap_context.model_copy(deep=True)
            unavailable = set(request.unavailable_question_ids)
            available = set(request.available_question_ids)
            gaps = snapshot.reasoning.evidence_gaps if snapshot.reasoning else []
            allowed = {gap.question_id for gap in gaps if gap.status in {"pending", "unavailable"}}
            allowed.update(context.unavailable_question_ids)
            if unavailable & available or not (unavailable | available).issubset(allowed):
                raise InvalidBusinessRequestError(
                    "可获取状态只能修改当前问题或已保存的无法获取问题，且不能同时设置相反状态"
                )
            evidence = await self._merge(snapshot, request)
            unavailable_before = set(context.unavailable_question_ids)
            context.unavailable_question_ids = sorted(
                (unavailable_before | unavailable) - available
            )
            previous = snapshot.clinical_evidence or ClinicalEvidence()
            changed = evidence.model_dump() != previous.model_dump() or unavailable_before != set(
                context.unavailable_question_ids
            )
            if not changed:
                raise InvalidBusinessRequestError("本次补证没有新增或修订信息，不消耗补证轮次")
            context.supplementation_round = snapshot.supplementation_round + 1
            updated = snapshot.model_copy(
                update={
                    "clinical_evidence": evidence,
                    "supplementation_round": snapshot.supplementation_round + 1,
                    "evidence_gap_context": context,
                    "status": "needs_evidence",
                    "reasoning": None,
                    "rerun_from_stage": "build_evidence",
                }
            )
            if request.resume:
                return await self._run(updated, evidence)
            return await self.save_action(
                updated.model_copy(update={"turn": snapshot.turn + 1}),
                HumanGraphCommand(
                    kind="save_evidence",
                    clinical_evidence=evidence,
                    supplementation_round=updated.supplementation_round,
                    evidence_gap_context=context,
                ),
                expected_turn=snapshot.turn,
            )

    async def _rerun(
        self, consultation_id: UUID, request: ConsultationRerunRequest
    ) -> ConsultationSnapshot:
        """从当前事实重算同一图线程；终态必须另建会话，不允许隐式复活。"""
        async with self.lock(consultation_id):
            snapshot = await self.get_snapshot(consultation_id)
            self.check_turn(snapshot, request.expected_turn)
            if snapshot.status not in {"needs_evidence", "awaiting_doctor_review", "failed"}:
                raise InvalidStateTransitionError("当前会话状态不允许重跑")
            if (
                request.top_k is not None
                and request.top_k != snapshot.top_k
                and request.from_stage in {"differential", "evidence_gap", "topk", "similar_cases"}
            ):
                # 方证召回数量依赖 top_k，改变它之后不能复用旧方证和后序结果。
                raise InvalidBusinessRequestError("Top-K 已改变，请从事实构建或方证候选前重算")
            if request.top_k is not None:
                snapshot.top_k = request.top_k
            if request.from_stage != "build_evidence":
                if snapshot.reasoning is None or snapshot.clinical_evidence is None:
                    raise InvalidBusinessRequestError("阶段重跑需要已完成且未修改的前序结果")
                if snapshot.clinical_evidence != snapshot.reasoning.clinical_evidence:
                    raise InvalidBusinessRequestError("事实已改变，必须从事实构建重算")
                if snapshot.versions != self.versions and self.versions:
                    raise InvalidBusinessRequestError("运行配置已改变，必须从事实构建重算")
                stage_seed(snapshot.reasoning, request.from_stage)
            snapshot.rerun_from_stage = request.from_stage
            return await self._run(snapshot, snapshot.clinical_evidence)

    async def cancel(self, consultation_id: UUID) -> ConsultationSnapshot:
        """业务取消先提交，再停止本地任务或由租约续期检测跨进程取消。"""
        result = await self._cancel(consultation_id)
        task = self._tasks.get(consultation_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.flush_actions(result)
        return result

    async def _cancel(self, consultation_id: UUID) -> ConsultationSnapshot:
        """取消尚未结束的会话，完成/取消后的再次调用返回状态冲突。"""
        async with self.lock(consultation_id):
            snapshot = await self.get_snapshot(consultation_id)
            if snapshot.status in {"completed", "cancelled"}:
                raise InvalidStateTransitionError("会话已结束")
            # 跨进程取消通过业务终态 fence；旧任务即使尚未收到取消也不能发布结果。
            # 活租约任务不在恢复页内，按当前轮次保存 job 索引便于取消。
            active = await self.repository.runtime.read("active", str(consultation_id))
            records: list[RuntimeWrite] = []
            if active:
                raw = await self.repository.runtime.read("job", str(active["run_id"]))
                if raw:
                    job = RunJob.model_validate(raw)
                    job.cancel_requested = True
                    records.append(("job", str(job.run_id), job.model_dump(mode="json")))
            updated = snapshot.model_copy(
                update={
                    "status": "cancelled",
                    "turn": snapshot.turn + 1,
                    "completed_at": datetime.now(UTC),
                }
            )
            return await self.save_action(
                updated,
                HumanGraphCommand(kind="cancel"),
                expected_turn=snapshot.turn,
                runtime_records=records,
            )

    async def request_review(
        self, consultation_id: UUID, request: ConsultationRerunRequest
    ) -> ConsultationSnapshot:
        """转交审核先保存权威状态，再在业务锁之外同步 checkpoint。"""
        result = await self._request_review(consultation_id, request)
        await self.flush_actions(result)
        return result

    async def _request_review(
        self, consultation_id: UUID, request: ConsultationRerunRequest
    ) -> ConsultationSnapshot:
        """医生决定以当前结果进入审核，缺口保留，不宣称已经补齐。"""
        async with self.lock(consultation_id):
            snapshot = await self.get_snapshot(consultation_id)
            self.check_turn(snapshot, request.expected_turn)
            if snapshot.status != "needs_evidence" or snapshot.reasoning is None:
                raise InvalidStateTransitionError("当前没有可转交审核的补证结果")
            result = snapshot.reasoning.model_copy(deep=True)
            if result.evidence_gap_decision is not None:
                result.evidence_gap_decision = result.evidence_gap_decision.model_copy(
                    update={"stop": True, "reason": "doctor_current_information"}
                )
            return await self.save_action(
                snapshot.model_copy(
                    update={
                        "status": "awaiting_doctor_review",
                        "reasoning": result,
                        "turn": snapshot.turn + 1,
                    }
                ),
                HumanGraphCommand(kind="review_ready"),
                expected_turn=snapshot.turn,
            )

    async def save_action(
        self,
        snapshot: ConsultationSnapshot,
        command: HumanGraphCommand,
        *,
        expected_turn: int,
        runtime_records: list[RuntimeWrite] | None = None,
    ) -> ConsultationSnapshot:
        """原子保存业务动作和 checkpoint 补偿；不要求跨库原子提交。"""
        action = GraphAction(
            action_id=uuid4(),
            consultation_id=snapshot.consultation_id,
            target_turn=snapshot.turn,
            thread_id=snapshot.thread_id,
            command=command.model_dump(mode="json"),
            snapshot=snapshot,
        )
        saved = await self.repository.save(
            snapshot,
            expected_turn=expected_turn,
            runtime_records=(runtime_records or [])
            + [("action", str(action.action_id), action.model_dump(mode="json"))],
        )
        return saved

    async def flush_actions(self, snapshot: ConsultationSnapshot) -> None:
        """业务锁释放后同步 outbox，失败保留记录并报告待恢复状态。"""
        try:
            actions = [
                GraphAction.model_validate(payload)
                for payload in await self.repository.runtime.scan(
                    "action", snapshot.consultation_id
                )
            ]
            for action in sorted(actions, key=lambda item: item.target_turn):
                await self.runtime.compensate(action)
        except Exception as exc:
            self.runtime._log("checkpoint_compensation_pending", snapshot, exc)
            await self.repository.runtime.emit(
                snapshot.consultation_id,
                snapshot.turn,
                "recovery_required",
                {"component": "checkpoint", "pending": True},
            )
