"""V2 图适配边界：运行唯一临床图、恢复人工暂停，不替代业务 Repository。"""

from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from backend.app.clinical_v2.extraction import extract_clinical_facts
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ClinicalV2RunOutcome,
    HumanGraphCommandV2,
    RetrievalBundleV2,
)
from backend.app.core.exceptions import ConsultationNotFoundError, InvalidStateTransitionError
from backend.app.core.ports import S4StreamPolicy, StructuredLLM
from backend.app.graph.clinical_v2_graph import build_clinical_v2_graph
from backend.app.knowledge.service import KnowledgeService


class ClinicalV2Service:
    """只操作已注入 Checkpointer 的图；业务互斥和幂等由应用服务负责。"""

    def __init__(
        self,
        llm: StructuredLLM,
        checkpointer: BaseCheckpointSaver,
        knowledge: KnowledgeService | None = None,
        *,
        max_clarification_rounds: int = 3,
        retrieval_top_k: int = 20,
        differential_top_k: int = 10,
        s4_policy: S4StreamPolicy | None = None,
    ) -> None:
        """注入应用级资源，检索预算来自 Settings。"""
        self.llm = llm
        self.knowledge = knowledge
        self.graph = build_clinical_v2_graph(
            llm,
            checkpointer,
            knowledge=knowledge,
            max_clarification_rounds=max_clarification_rounds,
            retrieval_top_k=retrieval_top_k,
            differential_top_k=differential_top_k,
            s4_policy=s4_policy,
        )

    async def parse(self, text: str) -> ClinicalEvidenceV2:
        """只执行 S1，不创建会诊、不执行 S2/RAG。"""
        return await extract_clinical_facts(text, self.llm)

    @staticmethod
    def _config(consultation_id: str) -> RunnableConfig:
        """临床线程隔离于旧协议和知识问答。"""
        return {"configurable": {"thread_id": f"clinical-v2:{consultation_id}"}}

    async def _validate_retrieval_sources(self, retrieval: RetrievalBundleV2) -> None:
        """核验 S3 来源仍属于当前知识快照；测试替身仅允许做结构化身份校验。"""
        if self.knowledge is None:
            raise InvalidStateTransitionError("知识服务未装配，不能验证 S3 来源")
        evidence_ids = [
            ref.evidence_id
            for candidate in retrieval.formula_candidates
            for ref in candidate.source_evidence
        ]
        try:
            resolved = await self.knowledge.get_evidence_by_ids(evidence_ids)
        except Exception as exc:
            raise InvalidStateTransitionError("S3 来源已变化，不能复用旧 Checkpoint") from exc
        if {item.evidence_id for item in resolved} != set(evidence_ids):
            raise InvalidStateTransitionError("S3 来源不完整，不能复用旧 Checkpoint")

    @staticmethod
    def _outcome(consultation_id: str, state: Mapping[str, Any]) -> ClinicalV2RunOutcome:
        """把真实图状态投影为公开结果。"""
        kind = {
            "needs_evidence": "needs_channel_clarification",
            "awaiting_doctor_review": "awaiting_doctor_review",
            "undetermined": "undetermined",
            "cancelled": "undetermined",
            "completed": "completed",
        }.get(str(state.get("status")), "failed")
        return ClinicalV2RunOutcome(
            kind=cast(Any, kind),
            consultation_id=consultation_id,  # type: ignore[arg-type]
            revision=int(state.get("revision", 0)),
            evidence=state.get("clinical_evidence"),
            localization=state.get("localization"),
            retrieval=state.get("retrieval"),
            comparison=state.get("comparison"),
            paused_at=state.get("paused_at"),
        )

    async def execute(
        self,
        consultation_id: str,
        text: str,
        command: HumanGraphCommandV2 | None = None,
    ) -> dict[str, Any]:
        """崩溃后从 Checkpoint 续跑；已消费动作不重复恢复 interrupt。"""
        config = self._config(consultation_id)
        checkpoint = await self.graph.aget_state(config)
        if not checkpoint.values:
            if command is not None:
                raise ConsultationNotFoundError("V2 恢复点不存在，不能恢复旧协议")
            return await self.graph.ainvoke(
                {"consultation_id": consultation_id, "clinical_text": text, "revision": 0},
                config,
            )
        consumed = command is None or checkpoint.values.get("action_id") == command.action_id
        if consumed:
            # 人工暂停或终态已经计算完成，发布重试只读取，不重新调用模型。
            if not checkpoint.next or any(task.interrupts for task in checkpoint.tasks):
                return dict(checkpoint.values)
            return await self.graph.ainvoke(None, config)
        assert command is not None
        return await self.graph.ainvoke(Command(resume=command.model_dump(mode="json")), config)

    async def create(
        self,
        clinical_text: str,
        consultation_id: str | None = None,
    ) -> ClinicalV2RunOutcome:
        """供隔离图实验使用；生产 API 通过 ConsultationService 创建业务意图。"""
        identifier = consultation_id or str(uuid4())
        return self._outcome(identifier, await self.execute(identifier, clinical_text))

    async def resume(
        self,
        consultation_id: str,
        command: HumanGraphCommandV2,
    ) -> ClinicalV2RunOutcome:
        """恢复已校验的人工动作；不自行校验业务版本。"""
        return self._outcome(consultation_id, await self.execute(consultation_id, "", command))

    async def snapshot(self, consultation_id: str) -> ClinicalV2RunOutcome:
        """仅供图诊断读取，业务 API 必须读取 Repository。"""
        checkpoint = await self.graph.aget_state(self._config(consultation_id))
        if not checkpoint.values:
            raise ConsultationNotFoundError("未找到 V2 图状态")
        return self._outcome(consultation_id, checkpoint.values)

    async def s4_failure_result(
        self, consultation_id: str, failure_code: str, *, retry_count: int = 0
    ) -> dict[str, Any] | None:
        """从同一图 Checkpoint 投影 S3 真实候选，失败时只发布只读兜底。"""
        checkpoint = await self.graph.aget_state(self._config(consultation_id))
        state = checkpoint.values
        if (
            not state
            or (checkpoint.next and checkpoint.next[0] != "s4_compare")
            or not state.get("retrieval")
            or not state.get("clinical_evidence")
            or not state.get("localization")
        ):
            return None
        # LangGraph 在某些版本会把异常节点保留为任务而使 next 为空；
        # 只要公开状态已经完整停在 S3 结果上，仍需允许一次来源校验后的只读兜底。
        if not checkpoint.next and state.get("status") not in {"running", "failed"}:
            return None
        try:
            retrieval = RetrievalBundleV2.model_validate(state["retrieval"])
            ClinicalEvidenceV2.model_validate(state["clinical_evidence"])
            ChannelLocalizationV2.model_validate(state["localization"])
        except Exception as exc:
            raise InvalidStateTransitionError("S4 失败时的 S3 Checkpoint 不完整") from exc
        if not retrieval.formula_candidates:
            return None
        # 兜底只允许复用当前 KnowledgeService 中仍存在、且版本一致的来源。
        # 这样知识库更新或 Evidence 删除后会安全关闭兜底，而不会展示过期候选。
        await self._validate_retrieval_sources(retrieval)
        return {
            "status": "failed",
            "failure_code": failure_code,
            "failed_stage": "s4_compare",
            "result_presentation": "s3_retrieval_only",
            "retryable": True,
            "s4_retry_count": retry_count,
            "clinical_evidence": state.get("clinical_evidence"),
            "localization": state.get("localization"),
            "retrieval": retrieval.model_dump(mode="json"),
            "comparison": None,
            "paused_at": None,
            "revision": state.get("revision", 0),
        }

    async def validate_s4_checkpoint(self, consultation_id: str) -> None:
        """验证单阶段 S4 重试仍停在合法 S3 Checkpoint，不重新执行前置阶段。"""
        checkpoint = await self.graph.aget_state(self._config(consultation_id))
        if not checkpoint.values or not checkpoint.next or checkpoint.next[0] != "s4_compare":
            raise InvalidStateTransitionError("当前图 Checkpoint 不是可重试的 S4 阶段")
        state = checkpoint.values
        try:
            retrieval = RetrievalBundleV2.model_validate(state["retrieval"])
            ClinicalEvidenceV2.model_validate(state["clinical_evidence"])
        except Exception as exc:
            raise InvalidStateTransitionError("S4 重试所需的 S1-S3 Checkpoint 不完整") from exc
        if not retrieval.formula_candidates:
            raise InvalidStateTransitionError("S3 没有真实候选，不能重试 S4")
        await self._validate_retrieval_sources(retrieval)
