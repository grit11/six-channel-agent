"""V2 图适配边界：运行唯一临床图、恢复人工暂停，不替代业务 Repository。"""

from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from backend.app.clinical_v2.extraction import extract_clinical_facts
from backend.app.clinical_v2.schemas import (
    ClinicalEvidenceV2,
    ClinicalV2RunOutcome,
    HumanGraphCommandV2,
)
from backend.app.core.exceptions import ConsultationNotFoundError
from backend.app.core.ports import StructuredLLM
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
    ) -> None:
        """注入应用级资源，检索预算来自 Settings。"""
        self.llm = llm
        self.graph = build_clinical_v2_graph(
            llm,
            checkpointer,
            knowledge=knowledge,
            max_clarification_rounds=max_clarification_rounds,
            retrieval_top_k=retrieval_top_k,
            differential_top_k=differential_top_k,
        )

    async def parse(self, text: str) -> ClinicalEvidenceV2:
        """只执行 S1，不创建会诊、不执行 S2/RAG。"""
        return await extract_clinical_facts(text, self.llm)

    @staticmethod
    def _config(consultation_id: str) -> RunnableConfig:
        """临床线程隔离于旧协议和知识问答。"""
        return {"configurable": {"thread_id": f"clinical-v2:{consultation_id}"}}

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
