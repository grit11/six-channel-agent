"""推理应用服务：运行应用级 LangGraph 子图，封装事实和 checkpoint 输出，不承载医学规则。"""

from typing import cast
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.core.ports import StructuredLLM
from backend.app.graph.six_channel_graph import (
    SixChannelState,
    build_six_channel_graph,
    response_from_state,
    stage_seed,
)
from backend.app.knowledge.run_cache import run_cache_scope
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text, parse_clinical_with_llm
from backend.app.schemas.clinical import ClinicalEvidence, FindingSource
from backend.app.schemas.consultation import ReasoningStage
from backend.app.schemas.reasoning import EvidenceGapContext, EvidenceGapPolicy, ScoringPolicy
from backend.app.schemas.reasoning_api import (
    ReasoningResponse,
    RetrospectiveReplayRequest,
    RetrospectiveReplayResponse,
    RetrospectiveReplayResult,
)


class ReasoningService:
    """每个应用生命周期编译一次图，每次独立分析生成不同 thread_id。"""

    def __init__(
        self,
        knowledge: KnowledgeService,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        channel_weight: float = 0.6,
        llm: StructuredLLM | None = None,
        reasoning_llm: StructuredLLM | None = None,
        scoring_policy: ScoringPolicy | None = None,
        evidence_gap_policy: EvidenceGapPolicy | None = None,
        max_supplement_rounds: int = 3,
        formula_branch_recall_top_n: int = 12,
    ) -> None:
        """注入知识服务与应用级 saver；单元测试可以使用无持久化图。"""
        self.llm = llm
        self.knowledge = knowledge
        self.graph: CompiledStateGraph = build_six_channel_graph(
            knowledge,
            checkpointer,
            channel_weight=channel_weight,
            llm=llm,
            reasoning_llm=reasoning_llm,
            scoring_policy=scoring_policy,
            evidence_gap_policy=evidence_gap_policy,
            max_supplement_rounds=max_supplement_rounds,
            formula_branch_recall_top_n=formula_branch_recall_top_n,
        )

    async def parse_text(
        self, text: str, *, source: FindingSource = "doctor_input"
    ) -> ClinicalEvidence:
        """统一解析入口，API 与图使用同一显式模式和校验边界。"""
        if self.llm is not None:
            return await parse_clinical_with_llm(text, self.llm, source=source)
        return parse_clinical_text(text, source=source)

    async def replay(self, request: RetrospectiveReplayRequest) -> RetrospectiveReplayResponse:
        """按医生分批信息累积运行独立线程；病例真实标签在推理完成后才比较。"""
        case = None
        if request.case_evidence_id:
            case = (await self.knowledge.get_evidence_by_ids([request.case_evidence_id]))[0]
            if case.knowledge_type != "medical_case" or case.source_level != "L5":
                raise InvalidBusinessRequestError("回放参考必须是真实 L5 医案引用")
        historical = case.metadata.get("six_channels", []) if case else []
        channels = (
            [value for value in historical if isinstance(value, str)]
            if isinstance(historical, list)
            else []
        )
        results: list[RetrospectiveReplayResult] = []
        cumulative: list[str] = []
        for step in request.steps:
            cumulative.append(step.text)
            result = await self.analyze_text("\n".join(cumulative), top_k=request.top_k)
            covered = (
                any(set(channels).issubset(path.six_channels) for path in result.reasoning_paths)
                if channels
                else None
            )
            results.append(
                RetrospectiveReplayResult(
                    label=step.label, result=result, historical_channel_covered=covered
                )
            )
        return RetrospectiveReplayResponse(
            steps=results,
            case_evidence_id=request.case_evidence_id,
            historical_channels=channels,
            knowledge_version=self.knowledge.version,
        )

    async def _invoke(self, inputs: SixChannelState) -> ReasoningResponse:
        """分配独立线程并验证输出，避免不同输入的 checkpoint 混入同一会话。"""
        thread_id = str(inputs.get("thread_id") or uuid4())
        if not thread_id.startswith("six-stage-v1.2:"):
            thread_id = "six-stage-v1.2:" + thread_id
        with run_cache_scope():
            state = await self.graph.ainvoke(
                {**inputs, "thread_id": thread_id}, {"configurable": {"thread_id": thread_id}}
            )
        return response_from_state(cast(SixChannelState, state), thread_id)

    async def analyze_text(
        self,
        text: str,
        *,
        source: FindingSource = "doctor_input",
        top_k: int = 3,
        thread_id: str | None = None,
        evidence_gap_context: EvidenceGapContext | None = None,
    ) -> ReasoningResponse:
        """从医生原文运行当前子图，返回经过 Schema 校验的快照。"""
        # 已存在的同 thread checkpoint 可能含上一轮 clinical_evidence，必须明确传入本轮解析。
        return await self._invoke(
            {
                "raw_input": text,
                "reset_clinical": True,
                "source": source,
                "from_stage": "clinical_parser",
                "top_k": top_k,
                "thread_id": thread_id or str(uuid4()),
                "evidence_gap_context": evidence_gap_context or EvidenceGapContext(),
            }
        )

    async def analyze_evidence(
        self,
        evidence: ClinicalEvidence,
        *,
        top_k: int = 3,
        thread_id: str | None = None,
        evidence_gap_context: EvidenceGapContext | None = None,
        from_stage: ReasoningStage = "clinical_parser",
        previous_result: ReasoningResponse | None = None,
    ) -> ReasoningResponse:
        """运行医生修订的事实与时间线；解析不会覆盖这份快照。"""
        return await self._invoke(
            {
                **(
                    stage_seed(previous_result, from_stage)
                    if previous_result is not None and from_stage != "clinical_parser"
                    else {}
                ),
                "from_stage": from_stage,
                "clinical_evidence": evidence,
                "reset_clinical": False,
                "top_k": top_k,
                "thread_id": thread_id or str(uuid4()),
                "evidence_gap_context": evidence_gap_context or EvidenceGapContext(),
            }
        )
