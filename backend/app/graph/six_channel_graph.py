"""六经分析子图编排：按协议连接节点，checkpoint 由生命周期注入，不保存客户端或隐藏推理。"""

from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter

from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import parse_clinical_text, parse_clinical_with_llm
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles_node
from backend.app.nodes.evidence_gap import build_evidence_gaps, decide_evidence_gap_stop
from backend.app.nodes.formula_candidate import generate_formula_candidates
from backend.app.nodes.similar_cases import retrieve_similar_cases
from backend.app.nodes.syndrome_relation import judge_syndrome_relationships
from backend.app.nodes.topk_builder import build_topk_paths
from backend.app.observability.progress import emit_progress, stage_scope
from backend.app.schemas.clinical import ClinicalEvidence, FindingSource
from backend.app.schemas.consultation import ReasoningStage
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    DifferentialResult,
    EightPrinciplesState,
    EvidenceGap,
    EvidenceGapContext,
    EvidenceGapDecision,
    EvidenceGapPolicy,
    FormulaCandidate,
    ReasoningPath,
    ReasoningStageSnapshot,
    ScoringPolicy,
    SimilarCaseAssessment,
    SyndromeRelationship,
)
from backend.app.schemas.reasoning_api import ReasoningResponse


class SixChannelState(TypedDict, total=False):
    """可序列化子图状态；尚未加入安全、审核和补充路由。"""

    thread_id: str
    raw_input: str
    source: FindingSource
    top_k: int
    clinical_evidence: ClinicalEvidence
    eight_principles: EightPrinciplesState
    channel_candidates: list[ChannelCandidate]
    syndrome_relationships: list[SyndromeRelationship]
    formula_candidates: list[FormulaCandidate]
    differential_results: list[DifferentialResult]
    evidence_gaps: list[EvidenceGap]
    evidence_gap_context: EvidenceGapContext
    evidence_gap_decision: EvidenceGapDecision
    reasoning_paths: list[ReasoningPath]
    excluded_paths: list[ReasoningPath]
    evidence_catalog: list[Evidence]
    completed_stages: list[str]
    similar_cases: list[SimilarCaseAssessment]
    stage_snapshots: list[ReasoningStageSnapshot]
    from_stage: ReasoningStage


STAGES: list[ReasoningStage] = [
    "build_evidence",
    "eight_principles",
    "channel_candidates",
    "syndrome_relationship",
    "formula_candidates",
    "differential",
    "evidence_gap",
    "topk",
    "similar_cases",
]


def stage_seed(result: ReasoningResponse, from_stage: ReasoningStage) -> SixChannelState:
    """复用同一已核对事实的前序公开 DTO，保留审计；缺任何前序快照则拒绝跳算。"""
    preceding = STAGES[: STAGES.index(from_stage)]
    snapshots = result.stage_snapshots[: len(preceding)]
    if [item.stage for item in snapshots] != preceding:
        raise InvalidBusinessRequestError("前序阶段快照不完整，请从事实构建重算")
    # Agent 结果中医案引用为后置增补；从 Top-K 快照重读，防止旧后置引用在重跑时残留。
    seed: SixChannelState = {
        "clinical_evidence": result.clinical_evidence.model_copy(deep=True),
        "eight_principles": result.eight_principles.model_copy(deep=True),
        "channel_candidates": [item.model_copy(deep=True) for item in result.candidates],
        "syndrome_relationships": [
            item.model_copy(deep=True) for item in result.syndrome_relationships
        ],
        "formula_candidates": [item.model_copy(deep=True) for item in result.formula_candidates],
        "differential_results": [
            item.model_copy(deep=True) for item in result.differential_results
        ],
        "evidence_gaps": [item.model_copy(deep=True) for item in result.evidence_gaps],
        "completed_stages": list(preceding),
        "stage_snapshots": [item.model_copy(deep=True) for item in snapshots],
        "from_stage": from_stage,
    }
    if from_stage == "similar_cases":
        topk = snapshots[-1].payload
        seed.update(
            {
                "reasoning_paths": TypeAdapter(list[ReasoningPath]).validate_python(
                    topk["reasoning_paths"]
                ),
                "excluded_paths": TypeAdapter(list[ReasoningPath]).validate_python(
                    topk["excluded_paths"]
                ),
                "evidence_catalog": TypeAdapter(list[Evidence]).validate_python(
                    topk["evidence_catalog"]
                ),
                "evidence_gap_decision": EvidenceGapDecision.model_validate(
                    topk["evidence_gap_decision"]
                ),
            }
        )
    return seed


def build_six_channel_graph(
    knowledge: KnowledgeService,
    checkpointer: BaseCheckpointSaver | Literal[False] | None = None,
    *,
    channel_weight: float = 0.6,
    llm: StructuredLLM | None = None,
    reasoning_llm: StructuredLLM | None = None,
    scoring_policy: ScoringPolicy | None = None,
    evidence_gap_policy: EvidenceGapPolicy | None = None,
    max_supplement_rounds: int = 3,
) -> CompiledStateGraph:
    """编译应用级子图，知识端口放在节点闭包，不进入 checkpoint State。"""

    async def build_evidence(state: SixChannelState) -> SixChannelState:
        """优先使用医生结构化输入，禁止原文解析覆盖编辑事实。"""
        evidence = state.get("clinical_evidence")
        if evidence is None:
            evidence = (
                await parse_clinical_with_llm(
                    state["raw_input"], llm, source=state.get("source", "doctor_input")
                )
                if llm is not None
                else parse_clinical_text(
                    state["raw_input"], source=state.get("source", "doctor_input")
                )
            )
        else:
            evidence = ClinicalEvidence.model_validate(evidence.model_dump())
        return {"clinical_evidence": evidence, "completed_stages": ["build_evidence"]}

    async def analyze_principles(state: SixChannelState) -> SixChannelState:
        """计算八纲倾向，返回局部状态更新。"""
        return {
            "eight_principles": await analyze_eight_principles_node(
                state["clinical_evidence"], knowledge, reasoning_llm
            ),
            "completed_stages": state["completed_stages"] + ["eight_principles"],
        }

    async def generate_channels(state: SixChannelState) -> SixChannelState:
        """调用候选节点，保持独立排序和来源校验。"""
        return {
            "channel_candidates": await generate_channel_candidates(
                state["clinical_evidence"],
                state["eight_principles"],
                knowledge,
                top_k=5,
                llm=reasoning_llm,
            ),
            "completed_stages": state["completed_stages"] + ["channel_candidates"],
        }

    async def judge_relationship(state: SixChannelState) -> SixChannelState:
        """在方证节点之前核验时间线；缺失保持 uncertain。"""
        return {
            "syndrome_relationships": await judge_syndrome_relationships(
                state["clinical_evidence"], state["channel_candidates"], knowledge
            ),
            "completed_stages": state["completed_stages"] + ["syndrome_relationship"],
        }

    async def generate_formulas(state: SixChannelState) -> SixChannelState:
        """召回来源分层的方证候选，不输出处方或剂量。"""
        return {
            "formula_candidates": await generate_formula_candidates(
                state["clinical_evidence"],
                state["channel_candidates"],
                knowledge,
                top_k=min(5, max(state.get("top_k", 3), 3)),
            ),
            "completed_stages": state["completed_stages"] + ["formula_candidates"],
        }

    async def compare_formulas(state: SixChannelState) -> SixChannelState:
        """应用来源鉴别规则，保留未解决维度。"""
        return {
            "differential_results": await compare_formula_candidates(
                state["clinical_evidence"], state["formula_candidates"], knowledge
            ),
            "completed_stages": state["completed_stages"] + ["differential"],
        }

    async def analyze_gaps(state: SixChannelState) -> SixChannelState:
        """生成来源核验的补证排序，主图负责人工暂停。"""
        return {
            "evidence_gaps": await build_evidence_gaps(
                state["differential_results"],
                state["clinical_evidence"],
                knowledge,
                context=state.get("evidence_gap_context"),
                policy=evidence_gap_policy,
            ),
            "completed_stages": state["completed_stages"] + ["evidence_gap"],
        }

    async def construct_paths(state: SixChannelState) -> SixChannelState:
        """将上游结果组装为待审核路径，通过唯一知识边界精确核验引用。"""
        excluded: list[ReasoningPath] = []
        paths, catalog = await build_topk_paths(
            state["clinical_evidence"],
            state["eight_principles"],
            state["channel_candidates"],
            state["syndrome_relationships"],
            state["formula_candidates"],
            state["differential_results"],
            knowledge,
            top_k=state.get("top_k", 3),
            channel_weight=channel_weight,
            scoring_policy=scoring_policy,
            excluded_paths=excluded,
            evidence_gaps=state["evidence_gaps"],
        )
        return {
            "reasoning_paths": paths,
            "evidence_gap_decision": decide_evidence_gap_stop(
                state["evidence_gaps"],
                state["differential_results"],
                paths,
                supplementation_round=state.get(
                    "evidence_gap_context", EvidenceGapContext()
                ).supplementation_round,
                max_rounds=max_supplement_rounds,
                policy=evidence_gap_policy,
            ),
            "excluded_paths": excluded,
            "evidence_catalog": catalog,
            "completed_stages": state["completed_stages"] + ["topk"],
        }

    async def similar_cases(state: SixChannelState) -> SixChannelState:
        """经典路径已排序后检索历史医案，禁止病例回流改写当前候选和评分。"""
        paths = [path.model_copy(deep=True) for path in state["reasoning_paths"]]
        audits, sources = await retrieve_similar_cases(state["clinical_evidence"], paths, knowledge)
        relation_refs = [
            ref
            for relation in state["syndrome_relationships"]
            for audit in relation.assessments
            for ref in audit.evidence_ids
        ]
        relation_sources = await knowledge.get_evidence_by_ids(relation_refs)
        return {
            "reasoning_paths": paths,
            "similar_cases": audits,
            "evidence_catalog": list(
                {
                    item.evidence_id: item
                    for item in [*state["evidence_catalog"], *sources, *relation_sources]
                }.values()
            ),
            "completed_stages": state["completed_stages"] + ["similar_cases"],
        }

    def recorded(
        stage: str, node: Callable[[SixChannelState], Awaitable[SixChannelState]]
    ) -> RunnableLambda[SixChannelState, SixChannelState]:
        """阶段只保存公开 DTO 输出；每轮从事实构建清零，旧 checkpoint 不混入新轮。"""

        async def run(state: SixChannelState) -> SixChannelState:
            """成功节点输出序列化快照，失败不补造完成状态。"""
            async with stage_scope(stage):
                started = perf_counter()
                await emit_progress("stage_started", {})
                try:
                    update = await node(state)
                except Exception:
                    await emit_progress(
                        "stage_failed", {"duration_ms": int((perf_counter() - started) * 1000)}
                    )
                    raise
                await emit_progress(
                    "stage_completed", {"duration_ms": int((perf_counter() - started) * 1000)}
                )
            previous = [] if stage == "build_evidence" else state.get("stage_snapshots", [])
            payload = TypeAdapter(dict[str, object]).dump_python(
                {
                    key: value
                    for key, value in update.items()
                    if key not in {"completed_stages", "stage_snapshots"}
                },
                mode="json",
            )
            return {
                **update,
                "stage_snapshots": [
                    *previous,
                    ReasoningStageSnapshot(
                        sequence=len(previous) + 1, stage=stage, payload=payload
                    ),
                ],
            }

        return RunnableLambda(run)

    graph = StateGraph(SixChannelState)
    graph.add_node("build_evidence", recorded("build_evidence", build_evidence))
    graph.add_node("eight_principles", recorded("eight_principles", analyze_principles))
    graph.add_node("channel_candidates", recorded("channel_candidates", generate_channels))
    graph.add_node("syndrome_relationship", recorded("syndrome_relationship", judge_relationship))
    graph.add_node("formula_candidates", recorded("formula_candidates", generate_formulas))
    graph.add_node("differential", recorded("differential", compare_formulas))
    graph.add_node("evidence_gap", recorded("evidence_gap", analyze_gaps))
    graph.add_node("topk", recorded("topk", construct_paths))
    graph.add_node("similar_cases", recorded("similar_cases", similar_cases))
    # 固定协议顺序：关系在方证之前，鉴别之后再分析证据缺口。
    stages = STAGES
    # 入口仅接受固定阶段枚举；服务已核对事实、配置和前序来源，不能由模型随意跳算。
    graph.add_conditional_edges(
        START,
        lambda state: state.get("from_stage", "build_evidence"),
        {stage: stage for stage in stages},
    )
    for current, following in zip(stages[:-1], stages[1:], strict=True):
        graph.add_edge(current, following)
    graph.add_edge(stages[-1], END)
    return graph.compile(checkpointer=checkpointer)
