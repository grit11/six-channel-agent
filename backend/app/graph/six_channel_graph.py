"""六经分析子图编排：按协议连接节点，checkpoint 由生命周期注入，不保存客户端或隐藏推理。"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import perf_counter
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter

from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.run_cache import cache_metrics
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.channel_candidate import generate_channel_candidates
from backend.app.nodes.clinical_parser import parse_clinical_text, parse_clinical_with_llm
from backend.app.nodes.differential_reasoner import compare_formula_candidates
from backend.app.nodes.eight_principles import analyze_eight_principles
from backend.app.nodes.evidence_gap import build_evidence_gaps, decide_evidence_gap_stop
from backend.app.nodes.formula_candidate import generate_formula_candidates
from backend.app.nodes.path_summary import summarize_completeness
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

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


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
    prompt_digest: str
    reset_clinical: bool


STAGE_VERSION = "stage-replay/v1.2-six-stage"
STAGES: list[ReasoningStage] = [
    "clinical_parser",
    "six_channel_screen",
    "formula_branch_retrieval",
    "differential_verification",
    "evidence_topk",
]


def stage_seed(result: ReasoningResponse, from_stage: ReasoningStage) -> SixChannelState:
    """只恢复真实五阶段前序字段；历史九阶段只读，拒绝重新包装旧结论。"""
    preceding = STAGES[: STAGES.index(from_stage)]
    snapshots = result.stage_snapshots[: len(preceding)]
    if [item.stage for item in snapshots] != preceding or any(
        item.version != STAGE_VERSION or item.sequence != index + 1 or not item.input_digest
        for index, item in enumerate(snapshots)
    ):
        raise InvalidBusinessRequestError("前序阶段快照不完整或版本过期，请从临床解析重算")
    seed: SixChannelState = {
        "clinical_evidence": result.clinical_evidence.model_copy(deep=True),
        "completed_stages": list(preceding),
        "stage_snapshots": [item.model_copy(deep=True) for item in snapshots],
        "from_stage": from_stage,
    }
    if snapshots:
        seed["prompt_digest"] = str(snapshots[0].payload.get("prompt_digest", ""))
    if len(preceding) >= 2:
        seed["eight_principles"] = result.eight_principles.model_copy(deep=True)
        seed["channel_candidates"] = [item.model_copy(deep=True) for item in result.candidates]
    if len(preceding) >= 3:
        seed["formula_candidates"] = [
            item.model_copy(deep=True) for item in result.formula_candidates
        ]
    if len(preceding) >= 4:
        seed["syndrome_relationships"] = [
            item.model_copy(deep=True) for item in result.syndrome_relationships
        ]
        seed["differential_results"] = [
            item.model_copy(deep=True) for item in result.differential_results
        ]
    return seed


def response_from_state(state: SixChannelState, thread_id: str) -> ReasoningResponse:
    """统一装配原公开 DTO；人工审核独立，历史医案不自动召回。"""
    return ReasoningResponse(
        thread_id=thread_id,
        clinical_evidence=state["clinical_evidence"],
        eight_principles=state["eight_principles"],
        candidates=state["channel_candidates"],
        syndrome_relationships=state["syndrome_relationships"],
        formula_candidates=state["formula_candidates"],
        differential_results=state["differential_results"],
        evidence_gaps=state["evidence_gaps"],
        evidence_gap_decision=state["evidence_gap_decision"],
        reasoning_paths=state["reasoning_paths"],
        excluded_paths=state.get("excluded_paths", []),
        evidence_catalog=state["evidence_catalog"],
        completed_stages=state["completed_stages"],
        similar_cases=[],
        stage_snapshots=state.get("stage_snapshots", []),
    )


def build_six_channel_graph(
    knowledge: KnowledgeService,
    checkpointer: BaseCheckpointSaver | Literal[False] | None = None,
    *,
    channel_weight: float = 0.6,
    llm: StructuredLLM | None = None,
    reasoning_llm: StructuredLLM | None = None,
    scoring_policy: ScoringPolicy | None = None,
    evidence_gap_policy: EvidenceGapPolicy | None = None,
    formula_branch_recall_top_n: int = 12,
    max_supplement_rounds: int = 3,
) -> CompiledStateGraph:
    """编译应用级子图，知识端口放在节点闭包，不进入 checkpoint State。"""

    async def build_evidence(state: SixChannelState) -> SixChannelState:
        """优先使用医生结构化输入，禁止原文解析覆盖编辑事实。"""
        evidence = None if state.get("reset_clinical") else state.get("clinical_evidence")
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
        prompt_paths = [
            _PROMPT_DIR / name for name in ("clinical_parser.md", "channel_candidate.md")
        ]
        texts = await asyncio.gather(*(asyncio.to_thread(path.read_bytes) for path in prompt_paths))
        return {
            "clinical_evidence": evidence,
            "completed_stages": ["clinical_parser"],
            "prompt_digest": hashlib.sha256(b"".join(texts)).hexdigest(),
        }

    async def six_channel_screen(state: SixChannelState) -> SixChannelState:
        """S2 内部确定性八纲加一次宽候选模型；允许全部六经，随后查证。"""
        principles = analyze_eight_principles(state["clinical_evidence"])
        return {
            "eight_principles": principles,
            "channel_candidates": await generate_channel_candidates(
                state["clinical_evidence"], principles, knowledge, top_k=6, llm=reasoning_llm
            ),
            "completed_stages": state["completed_stages"] + ["six_channel_screen"],
        }

    async def generate_formulas(state: SixChannelState) -> SixChannelState:
        """召回来源分层的方证候选，不输出处方或剂量。"""
        return {
            "formula_candidates": await generate_formula_candidates(
                state["clinical_evidence"],
                state["channel_candidates"],
                knowledge,
                top_k=formula_branch_recall_top_n,
            ),
            "completed_stages": state["completed_stages"] + ["formula_branch_retrieval"],
        }

    async def compare_formulas(state: SixChannelState) -> SixChannelState:
        """应用来源鉴别规则，保留未解决维度。"""
        return {
            "syndrome_relationships": await judge_syndrome_relationships(
                state["clinical_evidence"], state["channel_candidates"], knowledge
            ),
            "differential_results": await compare_formula_candidates(
                state["clinical_evidence"], state["formula_candidates"], knowledge
            ),
            "completed_stages": state["completed_stages"] + ["differential_verification"],
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
            verified_comparisons=True,
        )
        gaps = await build_evidence_gaps(
            state["differential_results"],
            state["clinical_evidence"],
            knowledge,
            context=state.get("evidence_gap_context"),
            policy=evidence_gap_policy,
        )
        pending = 0
        for gap in gaps:
            if gap.status == "pending":
                pending += 1
                if pending > 3:
                    gap.status = "deferred"

        # 排名已经确定；补问仅改变完整度展示与人工停止决策，不反向改变分数。
        for path in [*paths, *excluded]:
            relation = next(
                (
                    item
                    for item in state["syndrome_relationships"]
                    if item.relationship_type == path.relationship
                ),
                None,
            )
            path.completeness_detail = summarize_completeness(
                path, catalog, gaps, relation.missing_timeline_evidence if relation else []
            )
        return {
            "evidence_gaps": gaps,
            "similar_cases": [],
            "reasoning_paths": paths,
            "evidence_gap_decision": decide_evidence_gap_stop(
                gaps,
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
            "completed_stages": state["completed_stages"] + ["evidence_topk"],
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
                    "stage_completed",
                    {"duration_ms": int((perf_counter() - started) * 1000), **cache_metrics()},
                )
            previous = [] if stage == "clinical_parser" else state.get("stage_snapshots", [])
            payload = TypeAdapter(dict[str, object]).dump_python(
                {
                    key: value
                    for key, value in update.items()
                    if key not in {"completed_stages", "stage_snapshots", "evidence_catalog"}
                },
                mode="json",
            )
            if "evidence_catalog" in update:
                payload["evidence_catalog_ids"] = [
                    item.evidence_id for item in update["evidence_catalog"]
                ]
            return {
                **update,
                "stage_snapshots": [
                    *previous,
                    ReasoningStageSnapshot(
                        sequence=len(previous) + 1,
                        stage=stage,
                        payload=payload,
                        version=STAGE_VERSION,
                        input_digest=fingerprint(state | update),
                    ),
                ],
            }

        return RunnableLambda(run)

    def fingerprint(state: SixChannelState) -> str:
        """绑定事实、知识、Prompt 和评分配置；改动任一输入都不可复用旧结论。"""
        return hashlib.sha256(
            json.dumps(
                {
                    "facts": state["clinical_evidence"].model_dump(mode="json"),
                    "knowledge": knowledge.snapshot_digest,
                    "branches": knowledge.branch_digest,
                    "recall_top_n": formula_branch_recall_top_n,
                    "workflow": STAGE_VERSION,
                    "prompt_digest": state.get("prompt_digest", ""),
                    "scoring": (scoring_policy or ScoringPolicy()).model_dump(mode="json"),
                    "gap_policy": (evidence_gap_policy or EvidenceGapPolicy()).model_dump(
                        mode="json"
                    ),
                    "channel_weight": channel_weight,
                    "max_rounds": max_supplement_rounds,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    async def entry(state: SixChannelState) -> str:
        """重算入口检验全部前序快照，旧数据不能直接恢复新图。"""
        stage = state.get("from_stage", "clinical_parser")
        if stage != "clinical_parser":
            paths = [_PROMPT_DIR / name for name in ("clinical_parser.md", "channel_candidate.md")]
            texts = await asyncio.gather(*(asyncio.to_thread(path.read_bytes) for path in paths))
            if hashlib.sha256(b"".join(texts)).hexdigest() != state.get("prompt_digest"):
                raise InvalidBusinessRequestError("Prompt 已变化，请从临床解析重算")
            snapshots = state.get("stage_snapshots", [])
            preceding = STAGES[: STAGES.index(stage)]
            if [item.stage for item in snapshots] != preceding or any(
                item.version != STAGE_VERSION or item.input_digest != fingerprint(state)
                for item in snapshots
            ):
                raise InvalidBusinessRequestError("事实或配置已变化，请从临床解析重算")
        return stage

    graph = StateGraph(SixChannelState)
    node_specs = [
        ("clinical_parser", build_evidence),
        ("six_channel_screen", six_channel_screen),
        ("formula_branch_retrieval", generate_formulas),
        ("differential_verification", compare_formulas),
        ("evidence_topk", construct_paths),
    ]
    for stage, implementation in node_specs:
        graph.add_node(stage, recorded(stage, implementation))
    graph.add_conditional_edges(START, entry, {stage: stage for stage in STAGES})
    for current, following in zip(STAGES[:-1], STAGES[1:], strict=True):
        graph.add_edge(current, following)
    graph.add_edge(STAGES[-1], END)
    return graph.compile(checkpointer=checkpointer)
