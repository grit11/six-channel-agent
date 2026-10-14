"""唯一 V2 临床图：五个业务阶段和持久澄清门，合计六个技术节点。

只使用公开 DTO；Checkpointer 由 lifespan 注入，人工恢复不执行医学规则。
"""

from collections.abc import Awaitable, Callable
from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from backend.app.clinical_v2.channel_localization import localize_channels
from backend.app.clinical_v2.comparison import compare_formula_candidates
from backend.app.clinical_v2.extraction import extract_clinical_facts
from backend.app.clinical_v2.retrieval import (
    build_query_plan,
    retrieve_formula_candidates,
)
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonV2,
    HumanGraphCommandV2,
    RetrievalBundleV2,
)
from backend.app.core.exceptions import InvalidStateTransitionError
from backend.app.core.ports import S4StreamPolicy, StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.observability.progress import emit_progress, stage_scope


class ClinicalV2GraphState(TypedDict, total=False):
    """可序列化公开图状态，不存客户端、回调或模型隐藏思维。"""

    consultation_id: str
    clinical_text: str
    clinical_evidence: dict[str, object] | None
    localization: dict[str, object] | None
    retrieval: dict[str, object] | None
    comparison: dict[str, object] | None
    revision: int
    clarification_rounds: int
    status: str
    paused_at: str | None
    route: str
    action_id: str
    action_turn: int
    review: dict[str, object] | None


def build_clinical_v2_graph(
    llm: StructuredLLM,
    checkpointer: BaseCheckpointSaver,
    *,
    knowledge: KnowledgeService | None = None,
    max_clarification_rounds: int = 3,
    retrieval_top_k: int = 20,
    differential_top_k: int = 10,
    s4_policy: S4StreamPolicy | None = None,
) -> CompiledStateGraph:
    """构建完整五阶段与人工中断；未接检索时显式未决，绝不伪造方证。"""

    async def s1_extract(state: ClinicalV2GraphState) -> dict[str, object]:
        """新文本必须重抽；已审核事实可直接使用，后序状态清空。"""
        async with stage_scope("s1_extract"):
            await emit_progress("stage_started", {})
            facts = (
                ClinicalEvidenceV2.model_validate(state["clinical_evidence"])
                if state.get("clinical_evidence")
                else await extract_clinical_facts(
                    state["clinical_text"], llm, revision=state.get("revision", 0)
                )
            )
            await emit_progress("stage_completed", {"fact_count": len(facts.findings)})
        return {
            "clinical_evidence": facts.model_dump(mode="json"),
            "localization": None,
            "retrieval": None,
            "comparison": None,
            "status": "running",
            "paused_at": None,
        }

    async def s2_localize(state: ClinicalV2GraphState) -> dict[str, object]:
        """只调用定位模型，unknown 不执行后序检索。"""
        async with stage_scope("s2_localize"):
            await emit_progress("stage_started", {})
            result = await localize_channels(
                ClinicalEvidenceV2.model_validate(state["clinical_evidence"]), llm
            )
            await emit_progress("stage_completed", {"localization_status": result.status})
        proceed = result.proceed_to_retrieval
        return {
            "localization": result.model_dump(mode="json"),
            "status": "running" if proceed else "needs_evidence",
            "paused_at": None if proceed else "s2_clarification",
        }

    async def s3_retrieve(state: ClinicalV2GraphState) -> dict[str, object]:
        """S3 只读取真实 KnowledgeService；未装配知识时明确无证据。"""
        if knowledge is None:
            bundle = RetrievalBundleV2(
                query_plan=build_query_plan(
                    ClinicalEvidenceV2.model_validate(state["clinical_evidence"]),
                    ChannelLocalizationV2.model_validate(state["localization"]),
                ),
                formula_candidates=[],
                retrieved_evidence_ids=[],
                retrieval_status="no_relevant_evidence",
            )
        else:
            evidence = ClinicalEvidenceV2.model_validate(state["clinical_evidence"])
            localization = ChannelLocalizationV2.model_validate(state["localization"])
            bundle = await retrieve_formula_candidates(
                knowledge, build_query_plan(evidence, localization), recall_count=retrieval_top_k
            )
        return {"retrieval": bundle.model_dump(mode="json"), "status": "running"}

    async def s4_compare(state: ClinicalV2GraphState) -> dict[str, object]:
        """S4 只在 S3 有候选时调用模型，空池保留 insufficient。"""
        retrieval = RetrievalBundleV2.model_validate(state["retrieval"])
        if not retrieval.formula_candidates or knowledge is None:
            comparison = ComparisonV2(
                status="insufficient",
                ordered_paths=[],
                unresolved_reason="无可比较来源",
            )
        else:
            comparison = await compare_formula_candidates(
                ClinicalEvidenceV2.model_validate(state["clinical_evidence"]),
                ChannelLocalizationV2.model_validate(state["localization"]),
                retrieval,
                knowledge,
                llm,
                differential_top_k=differential_top_k,
                s4_policy=s4_policy,
            )
        return {
            "comparison": comparison.model_dump(mode="json"),
            "status": "awaiting_doctor_review",
            "paused_at": "s5_doctor_review",
        }

    def s5_doctor_review(state: ClinicalV2GraphState) -> dict[str, object]:
        """S5 真实人工中断；模型不能自行签发 completed。"""
        command = HumanGraphCommandV2.model_validate(
            interrupt(
                {
                    "paused_at": "s5_doctor_review",
                    "revision": state.get("revision", 0),
                    "comparison": state.get("comparison"),
                }
            )
        )
        if command.consultation_id != state["consultation_id"]:
            raise InvalidStateTransitionError("人工审核会诊身份不匹配")
        if command.revision != state.get("revision", 0):
            raise InvalidStateTransitionError("STALE_REVISION")
        if command.kind == "cancel":
            return {
                "status": "cancelled",
                "paused_at": None,
                "route": "end",
                "action_id": command.action_id,
            }
        if command.kind != "review":
            raise InvalidStateTransitionError("当前暂停点只允许医生审核")
        review = command.review or {}
        action = review.get("action")
        if action not in {"accept", "modify", "reject", "reorder", "note", "add_evidence"}:
            raise InvalidStateTransitionError("医生审核动作无效")
        if action == "add_evidence":
            if not command.clinical_text and command.clinical_evidence is None:
                raise InvalidStateTransitionError("补证没有新事实")
            return {
                "status": "needs_evidence",
                "paused_at": None,
                "route": "s1_extract",
                "clinical_text": command.clinical_text or state["clinical_text"],
                "clinical_evidence": (
                    command.clinical_evidence.model_dump(mode="json")
                    if command.clinical_evidence
                    else None
                ),
                "revision": command.revision + 1,
                "action_id": command.action_id,
                "retrieval": None,
                "comparison": None,
            }
        return {
            "status": "completed",
            "paused_at": None,
            "route": "end",
            "review": review,
            "action_id": command.action_id,
            "action_turn": command.action_turn,
        }

    def s2_clarification_gate(state: ClinicalV2GraphState) -> dict[str, object]:
        """真实 interrupt；恢复前无业务副作用，补证触发新版本并清空后序结果。"""
        localization = ChannelLocalizationV2.model_validate(state["localization"])
        if state.get("clarification_rounds", 0) >= max_clarification_rounds:
            return {"status": "undetermined", "paused_at": None, "route": "end"}
        command = HumanGraphCommandV2.model_validate(
            interrupt(
                {
                    "paused_at": "s2_clarification",
                    "revision": state.get("revision", 0),
                    "questions": [
                        q.model_dump(mode="json") for q in localization.clarification_questions
                    ],
                }
            )
        )
        if command.consultation_id != state["consultation_id"]:
            raise InvalidStateTransitionError("人工动作会诊身份不匹配")
        if command.revision != state.get("revision", 0):
            raise InvalidStateTransitionError("STALE_REVISION")
        if command.kind == "cancel":
            return {
                "status": "cancelled",
                "paused_at": None,
                "route": "end",
                "action_id": command.action_id,
            }
        if command.kind != "clarify":
            raise InvalidStateTransitionError("当前暂停点只允许六经澄清")
        if command.no_additional_information:
            return {
                "status": "undetermined",
                "paused_at": None,
                "route": "end",
                "action_id": command.action_id,
            }
        if not command.clinical_text and command.clinical_evidence is None:
            raise InvalidStateTransitionError("补证没有新事实")
        return {
            "clinical_text": command.clinical_text or state["clinical_text"],
            "clinical_evidence": (
                command.clinical_evidence.model_dump(mode="json")
                if command.clinical_evidence
                else None
            ),
            "revision": command.revision + 1,
            "clarification_rounds": state.get("clarification_rounds", 0) + 1,
            "route": "s1_extract",
            "retrieval": None,
            "comparison": None,
            "action_id": command.action_id,
            "action_turn": command.action_turn,
        }

    graph = StateGraph(ClinicalV2GraphState)
    graph.add_node("s1_extract", s1_extract)
    graph.add_node("s2_localize", s2_localize)
    graph.add_node("s2_clarification_gate", s2_clarification_gate)
    graph.add_edge(START, "s1_extract")
    graph.add_edge("s1_extract", "s2_localize")
    graph.add_conditional_edges(
        "s2_localize",
        lambda state: ("continue" if state["localization"]["proceed_to_retrieval"] else "clarify"),
        {"continue": "s3_retrieve", "clarify": "s2_clarification_gate"},
    )
    graph.add_conditional_edges(
        "s2_clarification_gate",
        lambda state: state["route"],
        {"s1_extract": "s1_extract", "end": END},
    )

    def observed_node(
        stage: str,
        node: Callable[[ClinicalV2GraphState], Awaitable[dict[str, object]]],
    ) -> Callable[[ClinicalV2GraphState], Awaitable[dict[str, object]]]:
        """统一标记真实节点边界，失败时不发布完成事件。"""

        async def run(state: ClinicalV2GraphState) -> dict[str, object]:
            """阶段上下文使检索和模型 usage 归属于当前公开阶段。"""
            async with stage_scope(stage):
                await emit_progress("stage_started", {})
                result = await node(state)
                await emit_progress("stage_completed", {})
                return result

        return run

    graph.add_node("s3_retrieve", RunnableLambda(observed_node("s3_retrieve", s3_retrieve)))
    graph.add_node("s4_compare", RunnableLambda(observed_node("s4_compare", s4_compare)))
    graph.add_node("s5_doctor_review", s5_doctor_review)
    graph.add_edge("s3_retrieve", "s4_compare")
    graph.add_edge("s4_compare", "s5_doctor_review")
    graph.add_conditional_edges(
        "s5_doctor_review",
        lambda state: state["route"],
        {"s1_extract": "s1_extract", "end": END},
    )
    return graph.compile(checkpointer=checkpointer)
