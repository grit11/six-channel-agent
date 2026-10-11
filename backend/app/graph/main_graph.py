"""会诊主图：编排六经分析、补证/审核暂停、同线程循环与终结。

临床和 QA 采用分别编译的状态图及显式意图入口，QA 不接临床 saver；Safety 待 S6。
医学规则由 SixChannelAgent 执行；主图只保存明确控制状态，不读取业务数据库。
补证合并、身份核验、审核合法性和审计落库由应用服务执行，不能交给模型决定路由。
"""

from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.agents.six_channel_agent import SixChannelAgent
from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.consultation import ReasoningStage
from backend.app.schemas.knowledge import KnowledgeQARequest
from backend.app.schemas.reasoning import EvidenceGapContext
from backend.app.schemas.reasoning_api import ReasoningResponse


def route_main_intent(intent: str) -> Literal["clinical", "knowledge"]:
    """由医生选择的应用入口路由，不把病例或知识问题中的自由文本当作控制指令。"""
    if intent not in {"clinical", "knowledge"}:
        raise InvalidBusinessRequestError("不支持的主图意图")
    return "knowledge" if intent == "knowledge" else "clinical"


class KnowledgeGraphState(TypedDict):
    """QA 专用主图状态；不存在临床事实、审核、会诊 ID 或 clinical thread。"""

    intent: Literal["knowledge"]
    request: dict[str, object]
    answer: NotRequired[dict[str, object]]


def build_knowledge_main_graph(agent: KnowledgeQAAgent) -> CompiledStateGraph:
    """编译独立 QA 分支，无 checkpoint、临床上下文或医生审核循环。"""

    def route(state: KnowledgeGraphState) -> str:
        """固定意图路由，错误临床意图不会启动临床节点。"""
        return route_main_intent(state["intent"])

    async def answer(state: KnowledgeGraphState) -> dict[str, object]:
        """只更新 QA 回答，不将问题转换为患者事实。"""
        result = await agent.answer(KnowledgeQARequest.model_validate(state["request"]))
        return {"answer": result.model_dump(mode="json")}

    graph = StateGraph(KnowledgeGraphState)
    graph.add_node("knowledge_qa", answer)
    graph.add_conditional_edges(START, route, {"knowledge": "knowledge_qa"})
    graph.add_edge("knowledge_qa", END)
    return graph.compile(checkpointer=False)


class ConsultationRoute(StrEnum):
    """人工流程的受控路由，模型不能生成自由节点名。"""

    REASON = "reason"
    WAIT = "wait"
    FINISH = "finish"


class MainGraphState(TypedDict, total=False):
    """主图可序列化状态；完整快照和人工审核仍由 Repository 持有。"""

    thread_id: str
    intent: Literal["clinical"]
    from_stage: ReasoningStage
    previous_result: dict[str, object] | None
    clinical_evidence: dict[str, object]
    top_k: int
    supplementation_round: int
    evidence_gap_context: dict[str, object]
    result: dict[str, object] | None
    phase: Literal["supplement", "review", "completed", "cancelled"]
    route: str
    review: dict[str, object] | None
    # 每次暂停生成新序号，允许辨认同线程的不同人工暂停点。
    waiting_revision: int
    run_turn: int
    result_turn: int
    action_id: str
    action_turn: int


class HumanGraphCommand(BaseModel):
    """服务传入的已校验操作；不接受模型、浏览器或数据库自由节点指令。"""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["reason", "save_evidence", "review_ready", "review", "finish", "cancel"]
    clinical_evidence: ClinicalEvidence | None = None
    top_k: int | None = Field(default=None, ge=1, le=5)
    supplementation_round: int | None = Field(default=None, ge=0)
    evidence_gap_context: EvidenceGapContext | None = None
    review: dict[str, object] | None = None
    from_stage: ReasoningStage = "build_evidence"
    previous_result: dict[str, object] | None = None
    run_turn: int | None = None
    action_id: str | None = None
    action_turn: int | None = None


def build_main_graph(
    agent: SixChannelAgent,
    checkpointer: BaseCheckpointSaver,
    *,
    max_supplement_rounds: int,
) -> CompiledStateGraph:
    """一次编译主图；显式 interrupt 由 Command resume 恢复，子图不持有独立 saver。"""

    async def reason(state: MainGraphState) -> MainGraphState:
        """运行完整分析并根据缺口/轮数选择人工暂停类型，不自动采用路径。"""
        result = await agent.analyze(
            ClinicalEvidence.model_validate(state["clinical_evidence"]),
            thread_id=state["thread_id"],
            top_k=state["top_k"],
            from_stage=state.get("from_stage", "build_evidence"),
            previous_result=(
                ReasoningResponse.model_validate(state["previous_result"])
                if state.get("previous_result")
                else None
            ),
            evidence_gap_context=EvidenceGapContext.model_validate(
                state.get("evidence_gap_context", {})
            ).model_copy(update={"supplementation_round": state.get("supplementation_round", 0)}),
        )
        if (
            result.evidence_gap_decision is not None
            and state.get("supplementation_round", 0) >= max_supplement_rounds
        ):
            result.evidence_gap_decision = result.evidence_gap_decision.model_copy(
                update={"stop": True, "reason": "max_rounds"}
            )
        phase: Literal["supplement", "review"] = (
            "supplement"
            if result.evidence_gap_decision is not None and not result.evidence_gap_decision.stop
            else "review"
        )
        return {
            "result": result.model_dump(mode="json"),
            "phase": phase,
            "route": ConsultationRoute.WAIT.value,
            "result_turn": state.get("run_turn", 0),
        }

    def wait_for_doctor(state: MainGraphState) -> MainGraphState:
        """暂停之前只构造问题摘要；恢复时不会重复执行医学子图。"""
        revision = state.get("waiting_revision", 0) + 1
        result = ReasoningResponse.model_validate(state["result"]) if state.get("result") else None
        answer = HumanGraphCommand.model_validate(
            interrupt(
                {
                    "phase": state["phase"],
                    "waiting_revision": revision,
                    "questions": (
                        [item.question for item in result.evidence_gaps if item.status == "pending"]
                        if result
                        else []
                    ),
                }
            )
        )
        update: MainGraphState = {
            "waiting_revision": revision,
            "route": ConsultationRoute.WAIT.value,
        }
        if answer.action_id:
            update["action_id"] = answer.action_id
            update["action_turn"] = answer.action_turn or 0
        if answer.clinical_evidence is not None:
            update["clinical_evidence"] = answer.clinical_evidence.model_dump(mode="json")
        if answer.top_k is not None:
            update["top_k"] = answer.top_k
        if answer.supplementation_round is not None:
            update["supplementation_round"] = answer.supplementation_round
        if answer.evidence_gap_context is not None:
            update["evidence_gap_context"] = answer.evidence_gap_context.model_dump(mode="json")
        if answer.kind == "reason":
            update["run_turn"] = answer.run_turn or 0
            update["from_stage"] = answer.from_stage
            update["previous_result"] = answer.previous_result
            update["result"] = None
            update["route"] = ConsultationRoute.REASON.value
        elif answer.kind == "save_evidence":
            update["result"] = None
            update["phase"] = "supplement"
        elif answer.kind == "review_ready":
            update["phase"] = "review"
            if result and result.evidence_gap_decision:
                result.evidence_gap_decision = result.evidence_gap_decision.model_copy(
                    update={"stop": True, "reason": "doctor_current_information"}
                )
                update["result"] = result.model_dump(mode="json")
        elif answer.kind == "review":
            update["review"] = answer.review
            update["phase"] = (
                "supplement"
                if answer.review and answer.review.get("action") == "add_evidence"
                else state["phase"]
            )
        elif answer.kind in {"finish", "cancel"}:
            update["route"] = ConsultationRoute.FINISH.value
            update["phase"] = "completed" if answer.kind == "finish" else "cancelled"
            update["review"] = answer.review
        return update

    def route(state: MainGraphState) -> str:
        """使用固定枚举映射，不把自由文本当作节点名。"""
        return ConsultationRoute(state["route"]).value

    graph = StateGraph(MainGraphState)
    graph.add_node("reason", reason)
    graph.add_node("doctor_wait", wait_for_doctor)
    # 临床服务使用已校验意图；QA 用另一个 TypedDict 编译，不能将两种状态混入同一 saver。
    graph.add_conditional_edges(
        START,
        lambda state: route_main_intent(state.get("intent", "clinical")),
        {"clinical": "reason"},
    )
    graph.add_edge("reason", "doctor_wait")
    graph.add_conditional_edges(
        "doctor_wait",
        route,
        {
            ConsultationRoute.REASON.value: "reason",
            ConsultationRoute.WAIT.value: "doctor_wait",
            ConsultationRoute.FINISH.value: END,
        },
    )
    return graph.compile(checkpointer=checkpointer)
