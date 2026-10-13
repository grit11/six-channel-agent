"""独立知识问答图：从临床 V2 Graph 拆出，无会诊事实、审核或临床 Checkpoint。

该图只调用 KnowledgeQAAgent，避免删除旧混合主图时让知识问答间接依赖旧医学推理。
"""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.exceptions import InvalidBusinessRequestError
from backend.app.schemas.knowledge import KnowledgeQARequest


class KnowledgeGraphState(TypedDict):
    """QA 专用状态，不含患者事实、会诊 ID 或临床 thread。"""

    intent: str
    request: dict[str, object]
    answer: NotRequired[dict[str, object]]


def build_knowledge_graph(agent: KnowledgeQAAgent) -> CompiledStateGraph:
    """编译无 checkpoint 的独立 QA 图。"""

    async def answer(state: KnowledgeGraphState) -> dict[str, object]:
        """只更新来源问答，不把问题解释为患者事实。"""
        if state.get("intent") != "knowledge":
            raise InvalidBusinessRequestError("知识问答图收到非法意图")
        result = await agent.answer(KnowledgeQARequest.model_validate(state["request"]))
        return {"answer": result.model_dump(mode="json")}

    graph = StateGraph(KnowledgeGraphState)
    graph.add_node("knowledge_qa", answer)
    graph.add_edge(START, "knowledge_qa")
    graph.add_edge("knowledge_qa", END)
    return graph.compile(checkpointer=False)
