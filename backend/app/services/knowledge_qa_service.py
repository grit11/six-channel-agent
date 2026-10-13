"""独立知识问答应用用例，装配 QA 专用主图入口及精确来源展开。

仅依赖 KnowledgeService/KnowledgeQAAgent，不持有会诊 Repository 或临床图状态。
问题每次以新状态调用无 checkpoint 的 QA 图，不允许浏览器选择临床 thread。
"""

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.exceptions import KnowledgeNotFoundError
from backend.app.graph.knowledge_graph import KnowledgeGraphState, build_knowledge_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import KnowledgeQARequest, KnowledgeQAResponse


class KnowledgeQAService:
    """独立查询和来源展开服务，QA 不写会诊或临床 checkpoint。"""

    def __init__(self, knowledge: KnowledgeService, agent: KnowledgeQAAgent) -> None:
        """复用应用级知识边界并编译无临床状态的主图分支。"""
        self.knowledge = knowledge
        self.graph = build_knowledge_graph(agent)

    async def ask(self, request: KnowledgeQARequest) -> KnowledgeQAResponse:
        """每个知识问题独立执行，主动 context_text 也不会解析为患者事实。"""
        state: KnowledgeGraphState = {
            "intent": "knowledge",
            "request": request.model_dump(mode="json"),
        }
        result = await self.graph.ainvoke(state)
        return KnowledgeQAResponse.model_validate(result["answer"])

    async def get_clause(self, clause_id: str) -> Evidence:
        """精确读取条文，未命中统一返回资源不存在，不转向模糊检索。"""
        evidence = await self.knowledge.get_clause_by_id(clause_id)
        if evidence is None:
            raise KnowledgeNotFoundError("未找到指定条文")
        return evidence

    async def get_formula(self, formula_name: str) -> Evidence:
        """精确读取完整方名，风险和来源原样返回，不自动生成剂量。"""
        evidence = await self.knowledge.get_formula_by_name(formula_name)
        if evidence is None:
            raise KnowledgeNotFoundError("未找到指定方名")
        return evidence
