"""知识查询 API：只转发 KnowledgeService 结果，不在 Router 编写医学规则或直接访问 Milvus。"""

from fastapi import APIRouter, Request

from backend.app.core.exceptions import KnowledgeNotFoundError, KnowledgeRetrievalError
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.knowledge import (
    CaseSearchRequest,
    ClassicSearchRequest,
    CoreSearchRequest,
    DifferentialSearchRequest,
    FormulaSearchRequest,
    KnowledgeQARequest,
    KnowledgeQAResponse,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/evidence/{evidence_id}", response_model=Evidence)
async def get_evidence(request: Request, evidence_id: str) -> Evidence:
    """精确回查临床引用，不用搜索命中替代不存在的来源。"""
    try:
        items = await request.app.state.container.knowledge_service.get_evidence_by_ids(
            [evidence_id]
        )
    except KnowledgeRetrievalError as exc:
        raise KnowledgeNotFoundError("精确来源不存在于当前知识快照") from exc
    return items[0]


@router.post("/qa", response_model=KnowledgeQAResponse)
async def ask_knowledge(request: Request, payload: KnowledgeQARequest) -> KnowledgeQAResponse:
    """独立知识问答，不读取或修改临床会诊状态。"""
    return await request.app.state.container.knowledge_qa_service.ask(payload)


@router.get("/clauses/{clause_id}", response_model=Evidence)
async def get_clause(request: Request, clause_id: str) -> Evidence:
    """精确展开条文，无向量检索或生成模型参与。"""
    return await request.app.state.container.knowledge_qa_service.get_clause(clause_id)


@router.get("/formulas/{formula_name}", response_model=Evidence)
async def get_formula(request: Request, formula_name: str) -> Evidence:
    """按完整方名展开真实方证与风险。"""
    return await request.app.state.container.knowledge_qa_service.get_formula(formula_name)


@router.post("/framework", response_model=list[Evidence])
@router.post("/search/framework", response_model=list[Evidence])
async def search_framework(request: Request, payload: CoreSearchRequest) -> list[Evidence]:
    """查询六经框架来源，并保持 Evidence 层级和来源字段。"""
    return await request.app.state.container.knowledge_service.search_core_framework(
        payload.query, payload.top_k
    )


@router.post("/classic", response_model=list[Evidence])
@router.post("/search/classic", response_model=list[Evidence])
async def search_classic(request: Request, payload: ClassicSearchRequest) -> list[Evidence]:
    """查询经典条文，精确条文号优先由 KnowledgeService 处理。"""
    return await request.app.state.container.knowledge_service.search_classic(
        payload.query, payload.channels, payload.clause_nos, payload.top_k
    )


@router.post("/formula", response_model=list[Evidence])
@router.post("/search/formula", response_model=list[Evidence])
async def search_formula(request: Request, payload: FormulaSearchRequest) -> list[Evidence]:
    """查询方证，保留 canonical 与 case-derived 的来源范围。"""
    return await request.app.state.container.knowledge_service.search_formula(
        payload.query, payload.channels, payload.association_scope, payload.top_k
    )


@router.post("/differential", response_model=list[Evidence])
@router.post("/search/differential", response_model=list[Evidence])
async def search_differential(
    request: Request, payload: DifferentialSearchRequest
) -> list[Evidence]:
    """查询原始鉴别来源，不由 API 层生成或改写临床结论。"""
    return await request.app.state.container.knowledge_service.search_differential(
        payload.candidates, payload.query, payload.top_k
    )


@router.post("/case", response_model=list[Evidence])
@router.post("/search/case", response_model=list[Evidence])
async def search_case(request: Request, payload: CaseSearchRequest) -> list[Evidence]:
    """查询历史医案，仅返回 L5 Evidence，不提升为 canonical 规则。"""
    return await request.app.state.container.knowledge_service.search_case(
        payload.query, payload.channels, payload.top_k
    )
