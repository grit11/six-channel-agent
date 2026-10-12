"""应用容器工厂，集中装配基础设施并按相反顺序关闭，不承载医学规则。"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.core.config import Settings
from backend.app.core.exceptions import KnowledgeRetrievalError, PersistenceError
from backend.app.core.llm import HTTPStructuredLLM, validate_llm_config
from backend.app.core.logging import log_backend_event
from backend.app.core.ports import EmbeddingProvider, StructuredLLM
from backend.app.graph.graph_factory import build_consultation_workflow
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.checkpoint import (
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.persistence.database import DatabaseRuntime, open_database
from backend.app.persistence.repositories import (
    ConsultationRepository,
    DisabledConsultationRepository,
    InMemoryConsultationRepository,
    PostgresConsultationRepository,
)
from backend.app.rag.embedding import HTTPEmbeddingProvider, validate_embedding_config
from backend.app.rag.reranker import HTTPReranker, Reranker, validate_reranker_config
from backend.app.rag.vector_store import MilvusVectorStore, build_milvus_store
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.knowledge_qa_service import KnowledgeQAService
from backend.app.services.reasoning_service import ReasoningService
from backend.app.services.review_service import ReviewService
from backend.app.services.stream_service import StreamService


@dataclass
class ApplicationContainer:
    """应用生命周期依赖，不保存到 LangGraph State。"""

    settings: Settings
    vector_store: MilvusVectorStore
    checkpoint_runtime: CheckpointRuntime
    database: DatabaseRuntime
    knowledge_service: KnowledgeService
    reasoning_service: ReasoningService
    consultation_service: ConsultationService
    review_service: ReviewService
    stream_service: StreamService
    knowledge_qa_service: KnowledgeQAService


@asynccontextmanager
async def open_llm(settings: Settings) -> AsyncIterator[StructuredLLM]:
    """在应用级生命周期创建模型客户端，密钥只进入请求头。"""
    validate_llm_config(settings)
    headers = {}
    if settings.llm_api_key.get_secret_value():
        headers["Authorization"] = "Bearer " + settings.llm_api_key.get_secret_value()
    async with httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=settings.llm_timeout_seconds,
    ) as client:
        yield HTTPStructuredLLM(settings, client)


@asynccontextmanager
async def open_embedding_provider(settings: Settings) -> AsyncIterator[EmbeddingProvider]:
    """按配置创建唯一 Embedding 客户端，并确保所有异常路径释放连接。"""
    validate_embedding_config(settings)
    headers = {}
    if settings.embedding_api_key.get_secret_value():
        headers["Authorization"] = "Bearer " + settings.embedding_api_key.get_secret_value()
    async with httpx.AsyncClient(
        base_url=settings.embedding_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=settings.embedding_timeout_seconds,
    ) as client:
        yield HTTPEmbeddingProvider(settings, client)


@asynccontextmanager
async def open_reranker(settings: Settings) -> AsyncIterator[Reranker]:
    """按配置构造 HTTP 重排器，密钥只进入请求头，不进入 Evidence 或日志。"""
    validate_reranker_config(settings)
    headers = {}
    if settings.reranker_api_key.get_secret_value():
        headers["Authorization"] = "Bearer " + settings.reranker_api_key.get_secret_value()
    async with httpx.AsyncClient(
        base_url=settings.reranker_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=settings.reranker_timeout_seconds,
    ) as client:
        yield HTTPReranker(settings, client)


async def configure_knowledge_service(
    settings: Settings, service: KnowledgeService, vector: MilvusVectorStore, stack: AsyncExitStack
) -> None:
    """注入检索端口，实际 Lite 补建缺失索引，完整校验后才允许应用就绪。"""
    embedding = (
        await stack.enter_async_context(open_embedding_provider(settings))
        if settings.retrieval_mode != "bm25"
        else None
    )
    reranker = (
        await stack.enter_async_context(open_reranker(settings))
        if settings.reranker_enabled
        else None
    )
    profile = (
        service.index_profile(
            provider=settings.embedding_provider,
            model=settings.embedding_model,
            model_version=settings.embedding_model_version,
            dimension=settings.embedding_dimension,
            metric_type=settings.milvus_metric_type,
            endpoint=settings.embedding_base_url,
        )
        if embedding is not None
        else None
    )
    service.configure_retrieval(
        mode=settings.retrieval_mode,
        embedding=embedding,
        vector_store=vector,
        profile=profile,
        reranker=reranker,
        recall_count=settings.retrieval_top_k,
        rrf_constant=settings.fusion_rrf_constant,
    )
    # 只在工厂判断实际后端，医学知识边界不感知 Remote/Lite。
    # 不能从配置 auto 推断最终后端：远程连接失败后这里已经是 Lite。
    if (
        vector.status.backend == "lite"
        and settings.milvus_lite_auto_index
        and embedding is not None
        and profile is not None
        and await vector.get_index(profile) is None
    ):
        log_backend_event("knowledge_index_build_started", "lite", "MissingIndex")
        try:
            # 复用版本化全量发布：模型成功、五类集合写完且计数核验后才发布。
            await service.build_index(embedding, vector, profile)
            await service.check_index()
        except Exception as exc:
            log_backend_event("knowledge_index_build_failed", "lite", type(exc).__name__)
            raise KnowledgeRetrievalError(
                "Milvus Lite 启动自动索引失败，请检查 Embedding 配置、网络及本地写入权限；"
                "修复后重新启动将自动重试"
            ) from exc
        log_backend_event("knowledge_index_build_completed", "lite", "IndexValidated")
        return
    await service.check_index()


@asynccontextmanager
async def open_container(settings: Settings) -> AsyncIterator[ApplicationContainer]:
    """装配资源并注册清理，后续初始化失败时也不会泄露连接。"""
    async with AsyncExitStack() as stack:
        llm = (
            await stack.enter_async_context(open_llm(settings))
            if settings.clinical_parser_mode == "llm"
            or settings.reasoning_mode == "llm"
            or settings.knowledge_qa_mode == "llm"
            else None
        )
        vector = await build_milvus_store(settings)
        stack.push_async_callback(vector.close)
        checkpoint = await build_checkpoint_runtime(settings)
        stack.push_async_callback(checkpoint.close)
        try:
            database = await open_database(settings)
            effective_storage = settings.consultation_storage
        except PersistenceError:
            # 当前用户明确允许开发环境丢失旧业务数据；PostgreSQL 故障时只把业务 Repository
            # 降为进程内存，LangGraph checkpoint 仍由独立 SQLite fallback 管理，二者不能混用。
            if settings.app_env == "prod" or settings.consultation_storage != "postgres":
                raise
            log_backend_event("business_database_fallback", "memory", "PostgreSQLUnavailable")
            database = DatabaseRuntime("disabled")
            effective_storage = "memory"
        stack.push_async_callback(database.close)
        knowledge = await KnowledgeService.from_directory(
            settings.knowledge_directory, settings.knowledge_base_version
        )
        await configure_knowledge_service(settings, knowledge, vector, stack)
        reasoning = ReasoningService(
            knowledge,
            checkpoint.checkpointer,
            channel_weight=settings.topk_channel_weight,
            llm=llm if settings.clinical_parser_mode == "llm" else None,
            reasoning_llm=llm if settings.reasoning_mode == "llm" else None,
            scoring_policy=settings.scoring_policy,
            evidence_gap_policy=settings.evidence_gap_policy,
            max_supplement_rounds=settings.max_supplement_rounds,
            formula_branch_recall_top_n=settings.formula_branch_recall_top_n,
        )
        repository: ConsultationRepository
        if effective_storage == "postgres":
            assert database.engine is not None
            repository = PostgresConsultationRepository(
                database.engine, database.coordination_engine
            )
        elif effective_storage == "memory":
            repository = InMemoryConsultationRepository()
        else:
            repository = DisabledConsultationRepository()
        consultations = ConsultationService(
            repository,
            reasoning,
            protocol_version="1.2-six-stage",
            knowledge_base_version=settings.knowledge_base_version,
            scoring_config_version=settings.scoring_config_version,
            max_supplement_rounds=settings.max_supplement_rounds,
            lease_seconds=settings.run_lease_seconds,
            poll_seconds=settings.run_poll_seconds,
            timeout_seconds=settings.run_timeout_seconds,
            max_attempts=settings.run_max_attempts,
            max_concurrent=settings.run_max_concurrent,
            workflow=build_consultation_workflow(
                knowledge,
                checkpoint.checkpointer,
                channel_weight=settings.topk_channel_weight,
                max_supplement_rounds=settings.max_supplement_rounds,
                formula_branch_recall_top_n=settings.formula_branch_recall_top_n,
                reasoning_llm=llm if settings.reasoning_mode == "llm" else None,
                scoring_policy=settings.scoring_policy,
                evidence_gap_policy=settings.evidence_gap_policy,
            ),
            asynchronous=settings.consultation_run_mode == "async",
            versions={
                "llm_model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "embedding_model_version": settings.embedding_model_version,
                "reranker_model": settings.reranker_model,
                "clinical_parser_mode": settings.clinical_parser_mode,
                "reasoning_mode": settings.reasoning_mode,
                "scoring_policy": settings.scoring_policy.model_dump_json(),
                "evidence_gap_policy": settings.evidence_gap_policy.model_dump_json(),
                "evidence_gap_version": "rp11/v1.0",
                "path_summary_version": "path-summary/v1.0",
                "path_completeness_version": "path-completeness/v1.0",
                "clinical_context_version": "clinical-context/v2.0",
                "relationship_version": "relationship/v2.0",
                "similar_case_version": "similar-case/v1.0",
                "stage_replay_version": "stage-replay/v1.2-six-stage",
                "retrospective_replay_version": "retrospective-replay/v1.0",
                "prompt_version": (
                    "clinical-parser/v1.2+channel-candidate/v1.2"
                    if settings.clinical_parser_mode == "llm" and settings.reasoning_mode == "llm"
                    else (
                        "clinical-parser/v1.2"
                        if settings.clinical_parser_mode == "llm"
                        else (
                            "channel-candidate/v1.2"
                            if settings.reasoning_mode == "llm"
                            else "rules-v0.3"
                        )
                    )
                ),
                "backend_build_version": "1.2.0",
                "source_condition_version": "source-condition/v2",
                "evidence_grader_version": "grounded-v2",
            },
        )
        stack.push_async_callback(consultations.close)
        if settings.consultation_storage != "disabled":
            await consultations.runtime.start()
        yield ApplicationContainer(
            settings,
            vector,
            checkpoint,
            database,
            knowledge,
            reasoning,
            consultations,
            ReviewService(consultations),
            StreamService(consultations, heartbeat_seconds=settings.sse_heartbeat_seconds),
            KnowledgeQAService(
                knowledge,
                KnowledgeQAAgent(knowledge, llm if settings.knowledge_qa_mode == "llm" else None),
            ),
        )
