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
from backend.app.knowledge.service import KnowledgeService
from backend.app.persistence.checkpoint import (
    CheckpointRuntime,
    build_checkpoint_runtime,
)
from backend.app.persistence.database import DatabaseRuntime, open_database
from backend.app.persistence.repositories import (
    ConsultationRepository,
)
from backend.app.rag.embedding import HTTPEmbeddingProvider, validate_embedding_config
from backend.app.rag.reranker import HTTPReranker, Reranker, validate_reranker_config
from backend.app.rag.vector_store import MilvusVectorStore, build_milvus_store
from backend.app.services.clinical_v2_service import ClinicalV2Service
from backend.app.services.consultation_service import ConsultationService
from backend.app.services.knowledge_qa_service import KnowledgeQAService
from backend.app.services.runtime_service import ReliableRuntime
from backend.app.services.stream_service import StreamService


@dataclass
class ApplicationContainer:
    """应用生命周期依赖，不保存到 LangGraph State。"""

    settings: Settings
    vector_store: MilvusVectorStore
    checkpoint_runtime: CheckpointRuntime
    database: DatabaseRuntime
    knowledge_service: KnowledgeService
    knowledge_qa_service: KnowledgeQAService
    clinical_v2_service: ClinicalV2Service | None
    consultation_service: ConsultationService
    stream_service: StreamService


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
async def open_embedding_provider(
    settings: Settings,
) -> AsyncIterator[EmbeddingProvider]:
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
    settings: Settings,
    service: KnowledgeService,
    vector: MilvusVectorStore,
    stack: AsyncExitStack,
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
        llm: StructuredLLM | None = None
        if settings.llm_base_url and settings.llm_model:
            llm = await stack.enter_async_context(open_llm(settings))
        vector = await build_milvus_store(settings)
        stack.push_async_callback(vector.close)
        checkpoint = await build_checkpoint_runtime(settings)
        stack.push_async_callback(checkpoint.close)
        try:
            database = await open_database(settings)
        except PersistenceError:
            # 当前用户明确允许开发环境丢失旧业务数据；PostgreSQL 故障时只把业务 Repository
            # 降为进程内存，LangGraph checkpoint 仍由独立 SQLite fallback 管理，二者不能混用。
            if settings.app_env == "prod" or settings.consultation_storage != "postgres":
                raise
            log_backend_event("business_database_fallback", "memory", "PostgreSQLUnavailable")
            database = DatabaseRuntime("disabled")
        stack.push_async_callback(database.close)
        knowledge = await KnowledgeService.from_directory(
            settings.knowledge_directory, settings.knowledge_base_version
        )
        await configure_knowledge_service(settings, knowledge, vector, stack)
        clinical_v2 = (
            ClinicalV2Service(
                llm,
                checkpoint.checkpointer,
                knowledge,
                max_clarification_rounds=settings.max_clarification_rounds,
                retrieval_top_k=settings.retrieval_top_k,
                differential_top_k=settings.rerank_top_k,
            )
            if llm is not None
            else None
        )
        repository: ConsultationRepository
        if settings.consultation_storage == "postgres" and database.engine is not None:
            from backend.app.persistence.repositories import PostgresConsultationRepository

            repository = PostgresConsultationRepository(
                database.engine, database.coordination_engine
            )
        elif settings.consultation_storage == "memory" or (
            settings.consultation_storage == "postgres" and settings.app_env != "prod"
        ):
            from backend.app.persistence.repositories import InMemoryConsultationRepository

            repository = InMemoryConsultationRepository()
        else:
            from backend.app.persistence.repositories import DisabledConsultationRepository

            repository = DisabledConsultationRepository()
        consultation_service = ConsultationService(
            repository,
            clinical_v2,
            run_mode=settings.consultation_run_mode,
            versions={
                "knowledge_base_version": settings.knowledge_base_version,
                "llm_model": settings.llm_model,
                "protocol_version": "clinical-v2.0",
            },
        )
        runtime = ReliableRuntime(
            consultation_service,
            lease_seconds=settings.run_lease_seconds,
            poll_seconds=settings.run_poll_seconds,
            timeout_seconds=settings.run_timeout_seconds,
            max_attempts=settings.run_max_attempts,
            max_concurrent=settings.run_max_concurrent,
        )
        consultation_service.runtime = runtime
        await runtime.start()
        stack.push_async_callback(runtime.close)
        stream_service = StreamService(repository.runtime)
        yield ApplicationContainer(
            settings,
            vector,
            checkpoint,
            database,
            knowledge,
            KnowledgeQAService(
                knowledge,
                KnowledgeQAAgent(knowledge, llm if settings.knowledge_qa_mode == "llm" else None),
            ),
            clinical_v2,
            consultation_service,
            stream_service,
        )
