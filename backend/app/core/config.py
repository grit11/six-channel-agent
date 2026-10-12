"""
应用配置模块。

职责：集中读取并校验环境变量，为 LLM、RAG、Milvus、业务数据库和 checkpoint
提供稳定配置对象。边界：不创建外部客户端，不承载医学推理逻辑。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.app.schemas.reasoning import EvidenceGapPolicy, ScoringPolicy


class Settings(BaseSettings):
    """六经辨证后端的集中配置。"""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="dev", description="app_env 配置值")
    app_host: str = Field(default="0.0.0.0", description="app_host 配置值")
    app_port: int = Field(default=8000, description="app_port 配置值")
    log_level: str = Field(default="INFO", description="log_level 配置值")
    llm_provider: str = Field(default="openai_compatible", description="llm_provider 配置值")
    llm_base_url: str = Field(default="", description="llm_base_url 配置值")
    llm_api_key: SecretStr = Field(default=SecretStr(""), description="llm_api_key 敏感配置")
    llm_model: str = Field(default="", description="llm_model 配置值")
    llm_temperature: float = Field(default=0.1, ge=0, le=2, description="LLM 采样温度")
    llm_timeout_seconds: float = Field(default=60.0, gt=0, description="LLM 请求超时")
    llm_max_retries: int = Field(default=2, ge=0, le=5, description="LLM 有限重试次数")
    clinical_parser_mode: Literal["rules", "llm"] = Field(
        default="rules", description="显式选择确定性基线或结构化 LLM，不因失败静默回退"
    )
    reasoning_mode: Literal["rules", "llm"] = Field(
        default="rules", description="八纲与六经候选校正模式；模型失败不静默回退"
    )
    knowledge_qa_mode: Literal["extractive", "llm"] = Field(
        default="extractive", description="独立 QA 来源摘录或结构化模型选择摘录；失败不静默降级"
    )

    embedding_provider: str = Field(default="siliconflow", description="embedding_provider 配置值")
    embedding_base_url: str = Field(default="", description="embedding_base_url 配置值")
    embedding_api_key: SecretStr = Field(
        default=SecretStr(""), description="embedding_api_key 敏感配置"
    )
    embedding_model: str = Field(default="", description="embedding_model 配置值")
    embedding_dimension: int = Field(default=1024, description="embedding_dimension 配置值")
    embedding_batch_size: int = Field(default=32, description="embedding_batch_size 配置值")
    embedding_model_version: str = Field(default="", description="实际模型修订版本，绑定索引")
    embedding_timeout_seconds: float = Field(default=30, gt=0, description="Embedding 请求超时")
    embedding_max_retries: int = Field(default=2, ge=0, le=5, description="Embedding 有限重试次数")

    reranker_provider: str = Field(default="siliconflow", description="reranker_provider 配置值")
    reranker_base_url: str = Field(default="", description="reranker_base_url 配置值")
    reranker_api_key: SecretStr = Field(
        default=SecretStr(""), description="reranker_api_key 敏感配置"
    )
    reranker_model: str = Field(default="", description="reranker_model 配置值")
    reranker_timeout_seconds: float = Field(
        default=30.0, description="reranker_timeout_seconds 配置值"
    )
    reranker_max_retries: int = Field(default=2, ge=0, le=5, description="重排有限重试次数")
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = Field(
        default="bm25", description="明确选择检索方式，模型故障不静默切换"
    )
    reranker_enabled: bool = Field(default=False, description="是否启用真实模型重排")
    fusion_rrf_constant: int = Field(default=60, ge=1, description="RRF 排名平滑常数")
    milvus_operation_timeout_seconds: float = Field(default=30, gt=0, description="向量读写超时")

    milvus_mode: Literal["auto", "remote", "lite"] = Field(
        default="auto",
        pattern="^(auto|remote|lite)$",
        description="milvus_mode 配置约束",
    )
    milvus_uri: str = Field(default="http://localhost:19530", description="milvus_uri 配置值")
    milvus_token: SecretStr = Field(default=SecretStr(""), description="milvus_token 敏感配置")
    milvus_database: str = Field(default="default", description="milvus_database 配置值")
    milvus_lite_path: str = Field(
        default="./data/milvus_lite.db", description="milvus_lite_path 配置值"
    )
    milvus_fallback_to_lite: bool = Field(
        default=True, description="milvus_fallback_to_lite 配置值"
    )
    milvus_lite_auto_index: bool = Field(
        default=True, description="实际 Lite 启动时自动补建缺失的知识/模型索引"
    )
    milvus_connect_timeout_seconds: float = Field(
        default=3.0, description="milvus_connect_timeout_seconds 配置值"
    )
    milvus_collection_classic: str = Field(
        default="six_channel_classic_v12",
        description="milvus_collection_classic 配置值",
    )
    milvus_collection_formula: str = Field(
        default="six_channel_formula_v12",
        description="milvus_collection_formula 配置值",
    )
    milvus_collection_differential: str = Field(
        default="six_channel_differential_v12",
        description="milvus_collection_differential 配置值",
    )
    milvus_collection_case: str = Field(
        default="six_channel_case_v12", description="milvus_collection_case 配置值"
    )
    milvus_collection_framework: str = Field(
        default="six_channel_framework_v12",
        description="milvus_collection_framework 配置值",
    )
    milvus_metric_type: Literal["COSINE", "IP", "L2"] = Field(
        default="COSINE", description="milvus_metric_type 配置值"
    )
    milvus_index_type: str = Field(default="HNSW", description="milvus_index_type 配置值")
    milvus_hnsw_m: int = Field(default=16, description="milvus_hnsw_m 配置值")
    milvus_hnsw_ef_construction: int = Field(
        default=200, description="milvus_hnsw_ef_construction 配置值"
    )
    milvus_search_ef: int = Field(default=64, description="milvus_search_ef 配置值")

    database_enabled: bool = Field(default=False, description="database_enabled 配置值")
    consultation_storage: Literal["disabled", "memory", "postgres"] = Field(
        default="disabled", description="会话存储能力开关；memory 仅限开发验证，重启不保留"
    )
    consultation_run_mode: Literal["async", "sync"] = Field(
        default="async", description="异步会诊默认启用；sync 用于确定性工程回归"
    )
    database_url: SecretStr = Field(default=SecretStr(""), description="database_url 敏感配置")
    database_pool_size: int = Field(default=10, description="database_pool_size 配置值")
    database_max_overflow: int = Field(default=20, description="database_max_overflow 配置值")
    database_pool_recycle_seconds: int = Field(
        default=1800, description="database_pool_recycle_seconds 配置值"
    )

    checkpoint_backend: Literal["auto", "postgres", "sqlite"] = Field(
        default="auto",
        pattern="^(auto|postgres|sqlite)$",
        description="checkpoint_backend 配置约束",
    )
    langgraph_database_url: SecretStr = Field(
        default=SecretStr(""), description="langgraph_database_url 敏感配置"
    )
    langgraph_sqlite_path: str = Field(
        default="./data/langgraph_checkpoints.sqlite3",
        description="langgraph_sqlite_path 配置值",
    )
    checkpoint_fallback_to_sqlite: bool = Field(
        default=True, description="checkpoint_fallback_to_sqlite 配置值"
    )
    postgres_connect_timeout_seconds: float = Field(
        default=3.0, description="postgres_connect_timeout_seconds 配置值"
    )

    knowledge_directory: Path = Field(
        default=Path(__file__).resolve().parents[3] / "knowledge" / "v1.2",
        description="五类运行时主库目录",
    )
    knowledge_base_version: str = Field(default="1.2", description="knowledge_base_version 配置值")
    formula_branch_recall_top_n: int = Field(default=12, ge=5, le=30)
    protocol_version: str = Field(default="1.2-six-stage", description="protocol_version 配置值")
    scoring_config_version: str = Field(default="1.0", description="scoring_config_version 配置值")
    scoring_policy: ScoringPolicy = Field(
        default_factory=ScoringPolicy, description="RP-13 七因子和反证惩罚配置，环境变量为 JSON"
    )
    evidence_gap_policy: EvidenceGapPolicy = Field(
        default_factory=EvidenceGapPolicy, description="RP-11 提问因子与停止阈值，环境变量为 JSON"
    )
    topk_channel_weight: float = Field(
        default=0.6, ge=0, le=1, description="Top-K 基线聚合中的六经分权重，方证权重为余数"
    )
    top_k_default: int = Field(default=3, ge=1, le=5, description="top_k_default 配置约束")
    top_k_max: int = Field(default=5, ge=1, le=5, description="top_k_max 配置约束")
    retrieval_top_k: int = Field(default=20, ge=1, le=100, description="retrieval_top_k 配置约束")
    rerank_top_k: int = Field(default=8, ge=1, description="rerank_top_k 配置约束")
    max_supplement_rounds: int = Field(
        default=3, ge=0, description="max_supplement_rounds 配置约束"
    )
    sse_heartbeat_seconds: int = Field(
        default=15, ge=1, description="sse_heartbeat_seconds 配置约束"
    )
    run_lease_seconds: float = Field(default=30, ge=3, le=300, description="运行续约租期")
    run_poll_seconds: float = Field(default=1, ge=0.1, le=30, description="持久任务恢复轮询")
    run_timeout_seconds: float = Field(default=300, ge=1, le=3600, description="单轮推理硬超时")
    run_max_attempts: int = Field(default=3, ge=1, le=10, description="崩溃接管最大执行次数")
    run_max_concurrent: int = Field(default=4, ge=1, le=16, description="每实例推理并发上限")

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        """启动时验证关联配置，避免写入索引或请求执行时才暴露错误。"""
        if self.top_k_default > self.top_k_max:
            raise ValueError("默认 Top-K 不得超过最大 Top-K")
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError("重排数量不得超过召回数量")
        if self.embedding_dimension <= 0 or self.embedding_batch_size <= 0:
            raise ValueError("Embedding 维度和批次必须大于零")
        if self.milvus_metric_type not in {"COSINE", "IP", "L2"}:
            raise ValueError("向量距离必须为 COSINE、IP 或 L2")
        if self.reranker_timeout_seconds <= 0:
            raise ValueError("Reranker 超时必须大于零")
        if min(self.milvus_hnsw_m, self.milvus_hnsw_ef_construction, self.milvus_search_ef) <= 0:
            raise ValueError("Milvus 索引和搜索参数必须大于零")
        if self.retrieval_mode != "bm25" and not all(
            (self.embedding_base_url, self.embedding_model, self.embedding_model_version)
        ):
            raise ValueError("Dense/Hybrid 必须配置 Embedding 地址、模型和版本")
        if self.reranker_enabled and not all((self.reranker_base_url, self.reranker_model)):
            raise ValueError("启用重排必须配置服务地址和模型")
        if self.milvus_connect_timeout_seconds <= 0 or self.postgres_connect_timeout_seconds <= 0:
            raise ValueError("基础设施连接超时必须大于零")
        if (
            self.checkpoint_backend == "postgres"
            and not self.langgraph_database_url.get_secret_value()
        ):
            raise ValueError("强制 PostgreSQL checkpoint 时必须配置连接")
        if self.database_enabled and not self.database_url.get_secret_value().startswith(
            "postgresql+asyncpg://"
        ):
            raise ValueError("业务数据库必须使用 PostgreSQL asyncpg URL")
        if self.consultation_storage == "postgres" and not self.database_enabled:
            raise ValueError("PostgreSQL 会话存储必须启用业务数据库")
        if self.clinical_parser_mode == "llm" and (
            not self.llm_model or not self.llm_base_url.startswith(("http://", "https://"))
        ):
            raise ValueError("LLM 解析模式必须配置 HTTP(S) 地址和模型名")
        if self.reasoning_mode == "llm" and (
            not self.llm_model or not self.llm_base_url.startswith(("http://", "https://"))
        ):
            raise ValueError("结构化推理模式必须配置 HTTP(S) 地址和模型名")
        if self.knowledge_qa_mode == "llm" and (
            not self.llm_model or not self.llm_base_url.startswith(("http://", "https://"))
        ):
            raise ValueError("知识问答模型模式必须配置 HTTP(S) 地址和模型名")
        if self.app_env == "prod":
            if self.consultation_storage == "memory":
                raise ValueError("生产环境禁止使用内存业务 Repository")
            for name in ("llm", "embedding", "reranker"):
                if not getattr(self, f"{name}_model"):
                    raise ValueError(f"生产环境必须配置 {name} 模型")
                if not getattr(self, f"{name}_api_key").get_secret_value():
                    raise ValueError(f"生产环境必须配置 {name} 密钥")
            if not self.database_enabled:
                raise ValueError("生产环境必须启用业务 PostgreSQL")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存应用配置，确保所有模块使用同一配置快照。"""

    return Settings()
