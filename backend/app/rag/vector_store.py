"""Milvus 适配：统一 Remote/Lite 生命周期、保活和版本化索引。

依赖配置、pymilvus 和 Lite 服务管理器；取消请求时先等待同步 SDK 结束，最后释放
客户端及本适配器持有的 Lite 服务，避免 Windows 文件锁残留。不承载医学判断。
"""

import asyncio
import json
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, TypeVar
from uuid import uuid4

from pymilvus import DataType, MilvusClient  # type: ignore[import-untyped]

from backend.app.core.config import Settings
from backend.app.core.exceptions import (
    ConfigurationError,
    KnowledgeRetrievalError,
    ProviderUnavailableError,
)
from backend.app.core.logging import log_backend_event
from backend.app.rag.embedding import validate_vectors
from backend.app.schemas.evidence import RetrievedDocument
from backend.app.schemas.knowledge import (
    IndexProfile,
    IndexReport,
    KnowledgeRecord,
    KnowledgeType,
    VectorDocument,
)

Result = TypeVar("Result")
# Lite 的 .db 实际为本地 gRPC 服务；关闭客户端并不自动停止服务或释放文件锁。
# 同一路径的工厂创建/关闭串行处理，只有最后一个本适配器客户端退出才释放服务。
_lite_lifecycle_lock = threading.RLock()
_lite_references: dict[str, int] = {}


async def _run_sync(operation: Callable[[], Result]) -> Result:
    """等待不可取消的同步 SDK 线程完成，避免关闭资源时仍有后台读写。"""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except Exception as exc:
            # 仍向调用者传播取消；线程失败单独记录安全类别，不输出 SDK URI。
            log_backend_event("vector_cancelled_operation_failed", "milvus", type(exc).__name__)
        raise


def _lite_path(settings: Settings) -> str:
    """统一相对路径和大小写，确保打开与释放指向同一 Lite 服务。"""
    return str(Path(settings.milvus_lite_path).resolve())


def _release_lite_server(path: str) -> None:
    """释放当前路径服务，不清理其他数据库、连接或索引文件。"""
    from milvus_lite.server_manager import server_manager_instance  # type: ignore[import-untyped]

    server_manager_instance.release_server(path)


def _close_client(client: MilvusClient, lite_path: str | None) -> None:
    """关闭专用通道，并在最后一个 Lite 引用退出时释放本地数据库锁。"""
    if lite_path is None:
        client.close()
        return
    with _lite_lifecycle_lock:
        client.close()
        remaining = _lite_references[lite_path] - 1
        if remaining:
            _lite_references[lite_path] = remaining
        else:
            _release_lite_server(lite_path)
            del _lite_references[lite_path]


@dataclass(frozen=True)
class VectorBackendStatus:
    """向量后端实际状态，客户端和密钥不得进入图 State。"""

    backend: Literal["remote", "lite"]
    degraded: bool
    detail: str


class VectorStorePort(Protocol):
    """统一向量生命周期、版本化写入和检索边界。"""

    @property
    def status(self) -> VectorBackendStatus:
        """返回实际后端状态。"""
        ...

    async def close(self) -> None:
        """关闭应用级客户端。"""
        ...

    async def get_index(self, profile: IndexProfile) -> IndexReport | None:
        """读取完整发布版本，未完成的暂存集合不可查询。"""
        ...

    async def upsert(
        self,
        documents: list[VectorDocument],
        profile: IndexProfile,
        *,
        rebuild: bool = False,
    ) -> IndexReport:
        """幂等发布完整快照，不逐批覆盖可见索引。"""
        ...

    async def search(
        self,
        *,
        query_vector: list[float],
        knowledge_type: KnowledgeType,
        top_k: int,
        profile: IndexProfile,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedDocument]:
        """使用匹配模型的完整索引并按确定元数据过滤。"""
        ...

    async def delete_by_version(self, knowledge_version: str) -> None:
        """显式移除该版本发布入口和其集合，不影响其他版本。"""
        ...


class MilvusVectorStore:
    """封装 Remote/Lite 客户端，禁止上层直接操作厂商 SDK。"""

    def __init__(
        self,
        client: MilvusClient,
        status: VectorBackendStatus,
        settings: Settings | None = None,
        *,
        close_client: Callable[[], None] | None = None,
    ) -> None:
        """绑定已探测的客户端与后端状态。"""
        self._client = client
        self._close_client = close_client or client.close
        self._closed = False
        self._operation_lock = asyncio.Lock()
        self._status = status
        self._settings = settings or Settings()
        self._bases = {
            "classic_clause": self._settings.milvus_collection_classic,
            "formula_pattern": self._settings.milvus_collection_formula,
            "differential": self._settings.milvus_collection_differential,
            "medical_case": self._settings.milvus_collection_case,
            "six_channel_framework": self._settings.milvus_collection_framework,
        }
        if len(set(self._bases.values())) != 5 or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,150}", base) for base in self._bases.values()
        ):
            raise ConfigurationError("五类 Milvus 集合名称必须合法且互不重复")
        self._manifest = self._settings.milvus_collection_classic + "_manifest"
        self._timeout = self._settings.milvus_operation_timeout_seconds
        # Lite 使用其通用 FLAT 能力，生产索引参数仍由配置控制。
        self._index_type = "FLAT" if status.backend == "lite" else self._settings.milvus_index_type
        self._write_lock = asyncio.Lock()

    def _read_options(self) -> dict[str, object]:
        """Remote 强一致读发布入口；Lite 本地即刻可见，避免调用未实现的时钟 RPC。"""
        return {"consistency_level": "Strong"} if self.status.backend == "remote" else {}

    def _ensure_loaded(self, collection_name: str) -> None:
        """进程重启后重新加载 Lite/Remote 集合，查询前统一恢复可读状态。"""
        self._client.load_collection(collection_name, timeout=self._timeout)

    @property
    def status(self) -> VectorBackendStatus:
        """返回 Remote 或 Lite 实际状态。"""
        return self._status

    async def close(self) -> None:
        """等待在途 SDK 调用后幂等关闭；Lite 最后一个引用退出时释放文件锁。"""
        async with self._operation_lock:
            if self._closed:
                return

            def close() -> None:
                """线程完成即记录关闭状态，外层取消也不能重复减少 Lite 引用。"""
                self._close_client()
                self._closed = True

            await _run_sync(close)

    async def _operate(self, operation: Callable[[], Result]) -> Result:
        """串行保护读写与关闭边界；取消不允许线程继续操作已关闭数据库。"""
        async with self._operation_lock:
            if self._closed:
                raise KnowledgeRetrievalError("Milvus 客户端已关闭")
            return await _run_sync(operation)

    def _ensure_manifest(self) -> None:
        """发布入口使用独立集合，单行更新使五类集合一起可见。"""
        if self._client.has_collection(self._manifest, timeout=self._timeout):
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("version", DataType.VARCHAR, max_length=128)
        schema.add_field("report", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
        index = MilvusClient.prepare_index_params()
        index.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        self._client.create_collection(
            self._manifest, schema=schema, index_params=index, timeout=self._timeout
        )

    def _get_index(self, profile: IndexProfile) -> IndexReport | None:
        """读取不可混用的模型/来源快照；不存在时保持未索引状态。"""
        if not self._client.has_collection(self._manifest, timeout=self._timeout):
            return None
        self._ensure_loaded(self._manifest)
        rows = self._client.get(
            self._manifest,
            ids=[profile.fingerprint()],
            output_fields=["report"],
            timeout=self._timeout,
            **self._read_options(),
        )
        if not rows:
            return None
        report = IndexReport.model_validate(rows[0]["report"])
        if report.profile != profile:
            raise ValueError("索引身份与查询不兼容")
        if set(report.collections) != set(self._bases) or set(report.counts) != set(self._bases):
            raise ValueError("索引发布记录缺少五类知识集合")
        if any(count < 1 for count in report.counts.values()) or not all(
            self._client.has_collection(name, timeout=self._timeout)
            for name in report.collections.values()
        ):
            raise ValueError("索引集合缺失或计数无效")
        return report.model_copy(update={"backend": self.status.backend, "reused": True})

    async def get_index(self, profile: IndexProfile) -> IndexReport | None:
        """在线程池读取完整版本，不将空集合当作成功索引。"""
        try:
            return await self._operate(partial(self._get_index, profile))
        except Exception as exc:
            raise KnowledgeRetrievalError("读取索引发布状态失败") from exc

    def _create_collection(self, name: str, profile: IndexProfile) -> None:
        """统一字段契约；metadata 与多六经标签保持完整，不转成单标签。"""
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for field, length in (
            ("id", 512),
            ("version", 128),
            ("knowledge_type", 64),
            ("association_scope", 64),
            ("risk_level", 64),
        ):
            schema.add_field(field, DataType.VARCHAR, max_length=length, is_primary=field == "id")
        schema.add_field("record", DataType.JSON)
        schema.add_field("channels", DataType.JSON)
        schema.add_field("canonical", DataType.BOOL)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=profile.dimension)
        index = MilvusClient.prepare_index_params()
        parameters: dict[str, int] = {}
        if self._index_type == "HNSW":
            parameters = {
                "M": self._settings.milvus_hnsw_m,
                "efConstruction": self._settings.milvus_hnsw_ef_construction,
            }
        index.add_index(
            field_name="vector",
            index_type=self._index_type,
            metric_type=self._settings.milvus_metric_type,
            params=parameters,
        )
        self._client.create_collection(
            name, schema=schema, index_params=index, timeout=self._timeout
        )

    def _publish(
        self, documents: list[VectorDocument], profile: IndexProfile, rebuild: bool
    ) -> IndexReport:
        """先完成独立暂存集合再发布指针，失败只清理本次集合，保留旧版本。"""
        validate_vectors(
            [document.vector for document in documents],
            len(documents),
            profile.dimension,
        )
        counts: Counter[str] = Counter(document.record.knowledge_type for document in documents)
        if profile.metric_type != self._settings.milvus_metric_type:
            raise ValueError("索引距离度量与配置不兼容")
        if set(counts) != set(self._bases) or len(
            {document.record.knowledge_id for document in documents}
        ) != len(documents):
            raise ValueError("发布必须包含五类完整且不重复的知识快照")
        if any(
            document.record.knowledge_version != profile.knowledge_version for document in documents
        ):
            raise ValueError("知识版本与索引身份不匹配")
        previous = self._get_index(profile)
        if previous is not None and not rebuild:
            return previous
        self._ensure_manifest()
        generation = uuid4().hex[:12]
        collections = {
            kind: f"{base}_{profile.fingerprint()}_{generation}"
            for kind, base in self._bases.items()
        }
        created: list[str] = []
        try:
            for kind, name in collections.items():
                created.append(name)
                self._create_collection(name, profile)
                rows = []
                for document in documents:
                    record = document.record
                    if record.knowledge_type != kind:
                        continue
                    scope = str(record.metadata.get("association_scope", ""))
                    rows.append(
                        {
                            "id": record.knowledge_id,
                            "version": record.knowledge_version,
                            "knowledge_type": kind,
                            "record": record.model_dump(mode="json"),
                            "channels": record.channels,
                            "association_scope": scope,
                            "risk_level": str(record.metadata.get("risk_level", "")),
                            "canonical": kind != "medical_case"
                            and scope != "case_derived_historical",
                            "vector": document.vector,
                        }
                    )
                for start in range(0, len(rows), self._settings.embedding_batch_size):
                    self._client.upsert(
                        name,
                        rows[start : start + self._settings.embedding_batch_size],
                        timeout=self._timeout,
                    )
                self._client.flush(name, timeout=self._timeout)
                self._ensure_loaded(name)
                actual = self._client.query(
                    name,
                    filter="",
                    output_fields=["count(*)"],
                    timeout=self._timeout,
                    **self._read_options(),
                )
                if int(actual[0]["count(*)"]) != counts[kind]:
                    raise ValueError("索引计数核验失败")
            report = IndexReport(
                backend=self.status.backend,
                index_type=self._index_type,
                profile=profile,
                counts=dict(counts),
                collections=collections,
                reused=False,
            )
        except BaseException:
            # 不吞掉清理失败；异常链与暂存集合名称仍可用于运维追查。
            for name in created:
                if self._client.has_collection(name, timeout=self._timeout):
                    self._client.drop_collection(name, timeout=self._timeout)
            raise
        # 发布请求遇到网络超时可能已在服务端生效；保留完整集合供重试确认，绝不删除。
        self._client.upsert(
            self._manifest,
            [
                {
                    "id": profile.fingerprint(),
                    "version": profile.knowledge_version,
                    "report": report.model_dump(mode="json"),
                    "vector": [1.0, 0.0],
                }
            ],
            timeout=self._timeout,
        )
        self._client.flush(self._manifest, timeout=self._timeout)
        return report

    async def upsert(
        self,
        documents: list[VectorDocument],
        profile: IndexProfile,
        *,
        rebuild: bool = False,
    ) -> IndexReport:
        """幂等发布完整快照；显式 rebuild 写入新集合，旧集合保留供历史引用。"""
        async with self._write_lock:
            try:
                return await self._operate(partial(self._publish, documents, profile, rebuild))
            except Exception as exc:
                raise KnowledgeRetrievalError("向量索引未能完整发布") from exc

    def _filter(
        self, profile: IndexProfile, kind: KnowledgeType, filters: dict[str, object]
    ) -> str:
        """只允许确定字段，所有值 JSON 转义，拒绝调用者注入 Milvus 表达式。"""
        if set(filters) - {"channels", "association_scope", "risk_level", "canonical"}:
            raise ValueError("不支持的向量过滤字段")
        clauses = [
            f"version == {json.dumps(profile.knowledge_version, ensure_ascii=False)}",
            f"knowledge_type == {json.dumps(kind)}",
        ]
        for field, value in filters.items():
            if field == "canonical":
                if not isinstance(value, bool):
                    raise ValueError("canonical 必须是布尔值")
                clauses.append(f"canonical == {str(value).lower()}")
            else:
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError("过滤条件必须为字符串列表")
                if value:
                    literal = json.dumps(value, ensure_ascii=False)
                    clauses.append(
                        f"json_contains_any(channels, {literal})"
                        if field == "channels"
                        else f"{field} in {literal}"
                    )
        return " and ".join(clauses)

    def _search(
        self,
        vector: list[float],
        kind: KnowledgeType,
        top_k: int,
        profile: IndexProfile,
        expression: str,
    ) -> list[RetrievedDocument]:
        """查询完整发布集合，L2 转负距离使所有召回分统一为越大越相关。"""
        report = self._get_index(profile)
        if report is None:
            raise ValueError("当前知识/模型快照尚未索引，请先运行 index_knowledge.py")
        params = (
            {"ef": max(top_k, self._settings.milvus_search_ef)}
            if report.index_type == "HNSW"
            else {}
        )
        self._ensure_loaded(report.collections[kind])
        hits = self._client.search(
            report.collections[kind],
            data=[vector],
            filter=expression,
            limit=top_k,
            output_fields=["record"],
            search_params={
                "metric_type": self._settings.milvus_metric_type,
                "params": params,
            },
            timeout=self._timeout,
            **self._read_options(),
        )
        results = []
        for hit in hits[0]:
            record = KnowledgeRecord.model_validate(hit["entity"]["record"])
            distance = float(hit["distance"])
            score = -distance if self._settings.milvus_metric_type == "L2" else distance
            results.append(
                RetrievedDocument(
                    **record.model_dump(exclude={"embedding_text", "channels"}),
                    retrieval_score=score,
                    retrieval_method="dense",
                    channel_scores={"dense": score},
                )
            )
        return results

    async def search(
        self,
        *,
        query_vector: list[float],
        knowledge_type: KnowledgeType,
        top_k: int,
        profile: IndexProfile,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievedDocument]:
        """校验向量与过滤条件，异步返回统一来源契约；无命中保持空列表。"""
        if not 1 <= top_k <= 100:
            raise ValueError("向量召回数量必须在 1 至 100")
        if profile.metric_type != self._settings.milvus_metric_type:
            raise ValueError("查询距离度量与索引不兼容")
        validate_vectors([query_vector], 1, profile.dimension)
        expression = self._filter(profile, knowledge_type, filters or {})
        try:
            return await self._operate(
                partial(self._search, query_vector, knowledge_type, top_k, profile, expression)
            )
        except Exception as exc:
            raise KnowledgeRetrievalError("向量查询失败或索引身份不兼容") from exc

    def _delete_version(self, version: str) -> None:
        """先取消版本可见性，再删除其完整快照集合；保留其他版本。"""
        if not self._client.has_collection(self._manifest, timeout=self._timeout):
            return
        expression = f"version == {json.dumps(version, ensure_ascii=False)}"
        rows = self._client.query(
            self._manifest,
            filter=expression,
            output_fields=["id", "report"],
            timeout=self._timeout,
        )
        if not rows:
            return
        self._client.delete(self._manifest, ids=[row["id"] for row in rows], timeout=self._timeout)
        self._client.flush(self._manifest, timeout=self._timeout)
        # 包括此前 rebuild 保留的旧集合，仅匹配当前命名空间及目标快照指纹。
        fingerprints = [str(row["id"]) for row in rows]
        for name in self._client.list_collections(timeout=self._timeout):
            if any(
                name.startswith(f"{base}_{fingerprint}_")
                for base in self._bases.values()
                for fingerprint in fingerprints
            ):
                self._client.drop_collection(name, timeout=self._timeout)

    async def delete_by_version(self, knowledge_version: str) -> None:
        """受锁保护的显式版本删除，仅用于运维，不在正常查询时自动执行。"""
        async with self._write_lock:
            try:
                await self._operate(partial(self._delete_version, knowledge_version))
            except Exception as exc:
                raise KnowledgeRetrievalError("删除知识索引版本失败") from exc


def _open_client(settings: Settings, *, lite: bool) -> MilvusClient:
    """创建并真实探测客户端；失败后关闭已构造的连接。"""
    if lite:
        path = Path(_lite_path(settings))
        path.parent.mkdir(parents=True, exist_ok=True)
        uri, token, database = str(path), "", ""
    else:
        uri, token, database = (
            settings.milvus_uri,
            settings.milvus_token.get_secret_value(),
            settings.milvus_database,
        )
    client = MilvusClient(
        uri=uri,
        token=token,
        db_name=database,
        timeout=settings.milvus_connect_timeout_seconds,
        # 共享连接池不会在 close() 后关闭通道；独占连接让 lifespan 真正拥有资源。
        dedicated=True,
        # SDK 默认 10 秒空闲 ping 会被 Lite/Remote 服务拒绝（too_many_pings）。
        # 不在无 RPC 时发保活；活跃调用的 ping 间隔遵循 gRPC 默认五分钟约束。
        grpc_options={
            "grpc.keepalive_time_ms": 300000,
            "grpc.keepalive_timeout_ms": 20000,
            "grpc.keepalive_permit_without_calls": False,
        },
    )
    try:
        client.list_collections(timeout=settings.milvus_connect_timeout_seconds)
    except Exception:
        client.close()
        raise
    return client


def _create_store(settings: Settings, status: VectorBackendStatus) -> MilvusVectorStore:
    """创建/引用计数/失败清理处于同一线程边界，不留下半初始化 Lite 服务。"""
    path = _lite_path(settings) if status.backend == "lite" else None

    def create() -> MilvusVectorStore:
        """工厂失败只释放自己持有的路径；已有引用不能被另一个失败客户端关闭。"""
        try:
            client = _open_client(settings, lite=path is not None)
        except Exception:
            if path is not None and path not in _lite_references:
                _release_lite_server(path)
            raise
        if path is not None:
            _lite_references[path] = _lite_references.get(path, 0) + 1
        try:
            return MilvusVectorStore(
                client, status, settings, close_client=partial(_close_client, client, path)
            )
        except Exception:
            _close_client(client, path)
            raise

    if path is None:
        return create()
    with _lite_lifecycle_lock:
        return create()


async def _open_store(settings: Settings, status: VectorBackendStatus) -> MilvusVectorStore:
    """启动取消时等待 SDK 创建结束并关闭资源，避免热重载留下 Lite 文件锁。"""
    task = asyncio.create_task(asyncio.to_thread(_create_store, settings, status))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            store = await task
        except Exception as exc:
            log_backend_event("vector_cancelled_startup_failed", status.backend, type(exc).__name__)
        else:
            await store.close()
        raise


async def _open_lite(settings: Settings, reason: str) -> MilvusVectorStore:
    """打开 Lite 文件，包缺失或文件不可用时明确失败并保留异常链。"""
    try:
        return await _open_store(settings, VectorBackendStatus("lite", True, reason))
    except Exception as exc:
        raise ProviderUnavailableError(
            "Milvus Lite 不可用，请检查 langgraph 环境的 milvus-lite 依赖与文件权限"
        ) from exc


async def build_milvus_store(settings: Settings) -> MilvusVectorStore:
    """按配置探测 Remote，仅 auto 且允许 fallback 时降级 Lite。"""
    if settings.milvus_mode == "lite":
        return await _open_lite(settings, "显式 Lite 模式")
    try:
        return await _open_store(settings, VectorBackendStatus("remote", False, "Remote 连接正常"))
    except Exception as exc:
        if settings.milvus_mode == "remote" or not settings.milvus_fallback_to_lite:
            raise ProviderUnavailableError("Remote Milvus 不可用，禁止降级") from exc
        # 只记录异常类型，SDK 错误文本可能包含带凭据的 URI。
        log_backend_event("vector_fallback", "lite", type(exc).__name__)
        return await _open_lite(settings, "Remote 不可用，已降级 Lite")
