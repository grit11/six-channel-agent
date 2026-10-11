"""唯一医学知识访问边界，返回可追溯来源，禁止生成患者诊断或混淆证据层级。"""

import asyncio
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import TypeAdapter

from backend.app.core.exceptions import ConfigurationError, KnowledgeRetrievalError
from backend.app.core.ports import EmbeddingProvider
from backend.app.knowledge.classic import load_records
from backend.app.knowledge.differential import source_candidate_label
from backend.app.observability.progress import emit_progress
from backend.app.rag.embedding import validate_vectors
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.rag.grader import grade_candidate_evidence
from backend.app.rag.reranker import Reranker
from backend.app.rag.retriever import BM25Retriever
from backend.app.rag.vector_store import VectorStorePort
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import (
    DifferentialRule,
    Evidence,
    EvidenceAssessment,
    EvidenceBundle,
    RetrievedDocument,
)
from backend.app.schemas.knowledge import (
    IndexProfile,
    IndexReport,
    KnowledgeRecord,
    KnowledgeType,
    VectorDocument,
)


class KnowledgeService:
    """五类知识加载、精确查询及配置化检索服务，统一保留风险和版本字段。"""

    def __init__(self, records: list[KnowledgeRecord], version: str) -> None:
        """建立只读来源索引，不把客户端或厂商配置放进医学记录。"""
        self.version = version
        self._records = {record.knowledge_id: record for record in records}
        self._retriever = BM25Retriever(records)
        self._clauses = {
            str(r.metadata["clause_no"]): r for r in records if r.knowledge_type == "classic_clause"
        }
        self._embedding: EmbeddingProvider | None = None
        self._vector: VectorStorePort | None = None
        self._reranker: Reranker | None = None
        self._profile: IndexProfile | None = None
        self._mode = "bm25"
        self._recall_count = 20
        self._rrf_constant = 60

    @classmethod
    async def from_directory(cls, directory: Path, version: str) -> "KnowledgeService":
        """在线程池加载知识文件，启动校验失败则不暴露半完成服务。"""
        records = await asyncio.to_thread(load_records, directory, version)
        return await asyncio.to_thread(cls, records, version)

    def counts(self) -> dict[str, int]:
        """返回按来源类型分组的加载数量，用于校验和审计。"""
        return dict(Counter(record.knowledge_type for record in self._records.values()))

    def index_profile(
        self,
        *,
        provider: str,
        model: str,
        model_version: str,
        dimension: int,
        metric_type: Literal["COSINE", "IP", "L2"] = "COSINE",
        endpoint: str = "",
    ) -> IndexProfile:
        """绑定完整来源快照摘要；即使文件版本不变，内容变化也会要求新索引。"""
        snapshot = [self._records[key].model_dump(mode="json") for key in sorted(self._records)]
        digest = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return IndexProfile(
            knowledge_version=self.version,
            embedding_provider=provider,
            embedding_model=model,
            embedding_model_version=model_version,
            embedding_endpoint=endpoint.rstrip("/"),
            dimension=dimension,
            corpus_digest=digest,
            metric_type=metric_type,
        )

    def configure_retrieval(
        self,
        *,
        mode: str,
        embedding: EmbeddingProvider | None = None,
        vector_store: VectorStorePort | None = None,
        profile: IndexProfile | None = None,
        reranker: Reranker | None = None,
        recall_count: int = 20,
        rrf_constant: int = 60,
    ) -> None:
        """由依赖工厂注入检索策略，禁止服务感知 Remote/Lite 或厂商 URL。"""
        if (
            mode not in {"bm25", "dense", "hybrid"}
            or not 1 <= recall_count <= 100
            or rrf_constant < 1
        ):
            raise ConfigurationError("检索策略或融合参数无效")
        if mode != "bm25" and (embedding is None or vector_store is None or profile is None):
            raise ConfigurationError("Dense/Hybrid 必须注入模型、向量端口和索引身份")
        if profile is not None:
            expected = self.index_profile(
                provider=profile.embedding_provider,
                model=profile.embedding_model,
                model_version=profile.embedding_model_version,
                dimension=profile.dimension,
                metric_type=profile.metric_type,
                endpoint=profile.embedding_endpoint,
            )
            if profile != expected or (
                embedding is not None and embedding.dimension != profile.dimension
            ):
                raise ConfigurationError("检索模型、来源或维度与索引身份不兼容")
        self._mode, self._embedding, self._vector, self._profile = (
            mode,
            embedding,
            vector_store,
            profile,
        )
        self._reranker, self._recall_count, self._rrf_constant = (
            reranker,
            recall_count,
            rrf_constant,
        )

    async def build_index(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStorePort,
        profile: IndexProfile,
        *,
        rebuild: bool = False,
    ) -> IndexReport:
        """通过唯一知识边界生成全量向量，模型失败时不修改已发布索引。"""
        expected = self.index_profile(
            provider=profile.embedding_provider,
            model=profile.embedding_model,
            model_version=profile.embedding_model_version,
            dimension=profile.dimension,
            metric_type=profile.metric_type,
            endpoint=profile.embedding_endpoint,
        )
        if profile != expected or embedding.dimension != profile.dimension:
            raise ConfigurationError("索引来源快照或模型维度不兼容")
        if not rebuild:
            existing = await vector_store.get_index(profile)
            if existing is not None:
                if existing.counts != self.counts():
                    raise KnowledgeRetrievalError("已发布索引数量与当前快照不一致")
                return existing
        records = list(self._records.values())
        vectors = await embedding.embed_documents([record.embedding_text for record in records])
        validate_vectors(vectors, len(records), profile.dimension)
        documents = [
            VectorDocument(record=record.model_copy(deep=True), vector=vector)
            for record, vector in zip(records, vectors, strict=True)
        ]
        return await vector_store.upsert(documents, profile, rebuild=rebuild)

    async def check_index(self) -> IndexReport | None:
        """启动时验证已发布快照，词项模式不要求向量索引。"""
        if self._mode == "bm25":
            return None
        if self._vector is None or self._profile is None:
            raise ConfigurationError("检索依赖未装配")
        report = await self._vector.get_index(self._profile)
        if report is None or report.counts != self.counts():
            raise KnowledgeRetrievalError("当前知识/模型快照尚未完整索引，请先运行索引脚本")
        return report

    def _evidence(
        self, record: KnowledgeRecord, score: float = 0.0, method: str = "exact"
    ) -> Evidence:
        """Evidence ID 由来源版本及真实 ID 派生，保留所有风险和审核字段。"""
        return Evidence(
            evidence_id=f"{record.knowledge_version}:{record.knowledge_id}",
            knowledge_id=record.knowledge_id,
            knowledge_type=record.knowledge_type,
            knowledge_version=record.knowledge_version,
            source=record.source,
            source_level=record.source_level,
            content=record.content,
            retrieval_score=score,
            metadata=deepcopy(record.metadata),
            retrieval_method=method,
        )

    def _lexical_search(
        self,
        kind: KnowledgeType,
        query: str,
        top_k: int,
        channels: list[str] | None = None,
        scopes: list[str] | None = None,
    ) -> list[Evidence]:
        """先按可靠 metadata 过滤，再词项召回；零命中不返回无关背景。"""
        if not 1 <= top_k <= 100:
            raise ValueError("召回数量必须在 1 至 100 之间")
        scored: list[tuple[float, KnowledgeRecord]] = []
        for record in self._records.values():
            if record.knowledge_type != kind:
                continue
            if channels and not set(channels).intersection(record.channels):
                continue
            if scopes and record.metadata.get("association_scope") not in scopes:
                continue
            score = self._retriever.score(query, record.knowledge_id)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda pair: (-pair[0], pair[1].knowledge_id))
        results = [self._evidence(record, score, "bm25") for score, record in scored[:top_k]]
        for evidence in results:
            evidence.channel_scores = {"bm25": evidence.retrieval_score}
        return results

    async def _search(
        self,
        kind: KnowledgeType,
        query: str,
        top_k: int,
        channels: list[str] | None = None,
        scopes: list[str] | None = None,
    ) -> list[Evidence]:
        """统一召回、去重和可选重排，故障明确抛出，不静默伪装另一检索方式。"""
        if not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1 至 50 之间")
        if not query.strip():
            return []
        started = perf_counter()
        await emit_progress("retrieval_started", {"knowledge_type": kind, "method": self._mode})
        recall_count = max(top_k, self._recall_count)
        rankings: dict[str, list[RetrievedDocument]] = {}
        if self._mode in {"bm25", "hybrid"}:
            rankings["bm25"] = list(
                await asyncio.to_thread(
                    self._lexical_search, kind, query, recall_count, channels, scopes
                )
            )
        if self._mode in {"dense", "hybrid"}:
            if self._embedding is None or self._vector is None or self._profile is None:
                raise ConfigurationError("检索依赖未装配")
            vector = await self._embedding.embed_query(query)
            rankings["dense"] = await self._vector.search(
                query_vector=vector,
                knowledge_type=kind,
                top_k=recall_count,
                profile=self._profile,
                filters={"channels": channels or [], "association_scope": scopes or []},
            )
        documents = (
            reciprocal_rank_fusion(rankings, recall_count, self._rrf_constant)
            if self._mode == "hybrid"
            else next(iter(rankings.values()))
        )
        if self._reranker is not None:
            documents = await self._reranker.rerank(query, documents, top_k)
        evidences = []
        for document in documents[:top_k]:
            record = self._records.get(document.knowledge_id)
            if (
                record is None
                or record.knowledge_version != document.knowledge_version
                or record.knowledge_type != kind
                or record.content != document.content
            ):
                raise KnowledgeRetrievalError("召回来源与当前知识快照不一致")
            evidence = self._evidence(record, document.retrieval_score, document.retrieval_method)
            evidence.rerank_score = document.rerank_score
            evidence.channel_scores = document.channel_scores.copy()
            evidences.append(evidence)
        await emit_progress(
            "retrieval_completed",
            {
                "knowledge_type": kind,
                "method": self._mode,
                "count": len(evidences),
                "duration_ms": int((perf_counter() - started) * 1000),
                "reranked": self._reranker is not None,
            },
        )
        return evidences

    async def get_clause_by_id(self, clause_id: str) -> Evidence | None:
        """按条文来源 ID 或完整条文号精确查询，不使用向量或生成模型。"""
        record = self._records.get(clause_id) or self._clauses.get(clause_id)
        return (
            self._evidence(record) if record and record.knowledge_type == "classic_clause" else None
        )

    async def get_evidence_by_ids(self, evidence_ids: list[str]) -> list[Evidence]:
        """精确展开当前版本的引用；缺失、跨版本或伪造 ID 明确失败，不再次模糊召回。"""
        results: list[Evidence] = []
        for evidence_id in dict.fromkeys(evidence_ids):
            version, separator, knowledge_id = evidence_id.partition(":")
            record = self._records.get(knowledge_id)
            if not separator or version != self.version or record is None:
                raise KnowledgeRetrievalError("路径引用不属于当前知识快照")
            if record.knowledge_version != version:
                raise KnowledgeRetrievalError("路径引用与来源版本不一致")
            results.append(self._evidence(record))
        return results

    async def get_formula_by_name(self, formula_name: str) -> Evidence | None:
        """按完整方名精确查询，不用模糊匹配替代精确展开。"""
        for record in self._records.values():
            if (
                record.knowledge_type == "formula_pattern"
                and record.metadata["formula_name"] == formula_name
            ):
                return self._evidence(record)
        return None

    async def identify_query_targets(self, query: str) -> tuple[list[str], list[str]]:
        """从只读来源匹配完整方名和鉴别标签；长名优先，避免大方名包含小方名的误取。"""
        names = {
            str(record.metadata["formula_name"])
            for record in self._records.values()
            if record.knowledge_type == "formula_pattern"
        }
        labels = {
            label
            for record in self._records.values()
            if record.knowledge_type == "differential"
            for label in TypeAdapter(list[str]).validate_python(record.metadata["candidates"])
        }

        def match(values: set[str]) -> list[str]:
            """按完整来源词匹配；不把词面别名当作精确方名。"""
            matches: list[str] = []
            for value in sorted(values, key=lambda item: (-len(item), item)):
                if value in query and not any(value in longer for longer in matches):
                    matches.append(value)
            return sorted(matches, key=query.index)

        return match(names), match(labels)

    async def search_core_framework(self, query: str, top_k: int = 5) -> list[Evidence]:
        """检索 L0 框架原则，返回真实来源文本。"""
        return await self._search("six_channel_framework", query, top_k)

    async def search_classic(
        self,
        query: str,
        semantic_channels: list[str] | None = None,
        clause_nos: list[str] | None = None,
        top_k: int = 8,
    ) -> list[Evidence]:
        """精确条文号优先，普通查询使用六经元数据和配置化召回。"""
        if not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1 至 50 之间")
        if clause_nos is not None:
            result = [await self.get_clause_by_id(number) for number in dict.fromkeys(clause_nos)]
            return [e for e in result if e is not None][:top_k]
        return await self._search("classic_clause", query, top_k, semantic_channels)

    async def search_formula(
        self,
        query: str,
        channels: list[str] | None = None,
        association_scope: list[str] | None = None,
        top_k: int = 8,
    ) -> list[Evidence]:
        """分别保留经典与医案派生归属，风险标签随 Evidence 返回。"""
        return await self._search("formula_pattern", query, top_k, channels, association_scope)

    def _rule(self, evidence: Evidence) -> DifferentialRule:
        """从真实来源字段构造规则，不由模型生成 decision_rules。"""
        raw = evidence.metadata
        return DifferentialRule(
            **evidence.model_dump(),
            rule_id=evidence.knowledge_id,
            candidates=TypeAdapter(list[str]).validate_python(raw["candidates"]),
            key_dimensions=TypeAdapter(dict[str, str]).validate_python(raw["key_dimensions"]),
            decision_rules=TypeAdapter(list[str]).validate_python(raw["decision_rules"]),
        )

    async def get_differential_by_id(self, rule_id: str) -> DifferentialRule | None:
        """按来源规则 ID 精确读取鉴别维度。"""
        record = self._records.get(rule_id)
        return (
            self._rule(self._evidence(record))
            if record and record.knowledge_type == "differential"
            else None
        )

    async def search_differential(
        self,
        candidates: list[str],
        query: str | None = None,
        top_k: int = 5,
    ) -> list[DifferentialRule]:
        """候选对精确、完整覆盖的组规则次之，再做有限语义召回并按 ID 去重。"""
        if not 1 <= top_k <= 50:
            raise ValueError("top_k 必须在 1 至 50 之间")
        exact: list[DifferentialRule] = []
        groups: list[DifferentialRule] = []
        for record in self._records.values():
            labels = (
                TypeAdapter(list[str]).validate_python(record.metadata["candidates"])
                if record.knowledge_type == "differential"
                else []
            )
            bound = [source_candidate_label(candidate, labels) for candidate in candidates]
            if (
                record.knowledge_type == "differential"
                and len(set(candidates)) >= 2
                and None not in bound
                and len(set(bound)) == len(candidates)
            ):
                target = exact if len(candidates) == len(labels) else groups
                method = "candidate_pair_exact" if target is exact else "candidate_group_exact"
                target.append(self._rule(self._evidence(record, method=method)))
        precise = exact + groups
        if len(precise) >= top_k:
            return precise[:top_k]
        related = await self._search("differential", query or " ".join(candidates), top_k)
        seen = {rule.knowledge_id for rule in precise}
        return (precise + [self._rule(e) for e in related if e.knowledge_id not in seen])[:top_k]

    async def resolve_differential_candidates(self, candidates: list[str]) -> dict[str, str]:
        """将同名或明确缩写映射到真实来源标签；跨规则有歧义时保留输入名。"""
        labels = {
            label
            for record in self._records.values()
            if record.knowledge_type == "differential"
            for label in TypeAdapter(list[str]).validate_python(record.metadata["candidates"])
        }
        resolved: dict[str, str] = {}
        for candidate in candidates:
            resolved[candidate] = source_candidate_label(candidate, list(labels)) or candidate
        return resolved

    async def search_case(
        self,
        query: str,
        channels: list[str] | None = None,
        top_k: int = 5,
    ) -> list[Evidence]:
        """查询 L5 历史病例，不提升为经典依据或患者处方。"""
        return await self._search("medical_case", query, top_k, channels)

    async def assess_candidate_evidence(
        self,
        evidence_ids: list[str],
        clinical: ClinicalEvidence,
        *,
        channels: list[str],
        formula_name: str | None = None,
    ) -> list[EvidenceAssessment]:
        """精确读取来源后按当前路径分级，杜绝调用方用伪造内容或 ID 背书。"""
        evidences = await self.get_evidence_by_ids(evidence_ids)
        return [
            grade_candidate_evidence(item, clinical, channels=channels, formula_name=formula_name)
            for item in evidences
        ]

    async def retrieve_evidence_bundle(self, query: str, top_k: int = 5) -> EvidenceBundle:
        """分层返回五类证据，保留 ID、版本及来源，不拼成无出处文本。"""
        return EvidenceBundle(
            framework=await self.search_core_framework(query, top_k),
            classic=await self.search_classic(query, top_k=top_k),
            formulas=await self.search_formula(query, top_k=top_k),
            differential=await self.search_differential([], query, top_k),
            cases=await self.search_case(query, top_k=top_k),
        )
