"""V1.2 检索联调：验证 Hybrid 链路或单独探测已配置重排器。

依赖 Settings、KnowledgeService 和 Provider 工厂，不枚举厂商或切换模型。
只输出来源 ID 和分数，不输出凭据、不改知识和业务数据；运行会调用在线服务。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.core.dependencies import open_embedding_provider, open_reranker
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.reranker import Reranker
from backend.app.rag.vector_store import build_milvus_store
from backend.app.schemas.evidence import RetrievedDocument


async def probe_reranker(knowledge: KnowledgeService, reranker: Reranker) -> dict[str, object]:
    """验证配置模型的来源与返回分数，不探测其他厂商模型。"""
    evidence = [await knowledge.get_clause_by_id(str(number)) for number in (1, 2, 35)]
    documents = [
        RetrievedDocument.model_validate(item.model_dump(exclude={"evidence_id"}))
        for item in evidence
        if item is not None
    ]
    if len(documents) != 3:
        raise ValueError("重排探测缺少真实经典来源")
    results = await reranker.rerank("脉浮恶寒汗出恶风", documents, top_k=len(documents))
    original = {item.knowledge_id: item.content for item in documents}
    # 重排只能改变来源顺序，不能产生新证据或改写原文。
    if len(results) != len(documents) or {item.knowledge_id for item in results} != set(original):
        raise ValueError("重排探测来源不完整")
    if any(original.get(item.knowledge_id) != item.content for item in results):
        raise ValueError("重排探测来源被改写")
    return {
        "mode": "reranker_probe",
        "source_preserved": True,
        "results": [
            {"knowledge_id": item.knowledge_id, "rerank_score": item.rerank_score}
            for item in results
        ],
    }


async def verify(settings: Settings, *, reranker_only: bool = False) -> dict[str, object]:
    """经工厂验证服务，独立重排模式不连接 Embedding 或 Milvus。"""
    service = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    if reranker_only:
        async with open_reranker(settings) as reranker:
            if reranker is None:
                raise ValueError("重排探测需要 RERANKER_ENABLED=true")
            return await probe_reranker(service, reranker)
    vector_store = await build_milvus_store(settings)
    try:
        async with (
            open_embedding_provider(settings) as embedding,
            open_reranker(settings) as reranker,
        ):
            profile = service.index_profile(
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                model_version=settings.embedding_model_version,
                dimension=settings.embedding_dimension,
                metric_type=settings.milvus_metric_type,
                endpoint=settings.embedding_base_url,
            )
            service.configure_retrieval(
                mode="hybrid",
                embedding=embedding,
                vector_store=vector_store,
                profile=profile,
                reranker=reranker,
                recall_count=settings.retrieval_top_k,
                rrf_constant=settings.fusion_rrf_constant,
            )
            await service.check_index()
            results = await service.search_classic(
                "脉浮恶寒汗出恶风", semantic_channels=["太阳"], top_k=3
            )
            return {
                "mode": "hybrid",
                "vector_backend": vector_store.status.backend,
                "retrieval_method": [item.retrieval_method for item in results],
                "results": [
                    {
                        "knowledge_id": item.knowledge_id,
                        "evidence_id": item.evidence_id,
                        "rerank_score": item.rerank_score,
                        "source_preserved": bool(item.source and item.content),
                    }
                    for item in results
                ],
            }
    finally:
        await vector_store.close()


def main() -> None:
    """执行一次真实联调，服务失败保持非零退出。"""
    parser = argparse.ArgumentParser(description="V1.2 Hybrid / 配置重排器联调")
    parser.add_argument("--probe-reranker", action="store_true", help="只测试配置重排器")
    args = parser.parse_args()
    result = asyncio.run(verify(Settings(), reranker_only=args.probe_reranker))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
