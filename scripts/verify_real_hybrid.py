"""真实 Hybrid 验证：硅基流动 Embedding、Milvus Lite、RRF 与 Reranker 串联。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.core.dependencies import open_embedding_provider, open_reranker
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.vector_store import build_milvus_store


async def main() -> None:
    """执行真实检索并只输出来源 ID、方法和分数，不输出密钥或患者信息。"""
    settings = Settings()
    service = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
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
            print(
                json.dumps(
                    {
                        "vector_backend": vector_store.status.backend,
                        "embedding_model": settings.embedding_model,
                        "reranker_model": settings.reranker_model,
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
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
