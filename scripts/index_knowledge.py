"""知识索引命令：依赖 KnowledgeService 与适配工厂，支持试运行，不修改医学来源。"""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.core.dependencies import open_embedding_provider
from backend.app.core.exceptions import ApplicationError
from backend.app.knowledge.service import KnowledgeService
from backend.app.rag.embedding import validate_embedding_config
from backend.app.rag.vector_store import build_milvus_store


async def run_index(
    settings: Settings, *, rebuild: bool = False, dry_run: bool = False
) -> dict[str, object]:
    """校验来源后索引；dry-run 不访问模型或 Milvus，不产生费用和写入。"""
    service = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    validate_embedding_config(settings)
    profile = service.index_profile(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        model_version=settings.embedding_model_version,
        dimension=settings.embedding_dimension,
        metric_type=settings.milvus_metric_type,
        endpoint=settings.embedding_base_url,
    )
    if dry_run:
        return {
            "dry_run": True,
            "profile": profile.model_dump(),
            "counts": service.counts(),
            "total": sum(service.counts().values()),
            "rebuild": rebuild,
            "configured_backend": settings.milvus_mode,
        }
    async with AsyncExitStack() as stack:
        vector = await build_milvus_store(settings)
        stack.push_async_callback(vector.close)
        embedding = await stack.enter_async_context(open_embedding_provider(settings))
        report = await service.build_index(embedding, vector, profile, rebuild=rebuild)
        return report.model_dump(mode="json")


def main() -> None:
    """解析版本与重建参数，仅输出安全报告或业务错误，不打印异常凭据。"""
    parser = argparse.ArgumentParser(description="五类知识版本化向量索引")
    parser.add_argument("--knowledge-version", help="知识来源版本，默认使用配置")
    parser.add_argument("--rebuild", action="store_true", help="新建完整集合后发布，保留旧集合")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅校验来源及模型配置，不调用外部服务"
    )
    args = parser.parse_args()
    try:
        overrides = (
            {"knowledge_base_version": args.knowledge_version} if args.knowledge_version else {}
        )
        report = asyncio.run(
            run_index(Settings(**overrides), rebuild=args.rebuild, dry_run=args.dry_run)
        )
    except ApplicationError as exc:
        print(
            json.dumps({"error": {"code": exc.code, "message": exc.message}}, ensure_ascii=False),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
