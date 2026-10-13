"""五类知识库校验入口，只通过 KnowledgeService 访问，不修改来源文件。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService


async def main() -> None:
    """加载校验并报告来源数量；异常自然产生非零退出码。"""
    settings = Settings()
    service = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    print(
        json.dumps(
            {
                "version": service.version,
                "counts": service.counts(),
                "snapshot_digest": service.snapshot_digest,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
