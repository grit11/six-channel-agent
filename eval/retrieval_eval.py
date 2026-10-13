"""检索工程回归：398 条精确条文、53 条鉴别候选对，不代替临床质量评测。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService


async def main() -> None:
    """对当前版本全量来源进行工程回归，任一缺失引用即失败。"""
    settings = Settings()
    service = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    clause_hits = 0
    for number in range(1, 399):
        evidence = await service.get_clause_by_id(str(number))
        if evidence is not None:
            assert evidence.content == evidence.metadata["classic_text"]
            clause_hits += 1
    pair_hits = 0
    for record in service.raw_documents("differential"):
        candidates = record.metadata.get("candidates", [])
        results = await service.search_differential(candidates, top_k=5)
        pair_hits += int(any(result.knowledge_id == record.knowledge_id for result in results))
    print(
        json.dumps(
            {
                "retrieval_backend": "bm25_and_exact",
                "exact_clause_hits": clause_hits,
                "clause_count": 398,
                "differential_pair_recall_at_5": pair_hits / 53,
                "scope": "工程回归，非独立语义或临床评测",
            },
            ensure_ascii=False,
        )
    )
    assert clause_hits == 398 and pair_hits == 53


if __name__ == "__main__":
    asyncio.run(main())
