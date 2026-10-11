"""证据排序融合：依赖标准检索契约，用 RRF 去重，分数不代表诊断概率。"""

from collections import defaultdict

from backend.app.schemas.evidence import RetrievedDocument


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievedDocument]], top_k: int, constant: int = 60
) -> list[RetrievedDocument]:
    """按排名融合异尺度分数，保留各召回通道原始分数及真实来源。"""
    if top_k < 1 or constant < 1:
        raise ValueError("融合参数必须大于零")
    scores: dict[tuple[str, str], float] = defaultdict(float)
    originals: dict[tuple[str, str], RetrievedDocument] = {}
    channel_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for channel, documents in rankings.items():
        seen: set[tuple[str, str]] = set()
        for rank, document in enumerate(documents, 1):
            key = (document.knowledge_version, document.knowledge_id)
            if key in seen:
                continue
            seen.add(key)
            originals.setdefault(key, document)
            # BM25 与向量分不可直接相加；同一来源在不同通道命中才累积排名贡献。
            scores[key] += 1.0 / (constant + rank)
            channel_scores[key][channel] = document.retrieval_score
    keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [
        originals[key].model_copy(
            deep=True,
            update={
                "retrieval_score": scores[key],
                "retrieval_method": "hybrid_rrf",
                "channel_scores": channel_scores[key],
            },
        )
        for key in keys
    ]
