"""中文词项 BM25 检索，独立提供证据排序，不宣称模型语义检索或患者诊断。"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from backend.app.schemas.knowledge import KnowledgeRecord


def tokenize(text: str) -> list[str]:
    """使用中文二字片段及英文数字词项，避免依赖外部分词词典。"""
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text.lower()):
        if re.fullmatch(r"[a-zA-Z0-9_]+", segment) or len(segment) == 1:
            tokens.append(segment)
        else:
            # 单字医学查询也要能命中文档；二字片段保留更高区分信息。
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


class BM25Retriever:
    """基于审核后检索文本的词项召回，模型不可用时不会伪造 Dense 分数。"""

    def __init__(self, records: Sequence[KnowledgeRecord]) -> None:
        """预计算词频、长度和文档频率，按来源 ID 保存索引。"""
        self._terms = {r.knowledge_id: Counter(tokenize(r.embedding_text)) for r in records}
        self._frequency: Counter[str] = Counter()
        for terms in self._terms.values():
            self._frequency.update(terms.keys())
        self._count = len(records)
        self._average = sum(sum(v.values()) for v in self._terms.values()) / max(1, self._count)

    def score(self, query: str, knowledge_id: str) -> float:
        """返回非概率的词项相关分；没有词项命中时返回零。"""
        terms = self._terms[knowledge_id]
        length = sum(terms.values())
        score = 0.0
        for term in set(tokenize(query)):
            frequency = terms.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                1 + (self._count - self._frequency[term] + 0.5) / (self._frequency[term] + 0.5)
            )
            score += (
                inverse
                * frequency
                * 2.5
                / (frequency + 1.5 * (0.25 + 0.75 * length / max(1, self._average)))
            )
        return score
