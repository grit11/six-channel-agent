"""S4 批量 LLM 方证比较：模型负责相对判断，程序只核验候选、事实和来源身份。

鉴别资料作为真实来源上下文传入；不执行旧 Python 规则、不计算医学分数、不自动改写模型顺序。
"""

import asyncio
import json
from pathlib import Path

from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.retrieval import _evidence_ref
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    ComparisonV2,
    RetrievalBundleV2,
)
from backend.app.core.exceptions import ApplicationError, ProviderUnavailableError
from backend.app.core.ports import StructuredLLM
from backend.app.knowledge.service import KnowledgeService


class ComparisonUngroundedError(ApplicationError):
    """S4 引用不存在或模型生成候选时拒绝发布。"""

    code = "COMPARISON_UNGROUNDED"


async def compare_formula_candidates(
    evidence: ClinicalEvidenceV2,
    localization: ChannelLocalizationV2,
    retrieval: RetrievalBundleV2,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    differential_top_k: int = 10,
) -> ComparisonV2:
    """一次结构化调用比较 S3 候选；程序签发 grounded 状态。"""
    guide_dir = Path(__file__).parent / "prompts"
    prompt, guide = await asyncio.gather(
        asyncio.to_thread((guide_dir / "s4_formula_comparison.md").read_text, encoding="utf-8"),
        asyncio.to_thread(
            (guide_dir / "formula_discrimination_guide.md").read_text, encoding="utf-8"
        ),
    )
    candidates = {item.candidate_id: item for item in retrieval.formula_candidates}
    if not candidates:
        return ComparisonV2(
            status="insufficient",
            ordered_paths=[],
            unresolved_reason="没有真实方证候选",
        )
    differential = await knowledge.search_differential(
        list(item.formula_name for item in candidates.values()),
        query=retrieval.query_plan.differential_query,
        top_k=differential_top_k,
    )
    payload = {
        "clinical_evidence": evidence.model_dump(mode="json"),
        "localization": localization.model_dump(mode="json"),
        "retrieval": retrieval.model_dump(mode="json"),
        "differential_sources": [item.model_dump(mode="json") for item in differential],
    }
    try:
        draft = await llm.ainvoke_structured(
            [
                {"role": "system", "content": prompt + "\n" + guide},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            ComparisonV2,
        )
    except ProviderUnavailableError as exc:
        raise ClinicalLLMUnavailableError("S4 模型不可用，无法生成证据比较") from exc
    fact_ids = {item.fact_id for item in evidence.findings}
    known_ids = {item.fact_id for item in evidence.findings if item.polarity != "unknown"}
    evidence_ids = {
        ref.evidence_id for candidate in candidates.values() for ref in candidate.source_evidence
    } | {item.evidence_id for item in differential}
    seen_candidates: set[str] = set()
    paths = []
    for position, path in enumerate(draft.ordered_paths, 1):
        if path.candidate_id not in candidates or path.candidate_id in seen_candidates:
            raise ComparisonUngroundedError("比较结果引用了不存在或重复的方证候选")
        seen_candidates.add(path.candidate_id)
        if not set(path.support_fact_ids + path.counter_fact_ids).issubset(known_ids):
            raise ComparisonUngroundedError("比较结果引用了不存在或未知的事实")
        if not set(path.support_evidence_ids + path.counter_evidence_ids).issubset(evidence_ids):
            raise ComparisonUngroundedError("比较结果引用了不存在的证据")
        expected_name = candidates[path.candidate_id].formula_name
        if path.formula_name != expected_name:
            raise ComparisonUngroundedError("比较结果方名与候选池不一致")
        candidate_sources = {
            ref.evidence_id for ref in candidates[path.candidate_id].source_evidence
        }
        if not set(path.support_evidence_ids) & candidate_sources:
            raise ComparisonUngroundedError("路径缺少当前候选自身的来源支持")
        if draft.status == "ranked" and path.rank != position:
            raise ComparisonUngroundedError("路径顺序与 rank 不一致")
        paths.append(path.model_copy(update={"verification_status": "grounded"}))
    if draft.status == "ranked" and not paths:
        raise ComparisonUngroundedError("ranked 结果必须包含至少一条路径")
    if len({path.path_id for path in paths}) != len(paths):
        raise ComparisonUngroundedError("比较路径身份重复")
    if any(not set(q.source_fact_ids).issubset(fact_ids) for q in draft.next_questions):
        raise ComparisonUngroundedError("补问引用了不存在的事实")
    # 来源由程序附加，不能相信模型自行生成的摘录或校验标志。
    refs = {
        ref.evidence_id: ref
        for candidate in candidates.values()
        for ref in candidate.source_evidence
    }
    refs.update({item.evidence_id: _evidence_ref(item) for item in differential})
    return draft.model_copy(update={"ordered_paths": paths, "source_evidence": list(refs.values())})
