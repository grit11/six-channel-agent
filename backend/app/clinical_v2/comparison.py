"""S4 批量 LLM 方证比较：模型负责相对判断，程序只核验候选、事实和来源身份。

本模块只负责 S4 的证据上下文裁剪、结构化模型调用和来源绑定；完整知识记录仍由
KnowledgeService 按 Evidence ID 提供展开，不在模型请求中重复发送检索 DTO、轨迹和冗余元数据。
不执行旧 Python 规则、不计算医学分数；仅按模型合法 rank 整理数组，不重判医学优先级。
"""

import asyncio
import hashlib
import json
from pathlib import Path
from time import perf_counter

from backend.app.clinical_v2.extraction import ClinicalLLMUnavailableError
from backend.app.clinical_v2.retrieval import _evidence_ref
from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ClinicalEvidenceV2,
    CandidateComparisonDraftV2,
    CandidateComparisonV2,
    ComparisonDraftV2,
    ComparisonV2,
    FormulaEvidenceCandidateV2,
    RetrievalBundleV2,
)
from backend.app.core.exceptions import ApplicationError, ProviderUnavailableError, S4StreamError
from backend.app.core.ports import S4StreamPolicy, StreamUsage, StructuredLLM
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.evidence import Evidence


class ComparisonUngroundedError(ApplicationError):
    """S4 引用不存在或模型生成候选时拒绝发布。"""

    code = "COMPARISON_UNGROUNDED"

    def __init__(
        self,
        message: str,
        *,
        validation_rule: str,
        field_path: str | None = None,
        item_index: int | None = None,
    ) -> None:
        """只暴露安全校验规则和路径索引，不保存模型正文。"""
        details: dict[str, object] = {"validation_rule": validation_rule}
        if field_path is not None:
            details["field_path"] = field_path
        if item_index is not None:
            details["item_index"] = item_index
        super().__init__(message, details=details)


def _clip_text(value: str, limit: int) -> str:
    """限制模型上下文中的摘录长度，同时保留来源可回查的 Evidence ID。"""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    # 不截断半句后把残片当完整条件；找不到完整句界时保留全文。
    end = max(text.rfind(mark, 0, limit) for mark in ("。", "！", "？", "\n"))
    return text[: end + 1] + "…（摘录未完整，未展示内容不等于不存在）" if end >= 0 else text


def _compact_clinical_evidence(evidence: ClinicalEvidenceV2) -> dict[str, object]:
    """只保留 S4 比较所需的事实身份、原文表现、极性和时点。"""
    return {
        "findings": [
            {
                "fact_id": item.fact_id,
                "concept": item.concept,
                "polarity": item.polarity,
                "time_scope": item.time_scope,
                "source_text": item.source_span.text,
                "review_status": item.review_status,
            }
            for item in evidence.findings
        ],
        "conflicts": evidence.conflicts,
        "unresolved_phrases": [item.text for item in evidence.unresolved_phrases],
        "doctor_revision": evidence.doctor_revision,
    }


def _compact_localization(localization: ChannelLocalizationV2) -> dict[str, object]:
    """只保留六经候选及其事实依据，避免把完整 S2 DTO重复送入 S4。"""
    return {
        "status": localization.status,
        "candidates": [
            {
                "candidate_id": item.candidate_id,
                "six_channels": item.six_channels,
                "support_fact_ids": item.support_fact_ids,
                "counter_fact_ids": item.counter_fact_ids,
                "missing_dimensions": item.missing_dimensions,
                "retrieval_focus": item.retrieval_focus,
                "short_basis": item.short_basis,
            }
            for item in localization.candidates
        ],
        "note": localization.note,
    }


def _compact_retrieval(retrieval: RetrievalBundleV2) -> dict[str, object]:
    """保留候选名称、来源 ID 和短摘录，排除检索轨迹及原始分数元数据。"""
    return {
        "formula_candidates": [
            {
                "candidate_id": item.candidate_id,
                "formula_name": item.formula_name,
                "six_channel_tags": item.six_channel_tags,
                "provenance": item.provenance,
                "coverage_notes": item.coverage_notes,
                "source_evidence": [
                    {
                        "evidence_id": ref.evidence_id,
                        "knowledge_type": ref.knowledge_type,
                        "knowledge_version": ref.knowledge_version,
                        "clause_no": ref.clause_no,
                        "excerpt": _clip_text(ref.excerpt, 500),
                        "review_status": ref.review_status,
                    }
                    for ref in item.source_evidence
                ],
            }
            for item in retrieval.formula_candidates
        ],
        "retrieval_status": retrieval.retrieval_status,
    }


def _compact_differential_sources(differential: list[Evidence]) -> list[dict[str, object]]:
    """从鉴别 Evidence 提取短摘录和关键维度，不发送完整 metadata。"""
    compact: list[dict[str, object]] = []
    for item in differential:
        metadata = item.metadata
        dimensions = metadata.get("key_dimensions", {})
        if isinstance(dimensions, dict) and dimensions:
            # 关键鉴别条件整体保留，不按患者词项筛选或只留下支持条件。
            key_dimensions = {str(key): value for key, value in dimensions.items()}
        else:
            key_dimensions = {}
        compact.append(
            {
                "evidence_id": item.evidence_id,
                "knowledge_type": item.knowledge_type,
                "knowledge_version": item.knowledge_version,
                "source_level": item.source_level,
                "review_status": metadata.get("review_status", "unknown"),
                "clinical_use_policy": metadata.get("clinical_use_policy", ""),
                "red_flags": metadata.get("red_flags", []),
                "title": metadata.get("title", ""),
                "candidates": metadata.get("candidates", []),
                "key_dimensions": key_dimensions,
                # 结构化鉴别维度与 content/decision_rules/text_for_embedding 重复，
                # 只发送一份；其他知识版本没有关键维度时保留来源正文，避免丢失依据。
                "excerpt": "" if key_dimensions else item.content,
            }
        )
    return compact


def _path_id(consultation_id: str, revision: int, candidate_id: str) -> str:
    """按会诊、修订和候选身份生成稳定路径 ID，模型不能伪造或重复它。"""
    digest = hashlib.sha256(
        f"{consultation_id}:{revision}:{candidate_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"path-{digest}"


def _publish_path(
    draft_path: CandidateComparisonDraftV2,
    candidate: FormulaEvidenceCandidateV2,
    *,
    consultation_id: str,
    revision: int,
) -> CandidateComparisonV2:
    """把通过语义和来源校验的草稿绑定为下游兼容发布对象。"""
    path = draft_path
    item = candidate
    return CandidateComparisonV2(
        path_id=_path_id(consultation_id, revision, path.candidate_id),
        candidate_id=path.candidate_id,
        six_channels=path.six_channels,
        formula_name=item.formula_name,
        rank=path.rank,
        support_fact_ids=path.support_fact_ids,
        counter_fact_ids=path.counter_fact_ids,
        missing_dimensions=path.missing_dimensions,
        support_evidence_ids=path.support_evidence_ids,
        counter_evidence_ids=path.counter_evidence_ids,
        brief_rationale=path.brief_rationale,
        differential_summary=path.differential_summary,
        verification_status="grounded",
    )


def _publish_comparison_draft(
    draft: ComparisonDraftV2,
    evidence: ClinicalEvidenceV2,
    retrieval: RetrievalBundleV2,
    differential: list[Evidence],
    *,
    consultation_id: str,
    revision: int,
) -> ComparisonV2:
    """校验一次模型草稿并签发兼容的 S4 发布对象。"""
    candidates = {item.candidate_id: item for item in retrieval.formula_candidates}
    fact_ids = {item.fact_id for item in evidence.findings}
    known_ids = {item.fact_id for item in evidence.findings if item.polarity != "unknown"}
    evidence_ids = {
        ref.evidence_id for candidate in candidates.values() for ref in candidate.source_evidence
    } | {item.evidence_id for item in differential}
    seen_candidates: set[str] = set()
    paths: list[CandidateComparisonV2] = []
    ordered_paths = draft.ordered_paths
    if draft.status == "ranked":
        ranks = [path.rank for path in ordered_paths]
        expected_ranks = set(range(1, len(ordered_paths) + 1))
        if not ranks or None in ranks or set(ranks) != expected_ranks:
            raise ComparisonUngroundedError(
                "ranked 排名必须从 1 开始且唯一连续，不能缺失或并列",
                validation_rule="noncontiguous_rank",
                field_path="ordered_paths.rank",
            )
        ordered_paths = sorted(ordered_paths, key=lambda path: path.rank or 0)
    for index, path in enumerate(ordered_paths):
        field_path = f"ordered_paths[{index}]"
        if path.candidate_id not in candidates or path.candidate_id in seen_candidates:
            raise ComparisonUngroundedError(
                "比较结果引用了不存在或重复的方证候选",
                validation_rule=(
                    "candidate_not_in_pool"
                    if path.candidate_id not in candidates
                    else "candidate_repeated"
                ),
                field_path=f"{field_path}.candidate_id",
                item_index=index,
            )
        seen_candidates.add(path.candidate_id)
        if not set(path.support_fact_ids + path.counter_fact_ids).issubset(known_ids):
            raise ComparisonUngroundedError(
                "比较结果引用了不存在或未知的事实",
                validation_rule="fact_id_not_allowed",
                field_path=f"{field_path}.support_fact_ids",
                item_index=index,
            )
        if not set(path.support_evidence_ids + path.counter_evidence_ids).issubset(evidence_ids):
            raise ComparisonUngroundedError(
                "比较结果引用了不存在的证据",
                validation_rule="evidence_id_not_allowed",
                field_path=f"{field_path}.support_evidence_ids",
                item_index=index,
            )
        candidate_sources = {
            ref.evidence_id for ref in candidates[path.candidate_id].source_evidence
        }
        if not set(path.support_evidence_ids) & candidate_sources:
            raise ComparisonUngroundedError(
                "路径缺少当前候选自身的来源支持",
                validation_rule="candidate_source_missing",
                field_path=f"{field_path}.support_evidence_ids",
                item_index=index,
            )
        paths.append(
            _publish_path(
                path,
                candidates[path.candidate_id],
                consultation_id=consultation_id,
                revision=revision,
            )
        )
    if draft.status == "ranked" and not paths:
        raise ComparisonUngroundedError(
            "ranked 结果必须包含至少一条路径", validation_rule="ranked_empty"
        )
    if draft.status != "ranked" and any(path.rank is not None for path in ordered_paths):
        raise ComparisonUngroundedError(
            "未决状态不得携带正式排名",
            validation_rule="provisional_rank_forbidden",
            field_path="ordered_paths.rank",
        )
    if any(not set(q.source_fact_ids).issubset(fact_ids) for q in draft.next_questions):
        raise ComparisonUngroundedError(
            "补问引用了不存在的事实",
            validation_rule="question_reference_invalid",
            field_path="next_questions.source_fact_ids",
        )
    refs = {
        ref.evidence_id: ref
        for candidate in candidates.values()
        for ref in candidate.source_evidence
    }
    refs.update({item.evidence_id: _evidence_ref(item) for item in differential})
    return ComparisonV2(
        status=draft.status,
        ordered_paths=paths if draft.status == "ranked" else [],
        provisional_candidates=paths if draft.status != "ranked" else [],
        next_questions=draft.next_questions,
        unresolved_reason=draft.unresolved_reason,
        source_evidence=list(refs.values()),
    )


async def compare_formula_candidates(
    evidence: ClinicalEvidenceV2,
    localization: ChannelLocalizationV2,
    retrieval: RetrievalBundleV2,
    knowledge: KnowledgeService,
    llm: StructuredLLM,
    *,
    differential_top_k: int = 10,
    s4_policy: S4StreamPolicy | None = None,
    consultation_id: str = "unknown-consultation",
    revision: int = 0,
) -> ComparisonV2:
    """一次结构化调用比较 S3 候选；程序签发 grounded 状态。"""
    guide_dir = Path(__file__).parent / "prompts"
    prompt, guide, few_shot = await asyncio.gather(
        asyncio.to_thread((guide_dir / "s4_formula_comparison.md").read_text, encoding="utf-8"),
        asyncio.to_thread(
            (guide_dir / "formula_discrimination_guide.md").read_text, encoding="utf-8"
        ),
        asyncio.to_thread((guide_dir / "s4_few_shot_v2_1.md").read_text, encoding="utf-8"),
    )
    candidates = {item.candidate_id: item for item in retrieval.formula_candidates}
    if not candidates:
        return ComparisonV2(
            status="insufficient",
            ordered_paths=[],
            provisional_candidates=[],
            unresolved_reason="没有真实方证候选",
        )
    differential = await knowledge.search_differential(
        list(item.formula_name for item in candidates.values()),
        query=retrieval.query_plan.differential_query,
        top_k=differential_top_k,
    )
    # S4 只接收紧凑证据上下文；完整 DTO、检索轨迹和来源 metadata 通过 Evidence ID
    # 保留可回查性，但不重复占用模型上下文窗口。
    payload = {
        "context_format": "compact_evidence_context_v1",
        "clinical_evidence": _compact_clinical_evidence(evidence),
        "localization": _compact_localization(localization),
        "retrieval": _compact_retrieval(retrieval),
        "differential_sources": _compact_differential_sources(differential),
    }
    messages = [
        {"role": "system", "content": prompt + "\n" + guide + "\n" + few_shot},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    from backend.app.observability.progress import emit_progress

    async def invoke_draft(
        request_messages: list[dict[str, str]], policy: S4StreamPolicy | None
    ) -> ComparisonDraftV2:
        """执行一次 S4 草稿调用；不保存正文，只返回已校验草稿。"""
        try:
            if policy is not None and hasattr(llm, "ainvoke_structured_stream"):
                await emit_progress("s4_stream_started", {"stream_status": "waiting"})

                async def on_progress(usage: StreamUsage) -> None:
                    await emit_progress(
                        "s4_stream_progress",
                        {
                            "elapsed_ms": usage.elapsed_ms,
                            "received_content_chars": usage.output_chars,
                            "content_chunks": usage.content_chunks,
                            "usage_status": usage.usage_status,
                        },
                    )

                result = await llm.ainvoke_structured_stream(  # type: ignore[attr-defined]
                    request_messages, ComparisonDraftV2, policy=policy, on_progress=on_progress
                )
            else:
                result = await llm.ainvoke_structured(request_messages, ComparisonDraftV2)
        except ProviderUnavailableError as exc:
            if isinstance(exc, S4StreamError):
                await emit_progress(
                    "s4_stream_failed",
                    {
                        "failure_code": exc.failure_code,
                        "fallback_available": True,
                        **{
                            key: exc.details[key]
                            for key in (
                                "provider",
                                "model",
                                "response_format_enabled",
                                "stream_options_enabled",
                                "finish_reason",
                                "saw_done",
                                "content_chars",
                                "http_status",
                                "error_category",
                                "thinking_type",
                                "reasoning_effort",
                                "provider_error_code",
                                "provider_error_param",
                            )
                            if key in exc.details
                        },
                    },
                )
                raise ClinicalLLMUnavailableError(
                    "S4 流式模型不可用，无法生成证据比较", details=dict(exc.details)
                ) from exc
            details = dict(exc.details) if isinstance(exc, ApplicationError) else {}
            if details.get("error_category") in {"schema_validation", "json_invalid"}:
                details["failure_code"] = "S4_INVALID_RESPONSE"
            else:
                details.setdefault("failure_code", "S4_PROVIDER_UNAVAILABLE")
            raise ClinicalLLMUnavailableError(
                "S4 模型不可用，无法生成证据比较", details=details
            ) from exc
        return result if isinstance(result, ComparisonDraftV2) else ComparisonDraftV2.model_validate(result)

    started = perf_counter()
    repair_used = False
    try:
        draft = await invoke_draft(messages, s4_policy)
    except ClinicalLLMUnavailableError as exc:
        # 流/非流结构校验失败没有把完整正文写入异常；只做一次无正文的格式重试。
        if exc.details.get("failure_code") != "S4_INVALID_RESPONSE":
            raise
        repair_used = True
        repair_messages = [
            *messages,
            {
                "role": "user",
                "content": "上次响应未通过 S4 草稿 Schema。请重新只输出一个严格的"
                "ComparisonDraftV2 JSON 对象，删除所有 Schema 外字段；不要输出方名、"
                "path_id、verification_status、source_evidence 或解释文字。",
            },
        ]
        repair_policy = s4_policy
        if s4_policy is not None:
            remaining = s4_policy.total_seconds - (perf_counter() - started)
            if remaining <= 0:
                raise
            repair_policy = S4StreamPolicy(
                total_seconds=remaining,
                first_content_seconds=min(s4_policy.first_content_seconds, remaining),
                idle_seconds=min(s4_policy.idle_seconds, remaining),
                max_output_tokens=s4_policy.max_output_tokens,
                connect_timeout_seconds=min(s4_policy.connect_timeout_seconds, remaining),
            )
        draft = await invoke_draft(repair_messages, repair_policy)
    try:
        return _publish_comparison_draft(
            draft,
            evidence,
            retrieval,
            differential,
            consultation_id=consultation_id,
            revision=revision,
        )
    except ComparisonUngroundedError as first_error:
        if repair_used:
            raise
        repair_used = True
        safe_feedback = {
            "validation_rule": first_error.details.get("validation_rule"),
            "field_path": first_error.details.get("field_path"),
        }
        repair_messages = [
            *messages,
            {"role": "assistant", "content": draft.model_dump_json()},
            {
                "role": "user",
                "content": "软件发布校验未通过。只按以下安全字段反馈修复当前草稿，"
                "不得增加候选、事实或 Evidence，不改变医学相对判断："
                + json.dumps(safe_feedback, ensure_ascii=False),
            },
        ]
        repair_policy = s4_policy
        if s4_policy is not None:
            remaining = s4_policy.total_seconds - (perf_counter() - started)
            if remaining <= 0:
                raise first_error
            repair_policy = S4StreamPolicy(
                total_seconds=remaining,
                first_content_seconds=min(s4_policy.first_content_seconds, remaining),
                idle_seconds=min(s4_policy.idle_seconds, remaining),
                max_output_tokens=s4_policy.max_output_tokens,
                connect_timeout_seconds=min(s4_policy.connect_timeout_seconds, remaining),
            )
        repaired = await invoke_draft(repair_messages, repair_policy)
        return _publish_comparison_draft(
            repaired,
            evidence,
            retrieval,
            differential,
            consultation_id=consultation_id,
            revision=revision,
        )
