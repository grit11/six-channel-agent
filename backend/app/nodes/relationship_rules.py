"""九类关系核验：消费医生病史、可信时间线与 KnowledgeService 返回的真实来源。

只生成待审核关系和公开条件审计；不按天数传经，不从治疗相邻推断误治，不生成处方。
历史经保存在 previous_six_channels，绝不伪造历史经的当前候选 ID。
"""

from backend.app.nodes.syndrome_relation import judge_syndrome_relationship
from backend.app.nodes.temporal_evidence import assess_temporal_evidence, usable_current_events
from backend.app.schemas.clinical import ClinicalEvidence, TimelineEvent, fact_time_group
from backend.app.schemas.evidence import Evidence, EvidenceAssessment
from backend.app.schemas.reasoning import (
    ChannelCandidate,
    RelationshipAssessment,
    RelationshipType,
    SyndromeRelationship,
)


def _sources(kind: RelationshipType, sources: list[Evidence]) -> list[Evidence]:
    """定义词和来源类别都明确才接受；召回排名、标题或候选标签不是定义证据。"""
    words = {
        "single": ("六经", "证候群"),
        "concurrent": ("多经候选", "并行状态"),
        "combined_disease": ("同时发病", "合病"),
        "sequential_concurrent_disease": ("一经未罢", "并病"),
        "direct_invasion": ("不经三阳", "直接入三阴"),
        "twofold_attack": ("表里两经", "同时感邪"),
        "transmission": ("脉证变化", "不按固定病程日数"),
        "changed_syndrome": ("证候变化", "重新辨证"),
        "spoiled_disease": ("失治", "误治", "重新辨证"),
    }.get(kind, ())
    matched = [
        item
        for item in sources
        if item.knowledge_type == "six_channel_framework"
        and words
        and all(word in item.content for word in words)
    ]
    if kind == "spoiled_disease":
        matched.extend(
            item
            for item in sources
            if item.knowledge_type == "classic_clause"
            and "坏病" in item.content
            and "知犯何逆" in item.content
        )
    return matched


def evaluate_relationships(
    clinical: ClinicalEvidence, candidates: list[ChannelCandidate], sources: list[Evidence]
) -> list[SyndromeRelationship]:
    """逐类核验再生成关系候选；未支持的类别保留缺失和反向条件，不凑结论。"""
    eligible = [item for item in candidates if item.status != "excluded"]
    if not eligible:
        return []
    temporal = assess_temporal_evidence(clinical)
    current = usable_current_events(
        clinical, {channel for item in eligible for channel in item.six_channels}
    )
    history = clinical.relationship_history
    history_ok = history.source != "inference" and bool(history.source_text.strip())
    base = judge_syndrome_relationship(
        clinical, eligible, [item.evidence_id for item in _sources("combined_disease", sources)]
    )[0]
    audit_by_id = {item.record_id: item for item in temporal if item.record_type == "timeline"}
    previous = [
        event
        for index, event in enumerate(clinical.timeline)
        if fact_time_group(event.time_scope) != "current"
        and audit_by_id[f"timeline:{index}"].status == "usable"
    ]
    valid = not any(
        item.status in {"conflicting", "unsupported"}
        for item in temporal
        if item.record_type == "timeline"
    )
    documented_current = [
        event for event in clinical.timeline if fact_time_group(event.time_scope) == "current"
    ]
    attribution = all(
        sum(event.six_channel in item.six_channels for item in eligible) == 1
        for event in documented_current
    )
    all_current_clear = all(
        event.active_now is not None and event.onset_order is not None
        for event in documented_current
    )
    present = [
        event for event in current if event.active_now is True and event.onset_order is not None
    ]
    present_channels = {event.six_channel for event in present}
    distinct = len(present_channels) == len(present)
    primary = history.primary_six_channel
    primary_events = [event for event in present if primary == event.six_channel]
    historical_distinct = len(
        {(event.six_channel, fact_time_group(event.time_scope)) for event in previous}
    ) == len(previous)
    changes = [
        (old, now)
        for old in previous
        for now in primary_events
        if old.onset_order is not None
        and old.onset_order < (now.onset_order or 0)
        and fact_time_group(old.time_scope) != fact_time_group(now.time_scope)
    ]
    changed_order = bool(changes)
    changed_facts = any(
        set(old.finding_keys) != set(now.finding_keys) or old.six_channel != now.six_channel
        for old, now in changes
    )
    transmitted = any(
        old.six_channel != now.six_channel and old.active_now is False for old, now in changes
    )
    treatment_audits = [item for item in temporal if item.record_type == "treatment"]
    treatments_ok = bool(history.treatment_event_ids) and all(
        sum(item.record_id == ref and item.status == "usable" for item in treatment_audits) == 1
        for ref in history.treatment_event_ids
    )
    # 治疗事件必须绑定本次关系的历史/当前事实，不能借其他病史事件证明原证失效。
    treatments_linked = treatments_ok and all(
        any(
            fact_time_group(treatment.before_time_scope) == fact_time_group(old.time_scope)
            and fact_time_group(treatment.after_time_scope) == fact_time_group(now.time_scope)
            and bool(set(treatment.before_finding_keys).intersection(old.finding_keys))
            and bool(set(treatment.after_finding_keys).intersection(now.finding_keys))
            for old, now in changes
        )
        for treatment in clinical.treatment_events
        if treatment.event_id in history.treatment_event_ids
    )
    # bool|None 条件明确区分未知和反向，缺失不能写入反证。
    common: list[tuple[str, bool | None]] = [
        ("可信时间线事实", valid),
        ("当前组明确且归属唯一", bool(present) and attribution and distinct),
        ("当前组次序与共存三态完整", True if all_current_clear else None),
        ("历史时点无重复组", historical_distinct),
    ]
    context: list[tuple[str, bool | None]] = [("医生病史核对原文", history_ok)]
    requirements: dict[RelationshipType, list[tuple[str, bool | None]]] = {
        "single": common
        + context
        + [
            ("排查其他当前证候组", history.no_other_current_syndrome),
            ("单组当前证候", len(present) == 1),
        ],
        "concurrent": common
        + context
        + [
            ("本证主经明确", primary in present_channels if primary else None),
            ("两或三经当前共存", 2 <= len(present) <= 3),
        ],
        "combined_disease": [
            (
                "同时起病且当前共存",
                (
                    base.relationship_type == "combined_disease"
                    if base.relationship_type != "uncertain"
                    else None
                ),
            )
        ],
        "sequential_concurrent_disease": [
            (
                "先后起病且前证未罢",
                (
                    base.relationship_type == "sequential_concurrent_disease"
                    if base.relationship_type != "uncertain"
                    else None
                ),
            )
        ],
        "direct_invasion": common
        + context
        + [
            ("病史完整", history.history_complete),
            (
                "明确未先经三阳",
                None if history.prior_yang_phase is None else not history.prior_yang_phase,
            ),
            (
                "所有当前组为三阴",
                bool(present_channels) and present_channels.issubset({"太阴", "少阴", "厥阴"}),
            ),
            (
                "时间线没有既往三阳组",
                not any(
                    event.six_channel in {"太阳", "阳明", "少阳"} for event in clinical.timeline
                ),
            ),
        ],
        "twofold_attack": common
        + context
        + [
            (
                "两组同时起病",
                len(present) == 2 and len({event.onset_order for event in present}) == 1,
            ),
            ("表里两经已核对", history.exterior_interior_pair),
            ("同时感邪已核对", history.simultaneous_exposure),
        ],
        "transmission": common
        + context
        + [
            ("病史完整", history.history_complete),
            ("当前主经明确", bool(primary_events)),
            ("脉证变化已核对", history.syndrome_changed),
            ("可信历史与当前先后次序", changed_order),
            ("历史当前脉证不同", changed_facts),
            ("前后涉及不同经", transmitted),
        ],
        "changed_syndrome": common
        + context
        + [
            ("当前主经明确", bool(primary_events)),
            ("脉证变化已核对", history.syndrome_changed),
            ("可信历史与当前先后次序", changed_order),
            ("历史当前脉证不同", changed_facts),
        ],
        "spoiled_disease": common
        + context
        + [
            ("当前主经明确", bool(primary_events)),
            ("脉证变化已核对", history.syndrome_changed),
            ("可信历史与当前先后次序", changed_order),
            (
                "原方证已不适用",
                (
                    None
                    if history.original_pattern_applicable is None
                    else not history.original_pattern_applicable
                ),
            ),
            ("失治误治关联由医生核对", history.treatment_error_confirmed),
            ("关联治疗前后事实完整", treatments_linked),
        ],
    }
    audits: list[RelationshipAssessment] = []
    for kind, conditions in requirements.items():
        refs = _sources(kind, sources)
        missing = [name for name, value in conditions if value is None]
        contrary = [name for name, value in conditions if value is False]
        derived_gaps = {
            "医生病史核对原文",
            "当前主经明确",
            "本证主经明确",
            "可信历史与当前先后次序",
            "关联治疗前后事实完整",
        }
        missing.extend(name for name in contrary if name in derived_gaps)
        contrary = [name for name in contrary if name not in derived_gaps]
        if not previous:
            missing.extend(
                name for name in contrary if name in {"历史当前脉证不同", "前后涉及不同经"}
            )
            contrary = [
                name for name in contrary if name not in {"历史当前脉证不同", "前后涉及不同经"}
            ]
        if kind == "spoiled_disease" and not any(
            item.knowledge_type == "classic_clause" for item in refs
        ):
            missing.append("坏病经典定义来源")
        # 未提供时间线/病史时，结构不足属于缺失；不把没有记录当成否定。
        if not clinical.timeline:
            missing.extend(contrary)
            contrary = []
        if not refs:
            missing.append("完整关系定义来源")
        audits.append(
            RelationshipAssessment(
                relationship_type=kind,
                status="contradicted" if contrary else "insufficient" if missing else "supported",
                evidence_ids=[item.evidence_id for item in refs],
                missing_conditions=missing,
                contradicting_conditions=contrary,
                rationale="按患者时间事实、医生核对与精确来源定义逐项校验，关系候选待审核",
            )
        )
    results: list[SyndromeRelationship] = []
    for audit in audits:
        if audit.status != "supported":
            continue
        kind = audit.relationship_type
        events: list[TimelineEvent] = (
            primary_events
            if kind in {"transmission", "changed_syndrome", "spoiled_disease"}
            else present
        )
        ordered = sorted(
            events,
            key=lambda event: (
                event.six_channel != primary,
                event.onset_order or 0,
                event.six_channel,
            ),
        )
        ids = [
            next(item.candidate_id for item in eligible if event.six_channel in item.six_channels)
            for event in ordered
        ]
        result = base.model_copy(
            deep=True,
            update={
                "relationship_type": kind,
                "primary_candidate_id": ids[0],
                "secondary_candidate_ids": list(dict.fromkeys(ids[1:])),
                "missing_timeline_evidence": [],
                "contradicting_evidence": [],
                "supporting_evidence": audit.evidence_ids,
                "temporal_evidence": [event.source_text for event in [*previous, *ordered]],
                "previous_six_channels": sorted({event.six_channel for event in previous}),
                "current_six_channels": [event.six_channel for event in ordered],
            },
        )
        results.append(result)
    if not results:
        results = [
            base.model_copy(
                update={
                    "relationship_type": "uncertain",
                    "missing_timeline_evidence": list(
                        dict.fromkeys(
                            [
                                *base.missing_timeline_evidence,
                                *(name for audit in audits for name in audit.missing_conditions),
                            ]
                        )
                    ),
                }
            )
        ]
    for result in results:
        result.assessments = audits
        result.source_assessments = [
            EvidenceAssessment(
                evidence_id=item.evidence_id,
                grade="background",
                rationale="关系定义来源，须结合医生病史事实，不直接证明患者诊断",
            )
            for item in sources
            if item.evidence_id in result.supporting_evidence
        ]
    return results
