"""病程事实核验：校验医生时间线及治疗前后引用，依赖 ClinicalEvidence 三值事实。

历史事实只支持历史时点，不能成为当前正证；治疗相邻不推断疗效、误治或传经因果。
本模块不检索知识、不给治疗建议，返回公开缺失/冲突摘要给关系节点。
"""

from collections import Counter
from typing import Literal

from backend.app.schemas.clinical import ClinicalEvidence, Finding, TimelineEvent, fact_time_group
from backend.app.schemas.reasoning import TemporalEvidenceAssessment


def _matching(
    evidence: ClinicalEvidence, keys: list[str], time_scope: str | None
) -> tuple[list[str], list[str], bool]:
    """按明确时间找可信事实，零确定性/推测/同时间正反冲突不能作为证据。"""
    supported, missing = [], []
    conflicting = False
    for key in keys:
        facts: list[Finding] = [
            finding
            for finding in evidence.findings
            if finding.key == key
            and fact_time_group(finding.time_scope) == fact_time_group(time_scope)
            and finding.source != "inference"
            and finding.certainty > 0
            and finding.polarity != "unknown"
        ]
        if len({finding.polarity for finding in facts}) > 1:
            conflicting = True
            missing.append(key)
        elif facts:
            supported.append(key)
        else:
            missing.append(key)
    return supported, missing, conflicting


def assess_temporal_evidence(evidence: ClinicalEvidence) -> list[TemporalEvidenceAssessment]:
    """每条输入都保留核验记录；前后内容未绑定时明确不完整，不补事实。"""
    results: list[TemporalEvidenceAssessment] = []
    status: Literal["usable", "incomplete", "conflicting", "unsupported"]
    for index, event in enumerate(evidence.timeline):
        support, missing, conflict = _matching(evidence, event.finding_keys, event.time_scope)
        positive = {
            finding.key
            for finding in evidence.findings
            if finding.polarity == "positive"
            and fact_time_group(finding.time_scope) == fact_time_group(event.time_scope)
            and finding.source != "inference"
            and finding.certainty > 0
        }
        missing = sorted(set(missing) | (set(event.finding_keys) - positive))
        support = [key for key in support if key not in missing]
        if event.source == "inference":
            support = []
        status = (
            "conflicting"
            if conflict
            else (
                "unsupported"
                if event.source == "inference" or missing
                else (
                    "incomplete"
                    if event.onset_order is None or event.active_now is None
                    else "usable"
                )
            )
        )
        results.append(
            TemporalEvidenceAssessment(
                record_type="timeline",
                record_id=f"timeline:{index}",
                source_text=event.source_text,
                status=status,
                supporting_findings=support,
                missing_findings=missing,
                reason="按医生指定时间和阳性事实核验；缺失不转阴性，仍需关系来源判断。",
            )
        )
    identities = Counter(event.event_id for event in evidence.treatment_events)
    for treatment in evidence.treatment_events:
        before, missing_before, conflict_before = _matching(
            evidence, treatment.before_finding_keys, treatment.before_time_scope
        )
        after, missing_after, conflict_after = _matching(
            evidence, treatment.after_finding_keys, treatment.after_time_scope
        )
        complete = bool(
            treatment.before_finding_keys
            and treatment.after_finding_keys
            and treatment.before_time_scope
            and fact_time_group(treatment.before_time_scope)
            != fact_time_group(treatment.after_time_scope)
            and treatment.event_order is not None
        )
        status = (
            "conflicting"
            if conflict_before or conflict_after or identities[treatment.event_id] > 1
            else (
                "unsupported"
                if treatment.source == "inference"
                else "incomplete" if not complete or missing_before or missing_after else "usable"
            )
        )
        results.append(
            TemporalEvidenceAssessment(
                record_type="treatment",
                record_id=treatment.event_id,
                source_text=treatment.source_text,
                status=status,
                supporting_findings=(
                    [] if treatment.source == "inference" else sorted(set(before + after))
                ),
                missing_findings=sorted(set(missing_before + missing_after)),
                reason=(
                    "事件 ID 重复，需医生修订"
                    if identities[treatment.event_id] > 1
                    else "只确认治疗前后事实引用与顺序完整；不认定治疗因果、疗效或误治。"
                ),
            )
        )
    return results


def usable_current_events(evidence: ClinicalEvidence, channels: set[str]) -> list[TimelineEvent]:
    """当前共存只能绑定当前可信阳性事实；历史时间线另存审计，不用于确认共存。"""
    known = {
        finding.key for finding in evidence.current_findings() if finding.polarity == "positive"
    }
    return [
        event
        for event in evidence.timeline
        if event.source != "inference"
        and event.six_channel in channels
        and fact_time_group(event.time_scope) == "current"
        and set(event.finding_keys).issubset(known)
    ]
