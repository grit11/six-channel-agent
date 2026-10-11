"""病证关系节点：依赖医生确认时间线和框架证据，不以多候选或病程天数代替关系证据。"""

from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.temporal_evidence import assess_temporal_evidence, usable_current_events
from backend.app.schemas.clinical import ClinicalEvidence, TimelineEvent
from backend.app.schemas.reasoning import ChannelCandidate, SyndromeRelationship


def _usable_events(
    evidence: ClinicalEvidence, candidates: list[ChannelCandidate]
) -> list[TimelineEvent]:
    """只使用关联了明确患者事实的非推测时间线；冲突、未知和候选外事实不用于确认关系。"""
    channels = {channel for candidate in candidates for channel in candidate.six_channels}
    return usable_current_events(evidence, channels)


def judge_syndrome_relationship(
    evidence: ClinicalEvidence, candidates: list[ChannelCandidate], framework_ids: list[str]
) -> list[SyndromeRelationship]:
    """仅在医生确认的两个症状组起病次序和当前共存明确时标记合病/并病，否则保留 uncertain。"""
    eligible = [candidate for candidate in candidates if candidate.status != "excluded"]
    if not eligible:
        return []
    result = SyndromeRelationship(
        relationship_type="uncertain",
        primary_candidate_id=eligible[0].candidate_id,
        secondary_candidate_ids=[candidate.candidate_id for candidate in eligible[1:]],
        temporal_evidence=[event.source_text for event in evidence.timeline],
        supporting_evidence=framework_ids,
        temporal_assessments=assess_temporal_evidence(evidence),
        treatment_evidence=[event.source_text for event in evidence.treatment_events],
        missing_timeline_evidence=[
            "各症状组与六经候选的关联",
            "起病先后或是否同时",
            "前证当前是否未罢",
        ],
    )
    events = _usable_events(evidence, eligible)
    # 无效/互相冲突的输入不能被过滤后假装不存在；医生须明确修订记录再确认关系。
    if any(
        item.status in {"unsupported", "conflicting"}
        for item in result.temporal_assessments
        if item.record_type == "timeline"
    ):
        result.contradicting_evidence = ["病程记录存在未绑定、未知或冲突事实，请核对时间与事实引用"]
        return [result]
    # 同一经出现多条相互不一致的时间线时不选择其中一条，交给医生核对。
    if len({event.six_channel for event in events}) != len(events):
        result.contradicting_evidence = ["同一经有多条时间线记录，需要医生核对"]
        return [result]
    if len(events) < 2 or any(
        event.onset_order is None or event.active_now is not True for event in events
    ):
        return [result]
    if not framework_ids:
        result.missing_timeline_evidence.append("病证关系框架来源证据")
        return [result]
    if len(events) > 3:
        result.missing_timeline_evidence.append("当前合病/并病来源未明确覆盖三经以上组合")
        return [result]
    by_channel = {
        channel: candidate.candidate_id
        for candidate in eligible
        for channel in candidate.six_channels
    }
    # 同经多个候选的关系归属不能靠字典覆盖决定，保留待核对状态。
    if any(
        sum(event.six_channel in candidate.six_channels for candidate in eligible) != 1
        for event in events
    ):
        result.missing_timeline_evidence.append("同一经的多个候选需明确关系归属")
        return [result]
    events.sort(key=lambda event: (event.onset_order or 0, event.six_channel))
    result.primary_candidate_id = by_channel[events[0].six_channel]
    result.secondary_candidate_ids = [by_channel[event.six_channel] for event in events[1:]]
    result.relationship_type = (
        "combined_disease"
        if len({event.onset_order for event in events}) == 1
        else "sequential_concurrent_disease"
    )
    result.temporal_evidence = [event.source_text for event in events]
    result.missing_timeline_evidence = []
    return [result]


async def judge_syndrome_relationships(
    evidence: ClinicalEvidence, candidates: list[ChannelCandidate], knowledge: KnowledgeService
) -> list[SyndromeRelationship]:
    """从唯一知识边界获取关系框架，再生成可审计关系候选。"""
    from backend.app.nodes.relationship_rules import evaluate_relationships

    framework = await knowledge.search_core_framework(
        "六经 证候群 多经 合病 并病 直中 两感 传经 脉证变化 失治 误治 重新辨证", top_k=11
    )
    # 不信任召回对象本身作为引用，精确重读并核验定义；客户端 ID 不能代替内容。
    classics = await knowledge.search_classic("坏病 知犯何逆", top_k=3)
    sources = await knowledge.get_evidence_by_ids(
        [item.evidence_id for item in [*framework, *classics]]
    )
    return evaluate_relationships(evidence, candidates, sources)
