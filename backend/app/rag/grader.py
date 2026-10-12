"""候选上下文的来源适用性分级。

职责：核对来源归属、完整 grounded 条件和当前三值事实，保留正反证与未解决条件。
边界：由 KnowledgeService 提供真实来源，不访问文件、向量库或模型；不生成诊断，
不根据单个冲突触发硬排除，不把召回分、历史医案或元数据标签当患者直接支持。
"""

from pydantic import BaseModel, Field

from backend.app.knowledge.differential import candidate_source_values, source_candidate_label
from backend.app.nodes.source_conditions import assess_source_condition
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.evidence import Evidence, EvidenceAssessment, EvidenceGrade
from backend.app.schemas.knowledge import FormulaBranchUnit


class GradedEvidence(BaseModel):
    """带候选上下文的证据分级结果，可追溯原始 Evidence。"""

    evidence: Evidence = Field(description="原始可追溯证据")
    grade: EvidenceGrade = Field(description="对当前候选的证据用途")
    rationale: str = Field(min_length=1, description="基于来源文本和候选词的简短理由")


def grade_evidence(
    evidence: Evidence,
    *,
    query_terms: list[str],
    contradict_terms: list[str] | None = None,
    direct_terms: list[str] | None = None,
) -> GradedEvidence:
    """保留原词项检索接口；仅用于查询相关性，不作为正式患者适用性判断。"""
    # 元数据中的标签不能替代原文支持；查询词重合只说明相关，不等于直接支持候选。
    text = evidence.content
    contradict_hits = [term for term in (contradict_terms or []) if term and term in text]
    direct_hits = [term for term in (direct_terms or []) if term and term in text]
    related_hits = [term for term in query_terms if term and term in text]
    if contradict_hits:
        grade: EvidenceGrade = "contradict"
        rationale = "命中候选反证词：" + "、".join(contradict_hits)
    elif direct_hits:
        grade = "direct" if evidence.source_level in {"L0", "L1", "L3", "L4"} else "indirect"
        rationale = "命中支持词：" + "、".join(direct_hits)
    elif related_hits:
        grade, rationale = "indirect", "原文命中查询词，仍需候选核验：" + "、".join(related_hits)
    elif evidence.source_level == "L0":
        grade, rationale = "background", "框架来源提供背景约束，未命中当前候选词"
    else:
        grade, rationale = "irrelevant", "当前证据未命中支持或反证词"
    return GradedEvidence(evidence=evidence, grade=grade, rationale=rationale)


def _strings(value: object) -> list[str]:
    """只使用来源中的字符串列表，禁止隐式补齐异常字段。"""
    return (
        list(dict.fromkeys(item for item in value if isinstance(item, str) and item))
        if isinstance(value, list)
        else []
    )


def _source_conditions(evidence: Evidence, formula_name: str | None) -> list[str]:
    """按知识类型读取来源条件；混合阴性/禁忌字段不得整体升级为排除规则。"""
    metadata = evidence.metadata
    if evidence.knowledge_type == "classic_clause":
        terms = _strings(metadata.get("symptom_terms_source_grounded"))
        # grounded 字段仍需与原文逐项对齐，元数据不独立产生医学条件。
        return [term for term in terms if term in evidence.content]
    if evidence.knowledge_type == "formula_pattern":
        terms = _strings(metadata.get("key_symptoms_source_grounded")) + _strings(
            metadata.get("negative_findings_source_grounded")
        )
        return [term for term in terms if term in evidence.content]
    if evidence.knowledge_type == "medical_case":
        return _strings(metadata.get("symptoms_source_grounded"))
    if evidence.knowledge_type == "differential" and formula_name:
        labels = _strings(metadata.get("candidates"))
        dimensions = metadata.get("key_dimensions")
        if isinstance(dimensions, dict):
            values = [
                candidate_source_values(description, labels, dimension=dimension).get(
                    source_candidate_label(formula_name, labels) or "", ""
                )
                for dimension, description in dimensions.items()
                if isinstance(description, str)
            ]
            return list(dict.fromkeys(value for value in values if value))
    return []


def _applicable(evidence: Evidence, channels: list[str], formula_name: str | None) -> bool:
    """来源标签只检查作用范围，不直接提供患者支持。"""
    metadata = evidence.metadata
    if evidence.knowledge_type == "classic_clause":
        return bool(set(channels).intersection(_strings(metadata.get("semantic_six_channels"))))
    if evidence.knowledge_type == "formula_pattern":
        scope = metadata.get("association_scope")
        associated = _strings(
            metadata.get(
                "canonical_six_channels"
                if scope == "classic_clause_based"
                else "case_derived_six_channels"
            )
        )
        return metadata.get("formula_name") == formula_name and set(channels).issubset(associated)
    if evidence.knowledge_type == "differential":
        return (
            formula_name is not None
            and source_candidate_label(formula_name, _strings(metadata.get("candidates")))
            is not None
        )
    if evidence.knowledge_type == "medical_case":
        return bool(set(channels).intersection(_strings(metadata.get("six_channels"))))
    return evidence.knowledge_type == "six_channel_framework"


def grade_candidate_evidence(
    evidence: Evidence,
    clinical: ClinicalEvidence,
    *,
    channels: list[str],
    formula_name: str | None = None,
    branches: list[FormulaBranchUnit] | None = None,
) -> EvidenceAssessment:
    """核对当前路径的来源用途；Unknown、历史或冲突只保留未解决条件。"""
    assessment = EvidenceAssessment(
        evidence_id=evidence.evidence_id,
        grade="background",
        rationale="来源提供背景，尚无完整的当前事实支持",
    )
    if not _applicable(evidence, channels, formula_name):
        assessment.grade = "irrelevant"
        assessment.rationale = "来源六经或方证归属不适用于当前路径"
        return assessment
    if evidence.knowledge_type == "six_channel_framework":
        assessment.rationale = "框架说明用于约束解释，不能仅凭框架标签形成患者直接支持"
        return assessment
    if branches is not None and evidence.knowledge_type == "formula_pattern":
        scored: list[EvidenceAssessment] = []
        for branch in branches:
            if branch.logical_scope != "single_clause":
                continue
            local = evidence.model_copy(deep=True)
            local.metadata["key_symptoms_source_grounded"] = branch.positive_conditions
            local.metadata["negative_findings_source_grounded"] = branch.negative_conditions
            local.content = branch.source_quote
            checked = grade_candidate_evidence(
                local, clinical, channels=channels, formula_name=formula_name
            )
            if branch.review_status != "expert_approved" and checked.grade == "direct":
                checked.grade = "indirect"
            checked.rationale += "；分支 " + branch.branch_id + "，" + branch.review_status
            scored.append(checked)
        if scored:
            return max(
                scored,
                key=lambda item: (
                    len(item.supporting_requirements) - len(item.contradicting_requirements),
                    -len(item.unresolved_requirements),
                    item.rationale,
                ),
            )
        assessment.unresolved_requirements = ["多方条文的局部适用分支尚未核准"]
        return assessment
    conditions = _source_conditions(evidence, formula_name)
    for condition in conditions:
        # 来源维度语义也参与核验，病机/治疗说明不能凭包含症状词变成患者正反证。
        dimensions = evidence.metadata.get("key_dimensions", {})
        labels = _strings(evidence.metadata.get("candidates"))
        dimension = (
            next(
                (
                    key
                    for key, value in dimensions.items()
                    if isinstance(value, str)
                    and condition in candidate_source_values(value, labels, dimension=key).values()
                ),
                "",
            )
            if isinstance(dimensions, dict)
            else ""
        )
        assessment_condition = assess_source_condition(condition, clinical, dimension=dimension)
        assessment.condition_assessments.append(assessment_condition)
        status = assessment_condition.status
        if status == "supports":
            assessment.supporting_requirements.append(condition)
        elif status == "contradicts":
            assessment.contradicting_requirements.append(condition)
        else:
            assessment.unresolved_requirements.append(condition)
    historical = (
        evidence.source_level == "L5"
        or evidence.metadata.get("association_scope") == "case_derived_historical"
    )
    if assessment.contradicting_requirements:
        assessment.grade = "contradict"
        assessment.rationale = "来源条件与当前明确事实存在冲突；条件适用分支仍需核验，不自动硬排除"
    elif assessment.supporting_requirements:
        # 只有全部来源条件均明确支持才标记 direct，部分共同表现仍是 indirect。
        # 历史来源永远不能升级为经典直接支持。
        direct = not historical and not assessment.unresolved_requirements
        if evidence.knowledge_type == "classic_clause":
            associated = _strings(evidence.metadata.get("semantic_six_channels"))
            direct = direct and set(channels).issubset(associated)
            if formula_name and formula_name not in _strings(
                evidence.metadata.get("related_formulas")
            ):
                direct = False
        assessment.grade = "direct" if direct else "indirect"
        assessment.rationale = (
            "来源条件均获当前事实支持；仍由医生核验其临床适用分支"
            if direct
            else "部分来源条件获当前事实支持；其余条件或历史来源限制保留待核验"
        )
    return assessment
