"""Phase 2 三值临床证据和 Query/Grader 单测，不调用模型，不做六经判断。"""

import pytest

from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.rag.grader import grade_evidence
from backend.app.rag.query_builder import (
    build_case_query,
    build_classic_query,
    build_differential_query,
    build_formula_query,
)
from backend.app.schemas.clinical import ClinicalEvidence, Finding
from backend.app.schemas.evidence import Evidence


def test_parser_preserves_unknown_and_negative() -> None:
    """未提及舌脉不生成阴性，明确无汗才是 negative。"""
    evidence = parse_clinical_text("发热，无汗，恶寒，脉浮")
    assert {item.key for item in evidence.positive_findings} == {
        "fever",
        "chills",
        "pulse_floating",
    }
    assert [item.key for item in evidence.negative_findings] == ["sweating"]
    assert {item.key for item in evidence.uncertain_findings} >= {"tongue", "pulse"}
    assert all(item.key != "tongue" or item.polarity == "unknown" for item in evidence.findings)


def test_parser_records_conflict_without_overwrite() -> None:
    """同一输入同时出现汗出与无汗时保留两侧冲突。"""
    evidence = parse_clinical_text("汗出，但又无汗，脉浮")
    assert len(evidence.contradictions) == 1
    assert evidence.contradictions[0].key == "sweating"
    assert len(evidence.contradictions[0].findings) == 2


def test_query_builders_skip_unknown_and_keep_negative() -> None:
    """查询只使用已知事实，方证查询保留关键阴性词。"""
    evidence = ClinicalEvidence(
        chief_complaint="发热",
        findings=[
            Finding(key="fever", value=True, polarity="positive", source="doctor_input"),
            Finding(key="sweating", value=False, polarity="negative", source="doctor_input"),
            Finding(key="pulse", value=None, polarity="unknown", source="doctor_input"),
        ],
    )
    assert "pulse" not in build_classic_query(evidence, ["太阳"])
    assert "sweating:negative:False" in build_formula_query(evidence, ["麻黄汤证"])
    assert build_differential_query(["桂枝汤证", "麻黄汤证"], ["汗"])
    assert "发热" in build_case_query(evidence, ["太阳"])


def test_evidence_grader_keeps_source_and_grade() -> None:
    """分级只改变用途标签，不改 Evidence ID 或内容。"""
    evidence = Evidence(
        evidence_id="1.2:clause_1",
        knowledge_id="clause_1",
        knowledge_type="classic_clause",
        knowledge_version="1.2",
        source="《伤寒论》",
        source_level="L1",
        content="太阳之为病，脉浮，头项强痛而恶寒。",
        retrieval_score=0.8,
        retrieval_method="bm25",
    )
    graded = grade_evidence(evidence, query_terms=["脉浮", "恶寒"], direct_terms=["脉浮"])
    assert graded.grade == "direct"
    assert graded.evidence.evidence_id == evidence.evidence_id
    assert graded.evidence.content == evidence.content


def test_empty_parser_input_rejected() -> None:
    """空输入不能伪造一个没有来源的 ClinicalEvidence。"""
    with pytest.raises(ValueError):
        parse_clinical_text(" ")
