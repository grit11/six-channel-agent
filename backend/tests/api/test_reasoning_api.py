"""六经候选 API 回归：验证证据引用、Unknown 和排序分边界。"""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_channel_candidate_api(tmp_path: Path) -> None:
    """API 返回八纲和 Top-K 候选，不返回诊断概率或处方字段。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    with TestClient(create_app(settings)) as client:
        versions = client.get("/api/v1/system/versions").json()
        assert versions["path_summary_version"] == "path-summary/v1.0"
        assert versions["path_completeness_version"] == "path-completeness/v1.0"
        response = client.post(
            "/api/v1/reasoning/channel-candidates",
            json={"text": "发热，无汗，恶寒，脉浮", "top_k": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["candidates"]) == 6  # 宽候选先保留，最终路径才裁剪为输出 K。
    assert body["formula_candidates"]
    assert all(
        item["association_scope"] in {"classic_clause_based", "case_derived_historical"}
        for item in body["formula_candidates"]
    )
    assert body["eight_principles"]["cold_heat"]["score_left"] > 0
    assert all("diagnosis" not in candidate for candidate in body["candidates"])
    assert body["thread_id"]
    assert len(body["reasoning_paths"]) == 3
    assert body["completed_stages"][-1] == "evidence_topk"
    catalog = {item["evidence_id"] for item in body["evidence_catalog"]}
    for path in body["reasoning_paths"]:
        assert path["evidence_assessments"]
        assert all(item["grader_version"] == "grounded-v2" for item in path["evidence_assessments"])
        assert all("condition_assessments" in item for item in path["evidence_assessments"])
        assert path["review_status"] == "pending"
        assert path["summary_version"] == "path-summary/v1.0"
        assert path["treatment_summary"] is not None
        assert path["treatment_principle"] == path["treatment_summary"]["text"]
        detail = path["completeness_detail"]
        assert (
            detail["total_count"]
            == detail["supported_count"] + detail["contradicted_count"] + detail["unknown_count"]
        )
        assert all(
            set(condition["evidence_ids"]).issubset(catalog) for condition in detail["conditions"]
        )
        if path["comparison_with_next"]:
            assert all(
                item["evidence_id"] in catalog
                for item in path["comparison_with_next"]["differences"]
            )
        assert set(path["classic_evidence"] + path["formula_evidence"]).issubset(catalog)
    assert body["completed_stages"].index("formula_branch_retrieval") < body[
        "completed_stages"
    ].index("differential_verification")
    assert body["syndrome_relationships"]


def test_evidence_input_uses_doctor_edits(tmp_path: Path) -> None:
    """结构化入口保留医生原文、来源、三值修订和病程，不被再次解析覆盖。"""
    settings = Settings(
        _env_file=None,
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "milvus.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "checkpoint.db"),
    )
    clinical = {
        "chief_complaint": "医生修订主诉",
        "course": "三日",
        "findings": [{"key": "fever", "polarity": "unknown", "value": None, "source": "record"}],
    }
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/reasoning/evidence", json={"clinical_evidence": clinical, "top_k": 1}
        )
        assert (
            client.post("/api/v1/reasoning/channel-candidates", json={"text": "   "}).status_code
            == 422
        )
    assert response.status_code == 200
    result = response.json()
    assert result["clinical_evidence"]["chief_complaint"] == clinical["chief_complaint"]
    assert result["clinical_evidence"]["course"] == clinical["course"]
    assert result["clinical_evidence"]["findings"][0]["polarity"] == "unknown"
    assert result["clinical_evidence"]["findings"][0]["source"] == "record"
    assert len(result["reasoning_paths"]) == 1
    assert result["reasoning_paths"][0]["formula_pattern"] is None
    # 路径缺口只包含相关鉴别条件；全量未知仍保存在事实中，不能当作阴性支持或反证。
    path = result["reasoning_paths"][0]
    assert path["supporting_findings"] == []
    assert path["contradicting_findings"] == []
    assert path["missing_key_findings"]
    assert result["syndrome_relationships"][0]["relationship_type"] == "uncertain"
