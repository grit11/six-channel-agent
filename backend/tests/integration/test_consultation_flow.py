"""会话纵向回归：真实知识/图/SQLite checkpoint 与明确的内存业务 Repository。

验证同线程补充、版本/状态约束及医生审核，不宣称 PostgreSQL 或业务 interrupt 已完成。
"""

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_consultation_merge_review_and_state_conflicts(tmp_path: Path) -> None:
    """输入→保存快照→同线程补充→审核；原始 Agent 路径不能被人工修改覆盖。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        retrieval_mode="bm25",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        created = client.post("/api/v1/consultations", json={"text": "发热，恶寒，脉浮"})
        assert created.status_code == 201
        initial = created.json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        assert client.get(path).json() == initial
        assert client.get("/api/v1/consultations/" + str(uuid4())).status_code == 404
        added = client.post(
            path + "/evidence",
            json={
                "clinical_text": "无汗，脉浮紧",
                "expected_turn": initial["turn"],
            },
        )
        assert added.status_code == 200
        updated = added.json()
        assert updated["thread_id"] == initial["thread_id"]
        assert updated["turn"] > initial["turn"]
        known = {
            item["key"]
            for item in updated["clinical_evidence"]["findings"]
            if item["polarity"] != "unknown"
        }
        assert {"fever", "chills", "pulse_floating", "pulse_tight", "sweating"}.issubset(known)
        assert (
            client.post(path + "/rerun", json={"expected_turn": initial["turn"]}).status_code == 409
        )
        assert client.post(path + "/review", json={"action": "accept"}).status_code == 400
        agent_paths = updated["reasoning"]["reasoning_paths"]
        ids = [item["path_id"] for item in agent_paths]
        assert ids
        note = client.post(
            path + "/review",
            json={"action": "note", "note": "医生核对", "expected_turn": updated["turn"]},
        )
        assert note.status_code == 200
        assert note.json()["status"] != "completed"
        ordered = client.post(
            path + "/review", json={"action": "reorder", "ordered_path_ids": list(reversed(ids))}
        )
        assert ordered.status_code == 200
        assert ordered.json()["status"] != "completed"
        reviewed = client.post(
            path + "/review",
            json={
                "action": "modify",
                "selected_path_id": ids[0],
                "modified_six_channels": ["太阳"],
                "modified_syndrome": "医生修订证候",
                "expected_turn": ordered.json()["turn"],
            },
        )
        assert reviewed.status_code == 200
        final = reviewed.json()
        assert final["status"] == "completed"
        assert final["doctor_review"]["doctor_adopted"] is True
        assert len(final["review_history"]) == 3
        assert final["reasoning"]["reasoning_paths"] == agent_paths
        assert client.post(path + "/evidence", json={"clinical_text": "口苦"}).status_code == 409
        assert client.post(path + "/rerun", json={}).status_code == 409


def test_memory_repository_is_explicit_and_disabled_is_unavailable(tmp_path: Path) -> None:
    """默认业务能力关闭；checkpoint 正常也不得冒充 consultation 数据库。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="disabled",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        result = client.post("/api/v1/consultations", json={"text": "无汗"})
        assert result.status_code == 503
        assert result.json()["error"]["code"] == "PERSISTENCE_ERROR"


def test_supplement_preserves_conflict_unless_doctor_explicitly_replaces(tmp_path: Path) -> None:
    """默认新增正反记录产生冲突，明确 replace_keys 才修订当前事实。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        initial = client.post("/api/v1/consultations", json={"text": "汗出"}).json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        conflict = client.post(path + "/evidence", json={"clinical_text": "无汗"}).json()
        assert conflict["clinical_evidence"]["contradictions"][0]["key"] == "sweating"
        fixed = client.post(
            path + "/evidence", json={"clinical_text": "无汗", "replace_keys": ["sweating"]}
        )
        assert fixed.status_code == 200
        assert fixed.json()["clinical_evidence"]["contradictions"] == []
        assert client.post(path + "/cancel").json()["status"] == "cancelled"
        assert client.post(path + "/cancel").status_code == 409


def test_saved_supplement_review_stop_and_rejected_path(tmp_path: Path) -> None:
    """仅保存清除旧结果，重算后按当前信息审核，驳回路径不能直接采用。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
        max_supplement_rounds=1,
    )
    with TestClient(create_app(settings)) as client:
        initial = client.post(
            "/api/v1/consultations", json={"clinical_text": "发热，无汗，恶寒，身痛，喘，脉浮"}
        ).json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        ready = client.post(path + "/review-ready", json={"expected_turn": initial["turn"]})
        assert ready.status_code == 200
        assert ready.json()["reasoning"]["evidence_gaps"] == initial["reasoning"]["evidence_gaps"]
        saved = client.post(
            path + "/evidence", json={"clinical_text": "脉浮紧", "resume": False}
        ).json()
        assert saved["reasoning"] is None
        rerun = client.post(path + "/rerun", json={"expected_turn": saved["turn"]}).json()
        assert rerun["status"] == "awaiting_doctor_review"
        assert rerun["thread_id"] == initial["thread_id"]
        ids = [item["path_id"] for item in rerun["reasoning"]["reasoning_paths"]]
        rejected = client.post(
            path + "/review", json={"action": "reject", "selected_path_id": ids[0]}
        )
        assert rejected.status_code == 200
        assert (
            client.post(
                path + "/review", json={"action": "accept", "selected_path_id": ids[0]}
            ).status_code
            == 400
        )
        assert (
            client.post(
                path + "/review",
                json={"action": "reorder", "ordered_path_ids": [ids[0]] * len(ids)},
            ).status_code
            == 400
        )
        accepted = client.post(
            path + "/review", json={"action": "accept", "selected_path_id": ids[1]}
        ).json()
        assert accepted["status"] == "completed"
        assert len(accepted["review_history"]) == 2
        stream = client.get(path + "/stream")
        assert stream.status_code == 200
        assert "event: completed" in stream.text


def test_gap_availability_validation_and_noop_preserve_turn(tmp_path: Path) -> None:
    """无法获取专用提交走真实 API/图；无效 ID 和重复提交不消耗轮次。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with TestClient(create_app(settings)) as client:
        initial = client.post(
            "/api/v1/consultations", json={"clinical_text": "发热，无汗，恶寒，身痛，喘，脉浮"}
        ).json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        assert (
            client.post(
                path + "/evidence", json={"unavailable_question_ids": ["invented"]}
            ).status_code
            == 400
        )
        assert client.post(path + "/evidence", json={"clinical_text": "无汗"}).status_code == 400
        unchanged = client.get(path).json()
        assert unchanged["turn"] == initial["turn"] and unchanged["supplementation_round"] == 0
        question_ids = [
            gap["question_id"]
            for gap in initial["reasoning"]["evidence_gaps"]
            if gap["status"] == "pending"
        ]
        unavailable = client.post(
            path + "/evidence", json={"unavailable_question_ids": question_ids}
        ).json()
        assert unavailable["status"] == "awaiting_doctor_review"
        assert unavailable["reasoning"]["evidence_gap_decision"]["reason"] == "all_unavailable"
        assert unavailable["clinical_evidence"] == initial["clinical_evidence"]
        assert (
            client.post(
                path + "/evidence", json={"unavailable_question_ids": question_ids}
            ).status_code
            == 400
        )
        restored = client.post(
            path + "/evidence", json={"available_question_ids": question_ids}
        ).json()
        assert restored["thread_id"] == initial["thread_id"]
        assert restored["status"] == "needs_evidence"
        assert restored["evidence_gap_context"]["unavailable_question_ids"] == []
        assert restored["evidence_gap_context"]["asked_counts"][question_ids[0]] == 2


def test_context_only_revision_recomputes_same_thread_and_can_clear(tmp_path: Path) -> None:
    """医生修订起病次序让合病转并病；清空和无操作校验不补造患者事实。"""
    settings = Settings(
        _env_file=None,
        consultation_storage="memory",
        consultation_run_mode="sync",
        retrieval_mode="bm25",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    timeline = [
        {
            "six_channel": "太阳",
            "finding_keys": ["chills"],
            "onset_order": 0,
            "active_now": True,
            "source_text": "医生确认恶寒当前存在且同时起病",
        },
        {
            "six_channel": "阳明",
            "finding_keys": ["thirst"],
            "onset_order": 0,
            "active_now": True,
            "source_text": "医生确认口渴当前存在且同时起病",
        },
    ]
    with TestClient(create_app(settings)) as client:
        initial = client.post(
            "/api/v1/consultations",
            json={
                "text": "恶寒，口渴",
                "clinical_evidence": {
                    "findings": [
                        {"key": key, "polarity": "positive", "source": "doctor_input"}
                        for key in ("chills", "thirst")
                    ],
                    "timeline": timeline,
                    "onset": "昨日",
                    "prior_treatment": [{"description": "旧记录"}],
                },
            },
        ).json()
        path = "/api/v1/consultations/" + initial["consultation_id"]
        assert (
            initial["reasoning"]["syndrome_relationships"][0]["relationship_type"]
            == "combined_disease"
        )
        assert (
            client.post(
                path + "/evidence",
                json={
                    "clinical_evidence": {},
                    "replace_context_fields": ["timeline"],
                },
            ).status_code
            == 400
        )
        timeline[1]["onset_order"] = 1
        revised = client.post(
            path + "/evidence",
            json={
                "expected_turn": initial["turn"],
                "clinical_evidence": {
                    "timeline": timeline,
                    "course": "先恶寒后口渴，当前共存",
                    "treatment_events": [
                        {
                            "event_id": "t1",
                            "description": "核对已用治疗",
                            "source_text": "前后事实尚未取得",
                        }
                    ],
                },
                "replace_context_fields": ["timeline", "course", "treatment_events"],
            },
        )
        assert revised.status_code == 200
        result = revised.json()
        assert result["thread_id"] == initial["thread_id"]
        assert result["clinical_evidence"]["findings"] == initial["clinical_evidence"]["findings"]
        assert result["clinical_evidence"]["prior_treatment"] == [{"description": "旧记录"}]
        relationship = result["reasoning"]["syndrome_relationships"][0]
        assert relationship["relationship_type"] == "sequential_concurrent_disease"
        assert relationship["temporal_assessments"][-1]["status"] == "incomplete"
        assert (
            client.post(
                path + "/evidence",
                json={
                    "expected_turn": initial["turn"],
                    "clinical_evidence": {"timeline": []},
                    "replace_context_fields": ["timeline"],
                },
            ).status_code
            == 409
        )
        cleared = client.post(
            path + "/evidence",
            json={
                "clinical_evidence": {
                    "timeline": [],
                    "treatment_events": [],
                    "onset": None,
                    "course": None,
                },
                "replace_context_fields": ["timeline", "treatment_events", "onset", "course"],
            },
        ).json()
        assert cleared["reasoning"]["syndrome_relationships"][0]["relationship_type"] == "uncertain"
        assert cleared["clinical_evidence"]["timeline"] == []
        assert cleared["clinical_evidence"]["onset"] is None
        assert client.get(path).json() == cleared
        assert (
            client.post(
                path + "/evidence",
                json={
                    "clinical_evidence": {"timeline": []},
                    "replace_context_fields": ["timeline"],
                },
            ).status_code
            == 400
        )
        assert client.get(path).json()["turn"] == cleared["turn"]
