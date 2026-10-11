"""S4 产品/API/主图回归：分页历史、基本资料、事实修订和安全前序阶段复用。

真实知识与 SQLite 图，业务存储显式 memory；PostgreSQL 实机验证仍后置。
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


def test_history_profile_facts_and_stage_rerun(tmp_path: Path) -> None:
    """核对正式事实修订、同线程阶段重算及只读摘要分页/过滤/校验边界。"""
    settings = Settings(
        _env_file=None,
        retrieval_mode="bm25",
        reranker_enabled=False,
        consultation_storage="memory",
        consultation_run_mode="sync",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    app = create_app(settings)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/consultations",
            json={
                "text": "发热无汗恶寒身痛喘脉浮",
                "patient_ref": "P001",
                "patient_profile": {"age": 42, "sex": "male"},
            },
        ).json()
        assert first["patient_profile"] == {"age": 42, "sex": "male"}
        assert first["last_run_duration_ms"] >= 0
        endpoint = "/api/v1/consultations/" + first["consultation_id"]
        assert (
            client.post(
                endpoint + "/rerun",
                json={"expected_turn": first["turn"], "from_stage": "topk", "top_k": 1},
            ).status_code
            == 400
        )
        # 从 Top-K 开始，八纲/候选/鉴别不可重新执行；后序会真正重新计算。
        with patch(
            "backend.app.graph.six_channel_graph.analyze_eight_principles_node",
            AsyncMock(side_effect=AssertionError("不该重复前序")),
        ):
            rerun = client.post(
                endpoint + "/rerun", json={"expected_turn": first["turn"], "from_stage": "topk"}
            )
        assert rerun.status_code == 200
        current = rerun.json()
        assert current["thread_id"] == first["thread_id"]
        assert current["rerun_from_stage"] == "topk"
        assert (
            current["reasoning"]["stage_snapshots"][:7] == first["reasoning"]["stage_snapshots"][:7]
        )
        assert current["reasoning"]["completed_stages"][-1] == "similar_cases"
        # 后置医案重跑不能把上一轮医案引用重复追加或回流到经典排序。
        case_rerun = client.post(
            endpoint + "/rerun",
            json={"expected_turn": current["turn"], "from_stage": "similar_cases"},
        )
        assert case_rerun.status_code == 200
        next_snapshot = case_rerun.json()
        assert (
            next_snapshot["reasoning"]["reasoning_paths"] == current["reasoning"]["reasoning_paths"]
        )
        assert [item["sequence"] for item in next_snapshot["reasoning"]["stage_snapshots"]] == list(
            range(1, 10)
        )
        current = next_snapshot
        # 医生将当前发热改为未知，历史发热保留；不重传完整陈旧事实列表。
        edited = client.post(
            endpoint + "/evidence",
            json={
                "expected_turn": current["turn"],
                "resume": True,
                "replace_fact_groups": [{"key": "fever", "time_scope": None}],
                "clinical_evidence": {
                    "findings": [
                        {
                            "key": "fever",
                            "polarity": "unknown",
                            "source": "doctor_input",
                            "value": None,
                        },
                        {
                            "key": "fever",
                            "polarity": "positive",
                            "source": "record",
                            "time_scope": "昨日",
                            "source_text": "昨日发热",
                        },
                    ]
                },
            },
        )
        assert edited.status_code == 200
        current = edited.json()
        assert current["rerun_from_stage"] == "build_evidence"
        fever = [
            item for item in current["clinical_evidence"]["findings"] if item["key"] == "fever"
        ]
        assert any(item["polarity"] == "unknown" and item["time_scope"] is None for item in fever)
        assert any(item["time_scope"] == "昨日" for item in fever)
        historical_only = client.post(
            endpoint + "/evidence",
            json={
                "expected_turn": current["turn"],
                "resume": False,
                "replace_fact_groups": [{"key": "fever", "time_scope": "昨日"}],
                "clinical_evidence": {"findings": []},
            },
        ).json()
        assert not any(
            item["time_scope"] == "昨日"
            for item in historical_only["clinical_evidence"]["findings"]
        )
        assert (
            client.post(
                endpoint + "/rerun",
                json={"expected_turn": historical_only["turn"], "from_stage": "topk"},
            ).status_code
            == 400
        )
        assert client.post(endpoint + "/rerun", json={"from_stage": "invented"}).status_code == 422
        second = client.post(
            "/api/v1/consultations", json={"text": "口苦", "patient_ref": "P002"}
        ).json()
        cancelled = client.post(
            "/api/v1/consultations/" + second["consultation_id"] + "/cancel"
        ).json()
        page = client.get("/api/v1/consultations", params={"page_size": 1}).json()
        assert page["total"] == 2 and len(page["items"]) == 1
        assert page["items"][0]["consultation_id"] == cancelled["consultation_id"]
        assert "raw_input" not in page["items"][0] and "clinical_evidence" not in page["items"][0]
        filtered = client.get("/api/v1/consultations", params={"patient_ref": "P001"}).json()
        assert (
            filtered["total"] == 1
            and filtered["items"][0]["consultation_id"] == first["consultation_id"]
        )
        assert (
            client.get("/api/v1/consultations", params={"status": "cancelled"}).json()["total"] == 1
        )
        assert (
            client.get("/api/v1/consultations", params={"page": 3, "page_size": 1}).json()["items"]
            == []
        )
        assert client.get("/api/v1/consultations", params={"page_size": 100}).status_code == 422
        assert client.get("/api/v1/consultations", params={"unexpected": "bad"}).status_code == 422
