"""S1.2/S1.4 纵向验收：API 工厂、真实图、SQLite 重开、模式隔离和失败契约。

只用模型/向量端口替身；使用实际知识来源与实际 saver，不调用在线服务。
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.core.ports import StructuredResult
from backend.app.graph.six_channel_graph import build_six_channel_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.main import create_app
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.rag.vector_store import VectorBackendStatus
from backend.app.schemas.reasoning import ScoringPolicy
from backend.app.services.reasoning_service import ReasoningService


class FlowModel:
    """基于输入返回可验证结构，故障模式用于证明模型错误不会切回规则。"""

    def __init__(self) -> None:
        """初始化调用记录与来源越界开关。"""
        self.calls: list[str] = []
        self.invalid = False

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """禁止无结构调用。"""
        raise AssertionError("unexpected unstructured model call")

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """从实际请求来源选取 ID，不在夹具中伪造成功来源。"""
        self.calls.append(schema.__name__)
        data = json.loads(messages[-1]["content"])
        if schema.__name__ == "CompactClinicalExtraction":
            evidence = parse_clinical_text(data["clinical_text"])
            return schema.model_validate(
                {
                    "findings": [
                        item.model_dump(
                            include={"key", "polarity", "source_text", "time_scope", "value"}
                        )
                        for item in evidence.findings
                        if item.source_text
                    ]
                }
            )
        return schema.model_validate(
            {
                "candidates": [
                    {
                        "six_channels": ["太阳"],
                        "provisional_syndrome": "太阳候选",
                        "supporting_keys": ["invented"] if self.invalid else ["fever"],
                    }
                ]
            }
        )


@pytest.mark.integration
@pytest.mark.parametrize("parser_mode", ["rules", "llm"])
def test_api_factory_injects_shared_model_and_fails_explicitly(
    tmp_path: Path,
    parser_mode: str,
) -> None:
    """仅 reasoning 或双模式均注入一个模型；失效返回 503，Prompt 模式版本一致。"""
    model = FlowModel()
    settings = Settings(
        _env_file=None,
        clinical_parser_mode=parser_mode,
        reasoning_mode="llm",
        llm_base_url="https://model.test/v1",
        llm_model="fixture",
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "api.db"),
        consultation_storage="memory",
        consultation_run_mode="sync",
    )
    store = MagicMock()
    store.status = VectorBackendStatus("lite", True, "测试端口")
    store.close = AsyncMock()
    with (
        patch("backend.app.core.dependencies.build_milvus_store", return_value=store),
        patch("backend.app.core.dependencies.HTTPStructuredLLM", return_value=model) as factory,
        TestClient(create_app(settings)) as client,
    ):
        response = client.post("/api/v1/reasoning/channel-candidates", json={"text": "发热，脉浮"})
        assert response.status_code == 200
        body = response.json()
        assert body["candidates"][0]["pass_a_score"] == 0.5
        assert body["reasoning_paths"][0]["scoring_version"] == "rp13/v1.0"
        assert "excluded_paths" in body
        known_ids = {item["evidence_id"] for item in body["evidence_catalog"]}
        path_ids = {
            evidence_id
            for path in body["reasoning_paths"]
            for evidence_id in path["framework_evidence"]
        }
        assert set(body["eight_principles"]["framework_evidence_ids"]).issubset(
            known_ids | path_ids
        )
        assert model.calls == (["CompactClinicalExtraction"] if parser_mode == "llm" else []) + [
            "ChannelCandidateList",
        ]
        factory.assert_called_once()
        snapshot = client.post("/api/v1/consultations", json={"text": "发热，脉浮"}).json()
        assert snapshot["versions"]["reasoning_mode"] == "llm"
        versions = client.get("/api/v1/system/versions").json()
        assert versions["prompt_version"] == snapshot["versions"]["prompt_version"]
        model.invalid = True
        failure = client.post("/api/v1/reasoning/channel-candidates", json={"text": "发热"})
        assert failure.status_code == 503
        assert failure.json()["error"]["code"] == "PROVIDER_UNAVAILABLE"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_model_scores_survive_sqlite_restart(tmp_path: Path) -> None:
    """Saver 重开后读取真实模型状态和评分，读取不重调用 Provider。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "reasoning.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    model = FlowModel()
    runtime = await build_checkpoint_runtime(settings)
    try:
        service = ReasoningService(
            knowledge, runtime.checkpointer, reasoning_llm=model, scoring_policy=ScoringPolicy()
        )
        result = await service.analyze_text("发热，脉浮", thread_id="reasoning-restart")
    finally:
        await runtime.close()
    runtime = await build_checkpoint_runtime(settings)
    try:
        graph = build_six_channel_graph(knowledge, runtime.checkpointer, reasoning_llm=model)
        state = await graph.aget_state({"configurable": {"thread_id": result.thread_id}})
        assert state.values["channel_candidates"][0].pass_a_score == 0.5
        assert (
            state.values["reasoning_paths"][0].score_components
            == result.reasoning_paths[0].score_components
        )
        assert len(model.calls) == 1
    finally:
        await runtime.close()
