"""结构化解析纵向回归：模型端口注入真实 API/Graph，验证来源校验及医生修订边界。"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.core.dependencies import open_llm
from backend.app.core.ports import StructuredResult
from backend.app.knowledge.service import KnowledgeService
from backend.app.main import create_app
from backend.app.schemas.clinical import ClinicalEvidence, Finding
from backend.app.services.reasoning_service import ReasoningService


class ParserLLM:
    """测试模型只抽取固定来源事实，供真实服务编排验证。"""

    def __init__(self) -> None:
        """调用计数用于证明医生快照不会被再次解析覆盖。"""
        self.calls = 0

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """不允许解析节点退回无 Schema 普通文本。"""
        raise AssertionError("unexpected text call")

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """输出待校验的负向汗证，不新增诊断或来源。"""
        self.calls += 1
        return schema.model_validate(
            {
                "findings": [
                    {
                        "key": "sweating",
                        "polarity": "negative",
                        "value": False,
                        "source": "doctor_input",
                        "source_text": "无汗",
                    }
                ]
            }
        )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_uses_injected_parser_but_preserves_doctor_snapshot() -> None:
    """原文入口调用模型；结构化医生输入无需重新调用模型；图仍完成八阶段。"""
    settings = Settings(_env_file=None)
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    model = ParserLLM()
    service = ReasoningService(knowledge, llm=model)
    first = await service.analyze_text("无汗")
    assert model.calls == 1
    assert first.clinical_evidence.negative_findings[0].key == "sweating"
    edited = ClinicalEvidence(
        findings=[
            Finding(
                key="sweating",
                polarity="unknown",
                value=None,
                source="doctor_input",
            )
        ]
    )
    second = await service.analyze_evidence(edited)
    assert model.calls == 1
    assert second.clinical_evidence == edited
    assert second.completed_stages[-1] == "similar_cases"


def test_parse_api_uses_model_and_reports_implemented_prompt(tmp_path: Path) -> None:
    """API 不直连厂商，工厂客户端生命周期保留，服务只注入模型端口。"""
    model = ParserLLM()
    settings = Settings(
        _env_file=None,
        clinical_parser_mode="llm",
        llm_base_url="https://model.test/v1",
        llm_model="test",
        milvus_mode="lite",
        milvus_lite_path=str(tmp_path / "vector.db"),
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    with (
        patch("backend.app.core.dependencies.HTTPStructuredLLM", return_value=model),
        TestClient(create_app(settings)) as client,
    ):
        result = client.post("/api/v1/clinical/parse", json={"text": "无汗"})
        assert result.status_code == 200
        assert result.json()["parser_version"] == "llm-clinical/v1.0"
        assert (
            client.get("/api/v1/system/versions").json()["prompt_version"] == "clinical-parser/v1.0"
        )
        invalid = client.post("/api/v1/clinical/parse", json={"text": "发热"})
        assert invalid.status_code == 503
        assert invalid.json()["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert model.calls == 2


@pytest.mark.asyncio
async def test_llm_factory_closes_client_on_failure() -> None:
    """生命周期异常仍释放模型客户端，后续请求不能使用已关闭资源。"""
    settings = Settings(_env_file=None, llm_base_url="https://model.test/v1", llm_model="test")
    with (
        pytest.raises(RuntimeError),
        patch("backend.app.core.dependencies.HTTPStructuredLLM") as factory,
    ):
        async with open_llm(settings):
            client = factory.call_args.args[1]
            assert not client.is_closed
            raise RuntimeError("test lifecycle")
    assert client.is_closed
