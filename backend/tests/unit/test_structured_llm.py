"""结构化 LLM 协议与解析约束测试；用 HTTP 替身验证实际适配，不调用付费模型。"""

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from backend.app.core.config import Settings
from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.core.llm import HTTPStructuredLLM
from backend.app.core.ports import StructuredResult
from backend.app.nodes.clinical_parser import parse_clinical_with_llm
from backend.app.schemas.clinical import (
    ClinicalEvidence,
    Finding,
    RelationshipHistory,
    TreatmentEvent,
)


class Payload(BaseModel):
    """测试协议契约，不包含医学判断。"""

    model_config = ConfigDict(extra="forbid")
    value: int


def settings(retries: int = 0) -> Settings:
    """不读取项目密钥的模型测试配置。"""
    return Settings(
        _env_file=None,
        llm_base_url="https://model.test/v1",
        llm_model="test",
        llm_max_retries=retries,
    )


def response(content: str, finish: str = "stop") -> httpx.Response:
    """构造兼容服务响应；隐藏字段只用于验证它不会出现在返回结果。"""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish,
                    "message": {
                        "content": content,
                        "reasoning_content": "private hidden field",
                    },
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_structured_adapter_retries_schema_and_ignores_hidden_fields() -> None:
    """第一次 Schema 错误允许有限重试，正常响应只返回指定对象。"""
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """检查路径、模型和 JSON 输出约束，正文不依赖真实网络。"""
        nonlocal attempts
        attempts += 1
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "test"
        assert payload["response_format"] == {"type": "json_object"}
        return response('{"value":"invalid"}' if attempts == 1 else '{"value":7}')

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        result = await HTTPStructuredLLM(settings(1), client).ainvoke_structured(
            [{"role": "user", "content": "protocol input"}], Payload
        )
    assert result == Payload(value=7)
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,finish", [(401, "stop"), (200, "length")])
async def test_permanent_http_error_or_truncated_content_fails(status: int, finish: str) -> None:
    """鉴权永久错误不重试，截断 JSON 不得作为完整对象成功返回。"""
    attempts = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """计数服务请求，避免重试造成无限调用。"""
        nonlocal attempts
        attempts += 1
        return httpx.Response(status) if status != 200 else response('{"value":7}', finish)

    async with httpx.AsyncClient(
        base_url="https://model.test/v1/", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ProviderUnavailableError):
            await HTTPStructuredLLM(settings(), client).ainvoke_structured([], Payload)
    assert attempts == 1


class FakeLLM:
    """结构化模型测试替身，返回待验证的抽取快照。"""

    def __init__(self, evidence: ClinicalEvidence) -> None:
        """注入可审查的抽取输出。"""
        self.evidence = evidence

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """本用例不调用普通文本模型。"""
        raise AssertionError("unexpected text invocation")

    async def ainvoke_structured(
        self, messages: list[dict[str, str]], schema: type[StructuredResult]
    ) -> StructuredResult:
        """验证节点专用 Prompt 后按实际 Schema 返回。"""
        assert "不得判断六经" in messages[0]["content"]
        return schema.model_validate(self.evidence.model_dump())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,polarity,fragment",
    [
        ("pulse_tight", "positive", "脉浮"),
        ("fever", "positive", "无发热"),
        ("fever", "positive", "不存在的原文"),
    ],
)
async def test_parser_rejects_invented_entity_or_changed_negation(
    key: str, polarity: str, fragment: str
) -> None:
    """LLM 输出 Schema 合法仍需事实核对，不能凭合法 JSON 接受虚构。"""
    evidence = ClinicalEvidence(
        findings=[
            Finding.model_validate(
                {
                    "key": key,
                    "polarity": polarity,
                    "value": True,
                    "source": "doctor_input",
                    "source_text": fragment,
                }
            )
        ]
    )
    with pytest.raises(ProviderUnavailableError):
        await parse_clinical_with_llm("无发热，脉浮", FakeLLM(evidence))


@pytest.mark.asyncio
async def test_parser_accepts_grounded_negative_with_regions() -> None:
    """结构化阴性事实可以填入四诊区，来源与值均受验证。"""
    evidence = ClinicalEvidence(
        sweating={"sweating": False},
        findings=[
            Finding(
                key="sweating",
                polarity="negative",
                value=False,
                source="doctor_input",
                source_text="无汗",
                time_scope="今日",
            )
        ],
    )
    result = await parse_clinical_with_llm("今日无汗", FakeLLM(evidence))
    assert result == evidence


@pytest.mark.asyncio
async def test_parser_cannot_turn_historical_fever_into_current_support() -> None:
    """原文虽然含发热，省略昨日仍会改变事实语义，必须拒绝。"""
    evidence = ClinicalEvidence(
        findings=[
            Finding(
                key="fever",
                polarity="positive",
                value=True,
                source="doctor_input",
                source_text="发热",
                time_scope=None,
            )
        ]
    )
    with pytest.raises(ProviderUnavailableError):
        await parse_clinical_with_llm("昨日发热", FakeLLM(evidence))


@pytest.mark.asyncio
async def test_model_missing_explicit_negative_fails_instead_of_succeeding_empty() -> None:
    """合法空 JSON 模型结果不能吞掉医生已明确的阴性事实。"""
    with pytest.raises(ProviderUnavailableError):
        await parse_clinical_with_llm("无汗", FakeLLM(ClinicalEvidence()))


@pytest.mark.asyncio
async def test_parser_cannot_fabricate_doctor_confirmed_treatment() -> None:
    """即使治疗文字位于输入，模型也不能冒充医生确认前后事实。"""
    evidence = ClinicalEvidence(
        treatment_events=[
            TreatmentEvent(event_id="t1", description="已用治疗", source_text="已用治疗")
        ]
    )
    with pytest.raises(ProviderUnavailableError):
        await parse_clinical_with_llm("已用治疗", FakeLLM(evidence))


@pytest.mark.asyncio
async def test_parser_cannot_confirm_relationship_history() -> None:
    """模型不能以 Schema 合法冒充医生确认误治、病史完整或主经。"""
    evidence = ClinicalEvidence(
        relationship_history=RelationshipHistory(
            treatment_error_confirmed=True, source_text="已用治疗"
        )
    )
    with pytest.raises(ProviderUnavailableError):
        await parse_clinical_with_llm("已用治疗", FakeLLM(evidence))
