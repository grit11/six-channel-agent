"""共享模型初始化参数回归测试。

通过合成 HTTP 响应验证临床与 QA 的推理强度配置一致，不访问真实模型、知识或数据库。
只核对供应商协议参数，不评价医学推理，也不保存隐藏推理。
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from backend.app.core.config import Settings
from backend.app.core.llm import HTTPStructuredLLM, build_llm_provider_options
from backend.app.core.ports import S4StreamPolicy


class Output(BaseModel):
    """合成响应契约，不包含医学含义。"""

    value: int


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("mode", ["text", "structured", "S1", "S2", "S4", "fallback"])
async def test_all_clinical_paths_share_initialized_effort(effort: str, mode: str) -> None:
    """所有临床传输入口复用初始化参数，非流式对照也不丢失推理强度。"""
    captured: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        """记录实际 JSON 参数并构造完整合成响应。"""
        payload = json.loads(request.content)
        captured.append(payload)
        choice = {"delta": {"content": '{"value":7}'}, "finish_reason": "stop"}
        if payload["stream"]:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps({"choices": [choice]}) + "\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"value":7}'}, "finish_reason": "stop"}]},
        )

    config = Settings(
        _env_file=None,
        llm_model="glm-5.3",
        llm_reasoning_effort=effort,
        llm_max_retries=0,
        clinical_stream_enabled=mode != "fallback",
    )
    async with httpx.AsyncClient(
        base_url="https://model.test/", transport=httpx.MockTransport(handle)
    ) as client:
        llm = HTTPStructuredLLM(config, client)
        messages = [{"role": "system", "content": "合成协议测试"}]
        if mode == "text":
            assert await llm.ainvoke_text([]) == '{"value":7}'
        elif mode == "structured":
            assert await llm.ainvoke_structured([], Output) == Output(value=7)
        elif mode == "S4":
            assert await llm.ainvoke_structured_stream(
                [], Output, policy=S4StreamPolicy(total_seconds=2)
            ) == Output(value=7)
        else:
            assert await llm.ainvoke_clinical_structured_stream(
                messages, Output, stage="S1" if mode == "fallback" else mode
            ) == Output(value=7)
    assert len(captured) == 1
    assert captured[0]["reasoning_effort"] == effort
    assert captured[0]["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"])
def test_empty_effort_preserves_provider_default(model: str) -> None:
    """不配置时不偷偷注入 low，是否思考与强度均沿用供应商默认。"""
    config = Settings(_env_file=None, llm_model=model, llm_reasoning_effort="")
    assert build_llm_provider_options(config) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["test-model", "qwen-test", "glm-4-flash-250414"])
async def test_switching_model_does_not_inject_glm_options(model: str) -> None:
    """保留环境中的 GLM 强度配置切换到其他模型时，请求参数保持原样。"""

    def handle(request: httpx.Request) -> httpx.Response:
        """从真实序列化请求检查参数隔离，而不只检查辅助函数。"""
        payload = json.loads(request.content)
        assert "thinking" not in payload
        assert "reasoning_effort" not in payload
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    async with httpx.AsyncClient(
        base_url="https://model.test/", transport=httpx.MockTransport(handle)
    ) as client:
        llm = HTTPStructuredLLM(
            Settings(_env_file=None, llm_model=model, llm_reasoning_effort="high"), client
        )
        assert await llm.ainvoke_text([]) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,effort",
    [("glm-5.3", "high"), ("glm-5.3-flash", "max"), ("other-model", "low"), ("glm-5.3", "")],
)
async def test_qa_factory_preserves_tool_calling_with_shared_options(
    model: str, effort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 ChatOpenAI 序列化共享参数后仍可绑定工具，QA Agent 逻辑无须改动。"""
    from langchain_openai import ChatOpenAI

    from backend.app.core import dependencies

    def handle(request: httpx.Request) -> httpx.Response:
        """检查 SDK 最终请求；避免 extra_body 嵌套或参数被 SDK 丢弃。"""
        payload = json.loads(request.content)
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "lookup"
        assert "extra_body" not in payload
        if model.startswith("glm-5.3") and effort:
            assert payload["thinking"] == {"type": "enabled"}
            assert payload["reasoning_effort"] == effort
        else:
            assert "thinking" not in payload
            assert "reasoning_effort" not in payload
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "合成回答"},
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        def create_model(**kwargs: object) -> ChatOpenAI:
            """只替换异步传输为本地替身，保留真实 SDK 初始化和序列化路径。"""
            return ChatOpenAI(**kwargs, http_async_client=client)

        monkeypatch.setattr(dependencies, "ChatOpenAI", create_model)
        chat = dependencies.build_qa_chat_model(
            Settings(
                _env_file=None,
                llm_base_url="https://model.test/v1",
                llm_model=model,
                llm_api_key="test-key",
                llm_reasoning_effort=effort,
            )
        )
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "只读合成工具",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result = await chat.bind_tools([tool]).ainvoke("合成问题")
        assert result.content == "合成回答"


@pytest.mark.asyncio
async def test_recovery_digest_changes_when_reasoning_effort_changes() -> None:
    """改变推理设置后不直接恢复旧指纹任务，医学消息本身不变。"""
    async with httpx.AsyncClient() as client:
        base = Settings(_env_file=None, llm_model="glm-5.3", llm_reasoning_effort="low")
        other = base.model_copy(update={"llm_reasoning_effort": "high"})
        assert (
            HTTPStructuredLLM(base, client).clinical_runtime_digest
            != HTTPStructuredLLM(other, client).clinical_runtime_digest
        )
