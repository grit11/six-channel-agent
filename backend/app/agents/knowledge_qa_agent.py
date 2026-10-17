"""QA V2 生成式知识 Agent。

职责：使用 LangChain create_agent 驱动四个只读知识工具，并返回模型生成的中文回答。
边界：不读取临床状态、不编排 QA StateGraph、不生成无来源诊断、处方或剂量。
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agents.knowledge_qa_tools import (
    EvidenceRegistry,
    KnowledgeToolFactory,
    QAError,
)
from backend.app.knowledge.service import KnowledgeService

PROMPT_PATH = Path(__file__).parent / "prompts" / "knowledge_qa_v2.md"


def visible_text(content: object) -> str:
    """只读取标准正文块，忽略 reasoning、工具参数和未知扩展字段。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return ""


@dataclass
class AgentTurnResult:
    """Agent 单轮输出和工具审计数据。"""

    answer: str
    registry: EvidenceRegistry
    messages: list[Any]
    usage: dict[str, int | None]
    llm_calls: int = 0
    covered_calls: int = 0
    time_to_first_token_ms: int | None = None


class KnowledgeQAAgent:
    """面向医生的只读知识聊天 Agent；每轮使用独立引用注册表。"""

    def __init__(self, knowledge: KnowledgeService, model: BaseChatModel | None = None) -> None:
        """注入 KnowledgeService 和 QA-only ChatModel，不接收临床图或会诊状态。"""
        self.knowledge = knowledge
        self.model = model

    @property
    def configured(self) -> bool:
        """返回是否装配了支持 LangChain Tool Calling 的聊天模型。"""
        return self.model is not None

    async def run(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        *,
        on_event: Callable[[str, dict[str, object]], Awaitable[None]] | None = None,
    ) -> AgentTurnResult:
        """执行一轮受预算限制的 RAG 问答并保留本轮真实来源。"""
        if self.model is None:
            raise RuntimeError("QA ChatModel 未配置")
        registry = EvidenceRegistry(self.knowledge.version)
        registry.on_event = on_event
        factory = KnowledgeToolFactory(self.knowledge, registry)
        prompt = await asyncio.to_thread(PROMPT_PATH.read_text, encoding="utf-8")
        agent = create_agent(
            self.model,
            tools=factory.build(),
            system_prompt=prompt,
            middleware=cast(
                Any,
                [
                    ModelCallLimitMiddleware(run_limit=4, exit_behavior="error"),
                    ToolCallLimitMiddleware(run_limit=5, exit_behavior="error"),
                ],
            ),
            name="knowledge_qa_v2",
        )
        messages: list[Any] = []
        for item in (history or [])[-12:]:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append(HumanMessage(content=message))
        started = perf_counter()
        ttft = None
        try:
            if on_event is None:
                result = await agent.ainvoke({"messages": messages})
            else:
                # astream_events 只发布正文 content；reasoning/tool 参数及外部工具文本不外传。
                # on_chain_end 的最终消息是权威结果，增量只属于临时 UI。
                result = {}
                async for event in agent.astream_events({"messages": messages}, version="v2"):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        text = visible_text(getattr(chunk, "content", None))
                        if (
                            isinstance(text, str)
                            and text
                            and not getattr(chunk, "tool_call_chunks", None)
                        ):
                            if ttft is None:
                                ttft = int((perf_counter() - started) * 1000)
                            await on_event("qa.answer_delta", {"text": text, "provisional": True})
                    if event["event"] == "on_chain_end" and not event.get("parent_ids"):
                        output = event["data"].get("output")
                        if isinstance(output, dict):
                            result = output
        except QAError:
            raise
        except TimeoutError as exc:
            raise QAError("QA_TOOL_TIMEOUT", "知识问答模型或工具调用超时") from exc
        except Exception as exc:
            # LangChain 中间件会以普通运行时异常报告模型限额；不能把这类失败伪装成回答成功。
            message = str(exc).lower()
            code = (
                "QA_TOOL_LIMIT_EXCEEDED"
                if "limit" in message or "budget" in message or "run_limit" in message
                else "QA_PROVIDER_UNAVAILABLE"
            )
            raise QAError(code, "知识问答模型暂时不可用，请稍后重试") from exc
        output_messages = list(result.get("messages", []))
        answer = ""
        usage: dict[str, int | None] = {"input_tokens": None, "output_tokens": None}
        # 输出包含传入历史；计费审计和正式回答只取本轮新增消息，避免重复计算旧轮 usage。
        new_messages = output_messages[len(messages) :]
        for item in reversed(new_messages):
            content = visible_text(getattr(item, "content", None))
            if (
                isinstance(item, AIMessage)
                and isinstance(content, str)
                and content.strip()
                and not item.tool_calls
            ):
                answer = content.strip()
                break
        seen: set[str] = set()
        calls = covered = 0
        if not answer:
            raise QAError("QA_PROVIDER_RESPONSE_INVALID", "模型未返回可核验的完整回答")
        for index, item in enumerate(new_messages):
            if not isinstance(item, AIMessage):
                continue
            identity = str(item.id or f"generation-{index}")
            if identity in seen:
                continue
            seen.add(identity)
            calls += 1
            metadata: Mapping[str, object] = item.usage_metadata or {}
            input_tokens = metadata.get("input_tokens")
            output_tokens = metadata.get("output_tokens")
            if (
                isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and input_tokens >= 0
                and isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and output_tokens >= 0
            ):
                covered += 1
                usage["input_tokens"] = (usage["input_tokens"] or 0) + input_tokens
                usage["output_tokens"] = (usage["output_tokens"] or 0) + output_tokens
        return AgentTurnResult(
            answer=answer,
            registry=registry,
            messages=output_messages,
            usage=usage,
            llm_calls=calls,
            covered_calls=covered,
            time_to_first_token_ms=ttft,
        )
