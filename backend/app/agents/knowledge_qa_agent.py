"""QA V2 生成式知识 Agent。

职责：使用 LangChain create_agent 驱动四个只读知识工具，并返回模型生成的中文回答。
边界：不读取临床状态、不编排 QA StateGraph、不生成无来源诊断、处方或剂量。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend.app.agents.knowledge_qa_tools import (
    EvidenceRegistry,
    KnowledgeToolFactory,
    QAError,
)
from backend.app.knowledge.service import KnowledgeService

PROMPT_PATH = Path(__file__).parent / "prompts" / "knowledge_qa_v2.md"


@dataclass
class AgentTurnResult:
    """Agent 单轮输出和工具审计数据。"""

    answer: str
    registry: EvidenceRegistry
    messages: list[Any]
    usage: dict[str, int | None]


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
        self, message: str, history: list[dict[str, str]] | None = None
    ) -> AgentTurnResult:
        """执行一轮受预算限制的 RAG 问答并保留本轮真实来源。"""
        if self.model is None:
            raise RuntimeError("QA ChatModel 未配置")
        registry = EvidenceRegistry(self.knowledge.version)
        factory = KnowledgeToolFactory(self.knowledge, registry)
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
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
        for item in (history or [])[-8:]:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append(HumanMessage(content=message))
        try:
            result = await agent.ainvoke({"messages": messages})
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
        for item in reversed(output_messages):
            content = getattr(item, "content", None)
            if isinstance(content, str) and content.strip():
                answer = content.strip()
                metadata = getattr(item, "usage_metadata", None) or {}
                usage = {
                    "input_tokens": metadata.get("input_tokens"),
                    "output_tokens": metadata.get("output_tokens"),
                }
                break
        return AgentTurnResult(
            answer=answer, registry=registry, messages=output_messages, usage=usage
        )
