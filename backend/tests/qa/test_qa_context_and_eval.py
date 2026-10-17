"""QA 多轮上下文和本地评测报告测试。

使用合成会话与 HTTP 夹具验证旧引用剥离、错误留档，不调用在线 LLM/RAGAS。
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from eval.qa_v21_eval import run

from backend.app.schemas.knowledge import KnowledgeChatResponse, QAMessage
from backend.app.services.qa_context import build_qa_context


def test_history_keeps_pairs_and_strips_old_citations() -> None:
    """新轮只继承已完成正文，不把旧引用或失败回答当成本轮证据。"""
    now = datetime.now(UTC)
    session_id = uuid4()
    messages: list[QAMessage] = []
    for index in range(8):
        fields = dict(
            session_id=session_id,
            client_message_id=uuid4(),
            request_digest="test",
            knowledge_version="1.2",
            model_id="synthetic",
            prompt_digest="test",
            expires_at=now + timedelta(seconds=30),
            created_at=now,
            updated_at=now,
        )
        turn = uuid4()
        messages.extend(
            [
                QAMessage(
                    message_id=uuid4(),
                    sequence=index * 2,
                    role="user",
                    content=f"合成问题{index}",
                    status="completed",
                    **fields,
                ),
                QAMessage(
                    message_id=turn,
                    sequence=index * 2 + 1,
                    role="assistant",
                    content=f"合成回答{index}[1]",
                    status="completed",
                    response=KnowledgeChatResponse(
                        session_id=session_id,
                        turn_id=turn,
                        answer=f"合成回答{index}[1]",
                        status="answered",
                        knowledge_version="1.2",
                        duration_ms=1,
                    ),
                    **fields,
                ),
            ]
        )
    context = build_qa_context(messages, 3500)
    assert len(context) == 12 and context[0]["content"] == "合成问题2"
    assert all("[1]" not in item["content"] for item in context)
    assert sum(len(item["content"]) for item in build_qa_context(messages, 20)) <= 20
    messages[-1].response.status = "no_evidence"
    assert all("合成回答7" not in item["content"] for item in build_qa_context(messages, 3500))


@pytest.mark.asyncio
async def test_eval_records_safe_failure_and_continues(tmp_path, monkeypatch) -> None:
    """单轮服务失败仍落盘并继续后续题目，报告保留业务码而非原始异常文本。"""
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps({"case_id": "synthetic", "questions": ["甲", "乙"]}), encoding="utf-8"
    )
    original = httpx.AsyncClient
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        """会话正常，问题由安全业务错误响应拒绝。"""
        nonlocal calls
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json={"session_id": str(uuid4())})
        calls += 1
        return httpx.Response(503, json={"error": {"code": "QA_STAGE_TIMEOUT"}})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    report = await run(
        argparse.Namespace(
            dataset=dataset,
            output=tmp_path / "report.json",
            base_url="https://test",
            timeout=1,
            limit=None,
            ragas=False,
        )
    )
    assert calls == 2 and report["summary"]["failures"] == 2
    assert all(row["error_code"] == "QA_STAGE_TIMEOUT" for row in report["rows"])
    assert (
        json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["ragas_scored"] is False
    )
