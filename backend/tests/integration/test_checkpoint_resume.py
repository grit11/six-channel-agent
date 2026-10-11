"""真实 SQLite 图状态测试：重启后仍恢复同 thread 的 interrupt。"""

from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from backend.app.core.config import Settings
from backend.app.persistence.checkpoint import build_checkpoint_runtime


class SupplementState(TypedDict):
    """只用于验证恢复语义的测试状态，不实现医学推理。"""

    text: str


def supplement(state: SupplementState) -> dict[str, str]:
    """测试医生补充暂停点，恢复时取同 thread 的补充信息。"""
    answer = interrupt({"question": "补充已知事实", "text": state["text"]})
    return {"text": str(answer)}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_restart_restores_interrupt(tmp_path: Path) -> None:
    """关闭并重开 SQLite saver 后，同 thread 恢复而其他 thread 保持隔离。"""
    settings = Settings(
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "restart.sqlite3"),
    )
    builder = StateGraph(SupplementState)
    builder.add_node("supplement", supplement)
    builder.add_edge(START, "supplement")
    builder.add_edge("supplement", END)
    config = {"configurable": {"thread_id": "consultation-1"}}
    first = await build_checkpoint_runtime(settings)
    try:
        graph = builder.compile(checkpointer=first.checkpointer)
        result = await graph.ainvoke({"text": "初始输入"}, config)
        assert result["__interrupt__"]
    finally:
        await first.close()
    second = await build_checkpoint_runtime(settings)
    try:
        restored = builder.compile(checkpointer=second.checkpointer)
        snapshot = await restored.aget_state(config)
        assert snapshot.values["text"] == "初始输入"
        result = await restored.ainvoke(Command(resume="医生补充"), config)
        assert result["text"] == "医生补充"
        other = await restored.aget_state({"configurable": {"thread_id": "consultation-2"}})
        assert other.values == {}
    finally:
        await second.close()
