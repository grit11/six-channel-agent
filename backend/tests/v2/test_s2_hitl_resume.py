"""真实 V2 图的 S2 SQLite 中断、关闭、重启与恢复测试，不编写影子图。"""

from pathlib import Path
from typing import TypeVar

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from backend.app.clinical_v2.schemas import ChannelLocalizationV2, S1ExtractionDraft
from backend.app.graph.clinical_v2_graph import build_clinical_v2_graph

T = TypeVar("T", bound=BaseModel)


class UncertainLLM:
    """只模拟 S1 与 S2 草稿；未知路由不得调用任何方证比较端口。"""

    def __init__(self) -> None:
        """记录真实图对模型端口的调用次数。"""
        self.calls: list[str] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """本图不使用非结构化模型接口。"""
        raise AssertionError("不允许文本生成")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """仅为事实和未知六经提供合法输出，其余阶段调用立即失败。"""
        self.calls.append(schema.__name__)
        if schema is S1ExtractionDraft:
            return schema.model_validate(
                {
                    "findings": [
                        {
                            "local_id": "a",
                            "concept": "不适",
                            "polarity": "positive",
                            "time_scope": "current",
                            "source_text": "难受",
                        }
                    ]
                }
            )
        assert schema is ChannelLocalizationV2, "S2 未知时不能调用 S4"
        return schema.model_validate(
            {
                "status": "unknown",
                "candidates": [],
                "proceed_to_retrieval": False,
                "clarification_questions": [
                    {
                        "question_id": "q1",
                        "question": "有何寒热表现？",
                        "targets": ["寒热"],
                        "source_fact_ids": [],
                        "expected_effect": "补足定位依据",
                    }
                ],
                "note": "依据不足",
            }
        )


async def test_unknown_survives_sqlite_restart_then_defers(tmp_path: Path) -> None:
    """同一物理线程持久暂停，重启后无法补证时终止未决，无 S3/S4。"""
    path = str(tmp_path / "state.sqlite")
    llm = UncertainLLM()
    config = {"configurable": {"thread_id": "clinical-v2:restart"}}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_clinical_v2_graph(llm, saver)
        first = await graph.ainvoke(
            {"consultation_id": "test", "clinical_text": "难受", "revision": 0}, config
        )
        assert first["paused_at"] == "s2_clarification"
        assert first["retrieval"] is None and first["comparison"] is None
        assert (await graph.aget_state(config)).next == ("s2_clarification_gate",)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_clinical_v2_graph(llm, saver)
        result = await graph.ainvoke(
            Command(
                resume={
                    "kind": "clarify",
                    "consultation_id": "test",
                    "revision": 0,
                    "action_id": "defer",
                    "action_turn": 2,
                    "no_additional_information": True,
                }
            ),
            config,
        )
        assert result["status"] == "undetermined"
        assert not (await graph.aget_state(config)).next
        assert llm.calls == ["S1ExtractionDraft", "ChannelLocalizationV2"]


async def test_added_text_reexecutes_s1_s2_in_same_thread(tmp_path: Path) -> None:
    """补证使事实版本递增，旧后序结果清空，不能重复使用旧轮医学结论。"""
    llm = UncertainLLM()
    config = {"configurable": {"thread_id": "clinical-v2:clarify"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        graph = build_clinical_v2_graph(llm, saver)
        await graph.ainvoke({"consultation_id": "test", "clinical_text": "难受"}, config)
        result = await graph.ainvoke(
            Command(
                resume={
                    "kind": "clarify",
                    "consultation_id": "test",
                    "revision": 0,
                    "action_id": "add",
                    "action_turn": 2,
                    "clinical_text": "难受，仍无明确线索",
                }
            ),
            config,
        )
        assert result["revision"] == 1
        assert result["clinical_evidence"]["doctor_revision"] == 1
        assert result["retrieval"] is None and result["comparison"] is None
        assert len(llm.calls) == 4
