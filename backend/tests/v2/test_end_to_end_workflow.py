"""V2 S1-S5 纵向测试：验证候选来源、审核中断和医生恢复。"""

from pathlib import Path
from typing import TypeVar

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from backend.app.clinical_v2.schemas import (
    ChannelLocalizationV2,
    ComparisonV2,
    S1ExtractionDraft,
)
from backend.app.graph.clinical_v2_graph import build_clinical_v2_graph
from backend.app.schemas.evidence import Evidence

T = TypeVar("T", bound=BaseModel)


class FakeKnowledge:
    """提供真实 Evidence 契约形状的最小 KnowledgeService 替身。"""

    async def search_formula(
        self, query: str, channels: list[str] | None = None, top_k: int = 20
    ) -> list[Evidence]:
        """返回两个不同方证及同方第二来源。"""
        return [
            Evidence(
                evidence_id="1.2:gui-1",
                knowledge_id="gui-1",
                knowledge_type="formula_pattern",
                knowledge_version="1.2",
                source="经典",
                source_level="L3",
                content="桂枝汤来源",
                retrieval_score=0.9,
                metadata={
                    "formula_name": "桂枝汤",
                    "review_status": "ok",
                    "canonical_six_channels": ["太阳"],
                },
                retrieval_method="hybrid",
            ),
            Evidence(
                evidence_id="1.2:gui-2",
                knowledge_id="gui-2",
                knowledge_type="formula_pattern",
                knowledge_version="1.2",
                source="经典",
                source_level="L3",
                content="桂枝汤另一来源",
                retrieval_score=0.8,
                metadata={
                    "formula_name": "桂枝汤",
                    "review_status": "ok",
                    "canonical_six_channels": ["太阳"],
                },
                retrieval_method="hybrid",
            ),
            Evidence(
                evidence_id="1.2:ma-1",
                knowledge_id="ma-1",
                knowledge_type="formula_pattern",
                knowledge_version="1.2",
                source="经典",
                source_level="L3",
                content="麻黄汤来源",
                retrieval_score=0.7,
                metadata={
                    "formula_name": "麻黄汤",
                    "review_status": "ok",
                    "canonical_six_channels": ["太阳"],
                },
                retrieval_method="hybrid",
            ),
        ]

    async def search_differential(
        self, candidates: list[str], query: str | None = None, top_k: int = 10
    ) -> list[Evidence]:
        """本测试不需要额外鉴别来源。"""
        return []


class FullLLM:
    """按 S1/S2/S4 Schema 返回合法结果，记录调用阶段。"""

    def __init__(self) -> None:
        """初始化调用计数。"""
        self.calls: list[str] = []

    async def ainvoke_text(self, messages: list[dict[str, str]]) -> str:
        """V2 临床节点不调用自由文本端口。"""
        raise AssertionError("不应调用文本模型")

    async def ainvoke_structured(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """只返回当前节点允许的结构化对象。"""
        self.calls.append(schema.__name__)
        if schema is S1ExtractionDraft:
            return schema.model_validate(
                {
                    "findings": [
                        {
                            "local_id": "a",
                            "concept": "发热",
                            "polarity": "positive",
                            "time_scope": "current",
                            "source_text": "发热",
                        }
                    ]
                }
            )
        if schema is ChannelLocalizationV2:
            import json

            payload = json.loads(messages[-1]["content"])
            fact_id = payload["findings"][0]["fact_id"]
            return schema.model_validate(
                {
                    "status": "resolved",
                    "candidates": [
                        {
                            "candidate_id": "channel-1",
                            "six_channels": ["太阳"],
                            "support_fact_ids": [fact_id],
                            "retrieval_focus": ["表证"],
                            "short_basis": "当前事实有据",
                        }
                    ],
                    "proceed_to_retrieval": True,
                    "clarification_questions": [],
                    "note": "可检索",
                }
            )
        assert schema is ComparisonV2
        import json

        payload = json.loads(messages[-1]["content"])
        candidates = payload["retrieval"]["formula_candidates"]
        first = candidates[0]
        fact_id = payload["clinical_evidence"]["findings"][0]["fact_id"]
        evidence_id = first["source_evidence"][0]["evidence_id"]
        return schema.model_validate(
            {
                "status": "ranked",
                "ordered_paths": [
                    {
                        "path_id": "path-1",
                        "candidate_id": first["candidate_id"],
                        "six_channels": ["太阳"],
                        "formula_name": first["formula_name"],
                        "rank": 1,
                        "support_fact_ids": [fact_id],
                        "counter_fact_ids": [],
                        "missing_dimensions": [],
                        "support_evidence_ids": [evidence_id],
                        "counter_evidence_ids": [],
                        "brief_rationale": "来源支持",
                        "differential_summary": "暂无",
                        "verification_status": "pending",
                    }
                ],
                "next_questions": [],
            }
        )


async def test_s1_to_s5_review_resume(tmp_path: Path) -> None:
    """完整链路进入 S5 后暂停，医生 accept 才变成 completed。"""
    llm = FullLLM()
    config = {"configurable": {"thread_id": "clinical-v2:e2e"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.sqlite")) as saver:
        graph = build_clinical_v2_graph(llm, saver, knowledge=FakeKnowledge())
        first = await graph.ainvoke({"consultation_id": "e2e", "clinical_text": "发热"}, config)
        assert first["status"] == "awaiting_doctor_review"
        assert first["paused_at"] == "s5_doctor_review"
        assert first["retrieval"]["formula_candidates"]
        assert (await graph.aget_state(config)).next == ("s5_doctor_review",)
        completed = await graph.ainvoke(
            Command(
                resume={
                    "kind": "review",
                    "consultation_id": "e2e",
                    "revision": 0,
                    "action_id": "review-1",
                    "action_turn": 2,
                    "review": {"action": "accept"},
                }
            ),
            config,
        )
        assert completed["status"] == "completed"
        assert completed["review"]["action"] == "accept"
        assert not (await graph.aget_state(config)).next
        assert llm.calls == ["S1ExtractionDraft", "ChannelLocalizationV2", "ComparisonV2"]
