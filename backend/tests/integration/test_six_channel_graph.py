"""真实 SQLite 子图回归：验证协议顺序、中间阶段恢复、隔离和医生编辑事实保留。"""

from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.core.config import Settings
from backend.app.graph.six_channel_graph import build_six_channel_graph
from backend.app.knowledge.service import KnowledgeService
from backend.app.nodes.clinical_parser import parse_clinical_text
from backend.app.persistence.checkpoint import build_checkpoint_runtime
from backend.app.services.reasoning_service import ReasoningService

STAGES = [
    "clinical_parser",
    "six_channel_screen",
    "formula_branch_retrieval",
    "differential_verification",
    "evidence_topk",
]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_restart_resumes_remaining_stages(tmp_path: Path) -> None:
    """在关系节点之前停下，重开真实 saver 后只恢复剩余节点，保留患者来源。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "graph.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    first = await build_checkpoint_runtime(settings)
    try:
        graph = build_six_channel_graph(knowledge, first.checkpointer)
        result = await graph.ainvoke(
            {
                "raw_input": "发热，无汗，恶寒，脉浮",
                "source": "record",
                "thread_id": thread_id,
                "top_k": 3,
            },
            config,
            interrupt_before=["differential_verification"],
        )
        assert result["completed_stages"] == STAGES[:3]
        assert (await graph.aget_state(config)).next == ("differential_verification",)
    finally:
        await first.close()
    second = await build_checkpoint_runtime(settings)
    try:
        restored = build_six_channel_graph(knowledge, second.checkpointer)
        result = await restored.ainvoke(None, config)
        assert result["completed_stages"] == STAGES
        assert result["reasoning_paths"]
        assert result["evidence_catalog"]
        assert all(path.evidence_assessments for path in result["reasoning_paths"])
        assert all(
            path.completeness_detail and path.treatment_summary
            for path in result["reasoning_paths"]
        )
        saved = await restored.aget_state(config)
        assert (
            saved.values["reasoning_paths"][0].evidence_assessments
            == result["reasoning_paths"][0].evidence_assessments
        )
        assert (
            saved.values["reasoning_paths"][0].completeness_detail
            == result["reasoning_paths"][0].completeness_detail
        )
        assert (
            saved.values["reasoning_paths"][0].comparison_with_next
            == result["reasoning_paths"][0].comparison_with_next
        )
        assert result["syndrome_relationships"][0].relationship_type == "uncertain"
        assert all(finding.source == "record" for finding in result["clinical_evidence"].findings)
        assert (await restored.aget_state(config)).next == ()
        assert (await restored.aget_state({"configurable": {"thread_id": "other"}})).values == {}
    finally:
        await second.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_service_preserves_edited_evidence_and_isolates_threads(tmp_path: Path) -> None:
    """编辑后的三值事实作为权威输入，不重新解析替换，独立运行使用不同线程。"""
    settings = Settings(
        _env_file=None,
        checkpoint_backend="sqlite",
        langgraph_sqlite_path=str(tmp_path / "service.db"),
    )
    knowledge = await KnowledgeService.from_directory(settings.knowledge_directory, "1.2")
    runtime = await build_checkpoint_runtime(settings)
    try:
        service = ReasoningService(knowledge, runtime.checkpointer)
        evidence = parse_clinical_text("发热，无汗，恶寒，脉浮", source="record")
        evidence.course = "三日"
        evidence.findings[0].polarity = "unknown"
        evidence.findings[0].value = None
        first = await service.analyze_evidence(evidence)
        second = await service.analyze_text("口苦", source="doctor_input")
        assert first.clinical_evidence == evidence
        assert first.completed_stages == STAGES
        assert first.thread_id != second.thread_id
        snapshot = await service.graph.aget_state({"configurable": {"thread_id": first.thread_id}})
        assert snapshot.values["clinical_evidence"].course == "三日"
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    ("text", "first"),
    [
        ("起病两天，发热恶风，微微汗出，头痛，脉浮而缓", "桂枝汤"),
        ("发热恶寒，无汗，头痛，身痛，脉浮紧", "麻黄汤"),
    ],
)
async def test_full_graph_uses_differential_to_separate_paths(text: str, first: str) -> None:
    """完整五节点读取来源鉴别；正反事实切换时排序随证据变化。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = ReasoningService(knowledge, formula_branch_recall_top_n=12)
    result = await service.analyze_text(text)
    assert result.completed_stages == STAGES
    assert result.reasoning_paths[0].formula_pattern == first
    assert result.reasoning_paths[0].score_components["differential_factor"] > 0
    assert any(
        set(item.candidate_ids) == {"桂枝汤", "麻黄汤"} for item in result.differential_results
    )
