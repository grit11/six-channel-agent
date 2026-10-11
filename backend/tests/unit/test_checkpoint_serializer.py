"""Checkpoint 类型注册回归：允许实际 DTO，拒绝没有登记的模型类。"""

from pydantic import BaseModel

from backend.app.persistence.checkpoint import checkpoint_serializer
from backend.app.schemas.clinical import ClinicalEvidence
from backend.app.schemas.reasoning import ReasoningStageSnapshot, SimilarCaseAssessment


class UnregisteredModel(BaseModel):
    """不应从图存储恢复的测试类。"""

    value: str


def test_registered_clinical_dto_round_trip_and_unknown_rejection() -> None:
    """启用受限序列化，不依赖当前 SDK 默认允许任意类的兼容模式。"""
    serializer = checkpoint_serializer()
    evidence = ClinicalEvidence(chief_complaint="测试")
    decoded = serializer.loads_typed(serializer.dumps_typed(evidence))
    assert isinstance(decoded, ClinicalEvidence)
    assert decoded == evidence
    blocked = serializer.loads_typed(serializer.dumps_typed(UnregisteredModel(value="禁止恢复")))
    assert not isinstance(blocked, UnregisteredModel)


def test_replay_and_case_registered_without_enabling_arbitrary_models() -> None:
    """只允许实际图 DTO；阶段 payload 仍为不可执行公开数据。"""
    serializer = checkpoint_serializer()
    for item in [
        ReasoningStageSnapshot(sequence=1, stage="topk", payload={"paths": []}),
        SimilarCaseAssessment(
            evidence_id="1.2:case_0001", title="医案", status="insufficient", reason="信息不足"
        ),
    ]:
        assert serializer.loads_typed(serializer.dumps_typed(item)) == item
