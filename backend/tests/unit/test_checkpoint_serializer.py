"""V2 Checkpoint 注册测试，保留任意类型拒绝与公开 DTO 恢复保障。"""

from pydantic import BaseModel

from backend.app.clinical_v2.schemas import ClinicalEvidenceV2, SourceSpanV2
from backend.app.persistence.checkpoint import checkpoint_serializer


class UnregisteredModel(BaseModel):
    """未注册类不能恢复成可执行对象。"""

    value: str


def test_registered_clinical_dto_round_trip_and_unknown_rejection() -> None:
    """V2 DTO 正常往返，未注册类型仍被拒绝。"""
    serializer = checkpoint_serializer()
    for item in (
        ClinicalEvidenceV2(raw_text="测试", findings=[]),
        SourceSpanV2(start=0, end=2, text="测试"),
    ):
        assert serializer.loads_typed(serializer.dumps_typed(item)) == item
    rejected = serializer.loads_typed(serializer.dumps_typed(UnregisteredModel(value="禁止")))
    assert not isinstance(rejected, UnregisteredModel)
