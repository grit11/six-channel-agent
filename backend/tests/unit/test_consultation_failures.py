"""会诊失败边界：推理异常仍保存失败状态，重跑不可呈现上一轮旧结果。"""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from backend.app.core.exceptions import ProviderUnavailableError
from backend.app.persistence.repositories import InMemoryConsultationRepository
from backend.app.persistence.runtime import RuntimeWrite
from backend.app.schemas.consultation import (
    ConsultationCreateRequest,
    ConsultationRerunRequest,
    ConsultationSnapshot,
)
from backend.app.services.consultation_service import ConsultationService


@pytest.mark.asyncio
async def test_failed_run_retains_snapshot_and_clears_old_result() -> None:
    """重跑失败的快照可供医生查看错误状态，旧推理不能被误认为本轮有效输出。"""
    repository = InMemoryConsultationRepository()
    reasoning = AsyncMock()
    reasoning.analyze_text.side_effect = ProviderUnavailableError("模型服务暂不可用")
    service = ConsultationService(
        repository,
        reasoning,
        protocol_version="1.0",
        knowledge_base_version="1.2",
        scoring_config_version="1.0",
    )
    # 捕获已创建的业务 ID，仅验证故障记录，不通过异常消息暴露患者内容。
    create = repository.create
    saved_ids: list[UUID] = []

    async def capture(
        snapshot: ConsultationSnapshot, *, runtime_records: list[RuntimeWrite] | None = None
    ) -> ConsultationSnapshot:
        """记录创建标识，以便验证客户端收到异常后的业务快照。"""
        saved_ids.append(snapshot.consultation_id)
        return await create(snapshot, runtime_records=runtime_records)

    repository.create = capture
    with pytest.raises(ProviderUnavailableError):
        await service.create(ConsultationCreateRequest(text="离线测试"))
    snapshot = await service.get_snapshot(saved_ids[0])
    assert snapshot.status == "failed"
    assert snapshot.turn == 3
    assert snapshot.reasoning is None
    with pytest.raises(ProviderUnavailableError):
        await service.rerun(snapshot.consultation_id, ConsultationRerunRequest(expected_turn=3))
    updated = await service.get_snapshot(snapshot.consultation_id)
    assert updated.status == "failed" and updated.turn == 5
    assert updated.reasoning is None
