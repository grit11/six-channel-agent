# V2 API 契约

临床推理入口统一使用 `/api/v2`：

- `POST /api/v2/clinical/parse`：只执行 S1 事实抽取，返回 `ClinicalEvidenceV2`；不会启动会诊。
- `POST /api/v2/consultations`：创建带稳定会诊 ID 的运行。
- `GET /api/v2/consultations/{id}`：读取 Repository 权威业务快照；Checkpoint 只承担图恢复。
- `POST /api/v2/consultations/{id}/resume`：通用人工恢复入口；S2/S5 也分别提供 `/clarify` 和 `/review`。
- `GET /api/v2/consultations/{id}/events`：按会诊游标重连公开 SSE 阶段事件。

S2 `unknown/conflicting` 必须暂停并保留 `paused_at=s2_clarification`；S5 在医生动作前不得标记 `completed`。所有事实、候选和证据引用均由 V2 DTO 校验。
