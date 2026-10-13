# V2 API 契约

临床推理入口统一使用 `/api/v2`：

- `POST /api/v2/clinical/parse`：只执行 S1 事实抽取，返回 `ClinicalEvidenceV2`；不会启动会诊。
- `POST /api/v2/consultations`：创建带稳定会诊 ID 的运行。
- `GET /api/v2/consultations/{id}`：读取 Repository 权威业务快照；Checkpoint 只承担图恢复。
- `POST /api/v2/consultations/{id}/resume`：通用人工恢复入口；S2/S5 也分别提供 `/clarify` 和 `/review`。
- `GET /api/v2/consultations/{id}/events`：按会诊游标重连公开 SSE 阶段事件。
- `GET /api/v2/consultations/{id}/events/page?after={event_id}`：读取最多 100 条公开事件；省略游标从头读取，返回满页时以最后一条 `event_id` 继续。不存在会诊返回 404，跨会诊或非法游标拒绝。

S2 `unknown/conflicting` 必须暂停并保留 `paused_at=s2_clarification`；S5 在医生动作前不得标记 `completed`。所有事实、候选和证据引用均由 V2 DTO 校验。

Console 在医生确认后保留病例、候选、证据和审核备注，只禁用已结束会诊的写入操作。工作台顶部及最终确认按钮右侧的“下一次问诊”共用重置逻辑；未完成会诊先完成或取消。历史记录进入独立只读详情，使用 `history` URL 参数恢复，不能覆盖当前工作台；不保存患者信息至浏览器缓存。

阶段完成/失败事件保存真实 `duration_ms`，模型事件保留供应商 usage 和重试。页面按 `event_id` 去重并按 turn/stage 展示各轮及累计耗时与输入/输出 Token。usage 缺失显示“未提供”，部分覆盖只展示已知用量与覆盖率；没有模型调用不推测 Token。S5 单独展示医生等待时间，不计入自动推理耗时。旧事件可通过阶段开始/结束时间推算耗时，丢失的用量不会补造。浏览器测试的合成 Token 只存在于隔离模型夹具。
