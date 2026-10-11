"""可靠运行协调：持久任务、租约、取消、结果发布和人工动作补偿。

依赖业务 Repository 与固定图接口；不提供医学规则或自动医生决策。
同库提交承接业务意图，跨库 checkpoint 使用可重试补偿，禁止宣称跨库原子事务。
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from backend.app.core.exceptions import (
    InvalidBusinessRequestError,
    InvalidStateTransitionError,
)
from backend.app.graph.main_graph import HumanGraphCommand
from backend.app.observability.progress import progress_scope
from backend.app.persistence.runtime import next_lease, operation_context, receipt_context
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.events import StreamEventName
from backend.app.schemas.reasoning_api import ReasoningResponse
from backend.app.schemas.runtime import GraphAction, RunJob

if TYPE_CHECKING:
    from backend.app.services.consultation_service import ConsultationService


class ReliableRuntime:
    """应用级运行协调，多个实例共享 PostgreSQL 锁、任务与持久序列。"""

    def __init__(
        self,
        service: "ConsultationService",
        *,
        lease_seconds: float = 30,
        poll_seconds: float = 1,
        timeout_seconds: float = 300,
        max_attempts: int = 3,
        max_concurrent: int = 4,
    ) -> None:
        """注入有界重试/超时，持久任务并发受信号量限制。"""
        self.service = service
        self.store = service.repository.runtime
        self.owner = str(uuid4())
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: dict[UUID, asyncio.Task[ConsultationSnapshot]] = {}
        self.worker: asyncio.Task[None] | None = None
        self.stopping = False

    async def start(self) -> None:
        """启动恢复轮询；不会在没有业务存储的场景恢复 checkpoint 业务数据。"""
        if self.worker is None:
            self.worker = asyncio.create_task(self._loop())

    def launch(self, job: RunJob) -> asyncio.Task[ConsultationSnapshot]:
        """加速本地已提交任务，子任务不得继承请求收据导致覆盖旧响应。"""
        existing = self.tasks.get(job.consultation_id)
        if existing is not None and not existing.done():
            return existing
        token = receipt_context.set(None)
        try:
            task = asyncio.create_task(self.execute(job))
        finally:
            receipt_context.reset(token)
        self.tasks[job.consultation_id] = task
        task.add_done_callback(lambda completed: self._finished(job, completed))
        return task

    def _finished(self, job: RunJob, task: asyncio.Task[ConsultationSnapshot]) -> None:
        """消费任务异常，只记录安全类型和会诊上下文。"""
        if self.tasks.get(job.consultation_id) is task:
            self.tasks.pop(job.consultation_id, None)
        if not task.cancelled() and task.exception() is not None:
            self._log("run_publish_or_execution_failed", job.snapshot, task.exception())

    @staticmethod
    def _log(
        event: str, snapshot: ConsultationSnapshot | None, error: BaseException | None
    ) -> None:
        """日志只写关联 ID 与异常类型，不写患者原文或连接。"""
        logging.getLogger("six_channel").error(
            json.dumps(
                {
                    "event": event,
                    "request_id": None,
                    "consultation_id": str(snapshot.consultation_id) if snapshot else None,
                    "thread_id": snapshot.thread_id if snapshot else None,
                    "cause_type": type(error).__name__ if error else None,
                }
            )
        )

    async def _loop(self) -> None:
        """有界扫描可恢复记录，基础设施失败保留任务并在后续轮询重试。"""
        while not self.stopping:
            try:
                actions = [
                    GraphAction.model_validate(payload)
                    for payload in await self.store.scan("action")
                ]
                for action in sorted(actions, key=lambda item: (item.thread_id, item.target_turn)):
                    try:
                        await self.compensate(action)
                    except Exception as exc:
                        self._log("checkpoint_compensation_pending", action.snapshot, exc)
                for payload in await self.store.scan("job"):
                    job = RunJob.model_validate(payload)
                    if job.consultation_id not in self.tasks:
                        self.launch(job)
            except Exception as exc:
                self._log("runtime_recovery_unavailable", None, exc)
            await asyncio.sleep(self.poll_seconds)

    async def compensate(self, action: GraphAction) -> None:
        """同图互斥后重放幂等人工控制，成功标记与重放状态均可恢复。"""
        async with self.store.mutex("graph:" + action.thread_id):
            raw = await self.store.read("action", str(action.action_id))
            if raw is None or raw.get("completed"):
                return
            if self.service.workflow is not None:
                await self.service.workflow.reconcile(
                    action.snapshot,
                    HumanGraphCommand.model_validate(action.command),
                    str(action.action_id),
                )
            action.completed = True
            await self.store.write("action", str(action.action_id), action.model_dump(mode="json"))

    async def drain_actions(self, snapshot: ConsultationSnapshot) -> None:
        """执行同线程前先完成其待补偿动作，不让新推理跳过已提交审核。"""
        actions = [
            GraphAction.model_validate(payload)
            for payload in await self.store.scan("action", snapshot.consultation_id)
        ]
        for action in sorted(actions, key=lambda item: item.target_turn):
            # 调用方已持有图锁，直接使用幂等 checkpoint 标记防止重复人工动作。
            if self.service.workflow is not None:
                await self.service.workflow.reconcile(
                    action.snapshot,
                    HumanGraphCommand.model_validate(action.command),
                    str(action.action_id),
                )
            action.completed = True
            await self.store.write("action", str(action.action_id), action.model_dump(mode="json"))

    async def _heartbeat(self, job: RunJob, execution: asyncio.Task[ReasoningResponse]) -> None:
        """周期续约并检测跨进程取消；存储故障停止执行以免失去所有权后发布。"""
        try:
            while not execution.done():
                await asyncio.sleep(self.lease_seconds / 3)
                async with self.store.mutex("business:" + str(job.consultation_id)):
                    raw = await self.store.read("job", str(job.run_id))
                    current = RunJob.model_validate(raw)
                    if current.owner != self.owner or current.cancel_requested:
                        execution.cancel()
                        return
                    current.lease_until = next_lease(self.lease_seconds)
                    await self.store.write("job", str(job.run_id), current.model_dump(mode="json"))
        except Exception:
            execution.cancel()
            raise

    async def execute(self, supplied: RunJob) -> ConsultationSnapshot:
        """获得图锁后认领任务；过期旧任务仍受版本/所有者 fence 约束。"""
        async with self.semaphore, self.store.mutex("graph:" + supplied.snapshot.thread_id):
            async with self.store.mutex("business:" + str(supplied.consultation_id)):
                raw = await self.store.read("job", str(supplied.run_id))
                job = RunJob.model_validate(raw)
                current = await self.service.get_snapshot(job.consultation_id)
                if current.turn != job.turn or current.status != "running":
                    job.status = "cancelled" if current.status == "cancelled" else "succeeded"
                    await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
                    return current
                if (
                    job.lease_until
                    and job.lease_until > datetime.now(UTC)
                    and job.owner != self.owner
                ):
                    return current
                job.owner = self.owner
                job.lease_until = next_lease(self.lease_seconds)
                job.status = "running"
                job.attempts += 1
                await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
            if job.result is not None:
                return await self._publish(job)
            if job.attempts > self.max_attempts or job.cancel_requested:
                return await self._fail(job)

            async def sink(event: StreamEventName, payload: dict[str, object]) -> None:
                """实际执行事件绑定本轮任务，不带病例或模型正文。"""
                await self.store.emit(job.consultation_id, job.turn, event, payload)

            started = perf_counter()
            execution = asyncio.create_task(self._compute(job, sink))
            heartbeat = asyncio.create_task(self._heartbeat(job, execution))
            try:
                result = await execution
                if heartbeat.done():
                    await heartbeat
                # 先持久保存计算结果，业务发布失败后直接重试发布，避免重复模型费用。
                async with self.store.mutex("business:" + str(job.consultation_id)):
                    raw = await self.store.read("job", str(job.run_id))
                    latest = RunJob.model_validate(raw)
                    if latest.cancel_requested or latest.owner != self.owner:
                        return await self.service.get_snapshot(job.consultation_id)
                    job.result = result.model_dump(mode="json")
                    job.duration_ms = int((perf_counter() - started) * 1000)
                    await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
            except asyncio.CancelledError:
                if self.stopping:
                    # 优雅关闭使租约立即失效，下一实例接管；不把进程退出误认为医生取消。
                    async with self.store.mutex("business:" + str(job.consultation_id)):
                        raw = await self.store.read("job", str(job.run_id))
                        latest = RunJob.model_validate(raw)
                        if latest.owner == self.owner:
                            latest.lease_until = None
                            latest.owner = None
                            await self.store.write(
                                "job", str(job.run_id), latest.model_dump(mode="json")
                            )
                    raise
                await self._fail(job)
                raise
            except Exception:
                await self._fail(job)
                raise
            finally:
                execution.cancel()
                heartbeat.cancel()
                await asyncio.gather(execution, heartbeat, return_exceptions=True)
            # 此步骤在计算异常处理之外，发布失败必须留下已算结果供恢复。
            return await self._publish(job)

    async def _compute(
        self, job: RunJob, sink: Callable[[StreamEventName, dict[str, object]], Awaitable[None]]
    ) -> ReasoningResponse:
        """在上下文隔离与硬超时中执行，不把回调保存进图状态。"""
        async with progress_scope(sink), asyncio.timeout(self.timeout_seconds):
            await self.drain_actions(job.snapshot)
            if self.service.workflow is not None:
                recovered = await self.service.workflow.recover_result(job.snapshot)
                if recovered is not None:
                    await sink("recovery_required", {"recovered": True})
                    return recovered
            return await self.service.compute(job.snapshot, job.evidence)

    async def _publish(self, job: RunJob) -> ConsultationSnapshot:
        """锁内再次核对运行轮次，结果与终态任务原子发布。"""
        async with self.store.mutex("business:" + str(job.consultation_id)):
            current = await self.service.get_snapshot(job.consultation_id)
            raw = await self.store.read("job", str(job.run_id))
            latest = RunJob.model_validate(raw)
            if current.turn != job.turn or current.status != "running" or latest.cancel_requested:
                latest.status = "cancelled" if current.status == "cancelled" else "succeeded"
                await self.store.write("job", str(job.run_id), latest.model_dump(mode="json"))
                return current
            if latest.owner != self.owner:
                raise InvalidStateTransitionError("运行租约已被其他进程接管")
            job.status = "succeeded"
            job.lease_until = None
            job.snapshot.last_run_duration_ms = job.duration_ms
            return await self.service._save_result(
                job.snapshot,
                ReasoningResponse.model_validate(job.result),
                runtime_records=[("job", str(job.run_id), job.model_dump(mode="json"))],
            )

    async def _fail(self, job: RunJob) -> ConsultationSnapshot:
        """失败有明确终态；取消后的旧计算不能覆盖医生状态。"""
        async with self.store.mutex("business:" + str(job.consultation_id)):
            current = await self.service.get_snapshot(job.consultation_id)
            job.status = "cancelled" if current.status == "cancelled" else "failed"
            job.lease_until = None
            records = [("job", str(job.run_id), job.model_dump(mode="json"))]
            if current.turn != job.turn or current.status != "running":
                await self.store.write(*records[0])
                return current
            return await self.service.repository.save(
                current.model_copy(update={"status": "failed", "turn": current.turn + 1}),
                expected_turn=current.turn,
                runtime_records=records,
            )

    async def close(self) -> None:
        """退出先停恢复轮询，再释放在途任务租约供后续实例接管。"""
        self.stopping = True
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def idempotent(
        self,
        key: str | None,
        scope: str,
        payload: dict[str, object],
        action: Callable[[], Awaitable[ConsultationSnapshot]],
    ) -> ConsultationSnapshot:
        """同 key 同动作返回首次结果，异体请求 409；提交收据与业务写入同事务。"""
        if key is None:
            return await action()
        if re.fullmatch(r"[A-Za-z0-9_.:-]{8,128}", key) is None:
            raise InvalidBusinessRequestError("Idempotency-Key 必须为 8–128 位字母数字或 _ . : -")
        digest = hashlib.sha256(
            json.dumps(
                [scope, payload], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        # key 本身不进入数据库键/日志；S6 身份层需把可信主体加入 scope。
        record_key = hashlib.sha256(key.encode()).hexdigest()
        async with self.store.mutex("request:" + record_key):
            receipt = await self.store.read("receipt", record_key)
            if receipt is not None:
                if receipt.get("digest") != digest:
                    raise InvalidStateTransitionError("同一幂等键不能用于不同请求")
                if receipt.get("result") is not None:
                    return ConsultationSnapshot.model_validate(receipt["result"])
            await self.store.write("receipt", record_key, {"digest": digest, "result": None})
            token = receipt_context.set((record_key, digest))
            operation = operation_context.set(uuid5(NAMESPACE_URL, record_key))
            try:
                result = await action()
                await self.store.write(
                    "receipt",
                    record_key,
                    {"digest": digest, "result": result.model_dump(mode="json")},
                )
                return result
            finally:
                receipt_context.reset(token)
                operation_context.reset(operation)
