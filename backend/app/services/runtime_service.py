"""V2 可靠运行协调：任务租约、图互斥、结果发布和跨库恢复。

业务意图与 PostgreSQL 快照同事务保存；Checkpoint 先写成功而发布失败时可重试读取，
不宣称不同数据库间有原子事务。Provider 失败明确失败，不进行医学规则回退。
"""

import asyncio
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID, uuid4

from backend.app.core.exceptions import ApplicationError
from backend.app.observability.progress import progress_scope
from backend.app.persistence.runtime import next_lease, receipt_context
from backend.app.schemas.consultation import ConsultationSnapshot
from backend.app.schemas.events import StreamEventName
from backend.app.schemas.review import DoctorReviewRecord, DoctorReviewRequest
from backend.app.schemas.runtime import RunJob
from backend.app.services.consultation_service import ConsultationService


class ReliableRuntime:
    """应用级后台调度，图锁在整个计算和发布期间阻止并发 checkpoint 写入。"""

    def __init__(
        self,
        service: ConsultationService,
        *,
        lease_seconds: float = 30,
        poll_seconds: float = 1,
        timeout_seconds: float = 300,
        max_attempts: int = 3,
        max_concurrent: int = 4,
    ) -> None:
        """注入运行预算与共享存储。"""
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
        """启动持久任务恢复扫描，未提交的任务不执行。"""
        self.worker = asyncio.create_task(self._loop())

    def launch(self, job: RunJob) -> asyncio.Task[ConsultationSnapshot]:
        """加速本地任务，清除请求收据上下文防止后台覆盖创建响应。"""
        if job.run_id in self.tasks:
            return self.tasks[job.run_id]
        token = receipt_context.set(None)
        try:
            task = asyncio.create_task(self.execute(job))
        finally:
            receipt_context.reset(token)
        self.tasks[job.run_id] = task
        task.add_done_callback(lambda result: self._finished(job.run_id, result))
        return task

    def _finished(self, identifier: UUID, task: asyncio.Task[ConsultationSnapshot]) -> None:
        self.tasks.pop(identifier, None)
        if not task.cancelled() and task.exception():
            self._log(identifier, task.exception())

    @staticmethod
    def _log(identifier: UUID | None, error: BaseException | None) -> None:
        logging.getLogger("six_channel").error(
            "v2_runtime_error consultation_id=%s thread_id=clinical-v2:%s "
            "request_id=%s cause_type=%s",
            identifier,
            identifier,
            None,
            type(error).__name__,
        )

    async def _loop(self) -> None:
        while not self.stopping:
            try:
                for raw in await self.store.scan("job"):
                    job = RunJob.model_validate(raw)
                    if job.run_id not in self.tasks:
                        self.launch(job)
            except Exception as exc:
                self._log(None, exc)
            await asyncio.sleep(self.poll_seconds)

    async def _heartbeat(self, job: RunJob, execution: asyncio.Task[dict[str, Any]]) -> None:
        try:
            while not execution.done():
                await asyncio.sleep(self.lease_seconds / 3)
                raw = await self.store.read("job", str(job.run_id))
                current = RunJob.model_validate(raw)
                if current.owner != self.owner or current.status != "running":
                    execution.cancel()
                    return
                current.lease_until = next_lease(self.lease_seconds)
                await self.store.write("job", str(job.run_id), current.model_dump(mode="json"))
        except Exception:
            execution.cancel()
            raise

    async def execute(self, supplied: RunJob) -> ConsultationSnapshot:
        """认领已提交意图；崩溃恢复读取图节点进度，不重复已消费人工动作。"""
        async with (
            self.semaphore,
            self.store.mutex("graph:clinical-v2:" + str(supplied.consultation_id)),
        ):
            raw = await self.store.read("job", str(supplied.run_id))
            job = RunJob.model_validate(raw)
            current = await self.service.get_snapshot(job.consultation_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return current
            if current.turn != job.turn or current.status != "running":
                job.status = "cancelled"
                await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
                return current
            if job.lease_until and job.lease_until > datetime.now(UTC) and job.owner != self.owner:
                return current
            job.owner = self.owner
            job.status = "running"
            job.lease_until = next_lease(self.lease_seconds)
            job.attempts += 1
            await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
            if job.result is not None:
                return await self._publish(job)
            if job.attempts > self.max_attempts:
                job.result = {"status": "failed", "failure_code": "RECOVERY_EXHAUSTED"}
                return await self._publish(job)

            async def sink(event: StreamEventName, payload: dict[str, object]) -> None:
                """事件只携带阶段、计数、usage 和延迟，不包含患者原文。"""
                await self.store.emit(job.consultation_id, job.turn, event, payload)

            async def compute() -> dict[str, Any]:
                assert self.service.workflow is not None
                async with progress_scope(sink), asyncio.timeout(self.timeout_seconds):
                    return await self.service.workflow.execute(
                        str(job.consultation_id),
                        current.raw_input,
                        job.command,
                    )

            started = perf_counter()
            execution = asyncio.create_task(compute())
            heartbeat = asyncio.create_task(self._heartbeat(job, execution))
            try:
                job.result = await execution
                if heartbeat.done():
                    await heartbeat
            except asyncio.CancelledError:
                job.owner = None
                job.lease_until = None
                await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
                raise
            except Exception as exc:
                self._log(job.consultation_id, exc)
                job.result = {
                    "status": "failed",
                    "failure_code": exc.code if isinstance(exc, ApplicationError) else "RUN_FAILED",
                }
                await self.store.emit(
                    job.consultation_id,
                    job.turn,
                    "stage_failed",
                    {"failure_code": job.result["failure_code"]},
                )
            finally:
                execution.cancel()
                heartbeat.cancel()
                await asyncio.gather(execution, heartbeat, return_exceptions=True)
            job.duration_ms = int((perf_counter() - started) * 1000)
            # 计算和业务发布分两步：发布错误留存结果供接管，不能重复模型调用。
            await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
            return await self._publish(job)

    async def _publish(self, job: RunJob) -> ConsultationSnapshot:
        async with self.store.mutex("business:" + str(job.consultation_id)):
            current = await self.service.get_snapshot(job.consultation_id)
            if current.turn != job.turn or current.status != "running":
                job.status = "cancelled"
                job.lease_until = None
                await self.store.write("job", str(job.run_id), job.model_dump(mode="json"))
                return current
            state = job.result or {}
            status = str(state.get("status", "failed"))
            changes: dict[str, Any] = {
                "status": status,
                "turn": current.turn + 1,
                "revision": state.get("revision", current.revision),
                "paused_at": state.get("paused_at"),
                "failure_code": state.get("failure_code"),
                "last_run_duration_ms": job.duration_ms,
            }
            for key in ("clinical_evidence", "localization", "retrieval", "comparison"):
                if key in state:
                    changes[key] = state[key]
            stages = [
                s
                for s, key in (
                    ("s1_extract", "clinical_evidence"),
                    ("s2_localize", "localization"),
                    ("s3_retrieve", "retrieval"),
                    ("s4_compare", "comparison"),
                )
                if state.get(key) is not None
            ]
            changes["completed_stages"] = stages + (
                ["s5_doctor_review"] if status == "completed" else []
            )
            if status in {"completed", "undetermined", "cancelled", "failed"}:
                changes["completed_at"] = datetime.now(UTC)
            if status == "completed" and job.command and job.command.review:
                review = DoctorReviewRequest.model_validate(job.command.review)
                record = DoctorReviewRecord(
                    review_id=uuid4(),
                    consultation_id=current.consultation_id,
                    action=review.action,
                    selected_path_id=review.selected_path_id,
                    ordered_path_ids=review.ordered_path_ids,
                    note=review.note,
                    modified_six_channels=review.modified_six_channels,
                    modified_syndrome=review.modified_syndrome,
                    modified_formula_pattern=review.modified_formula_pattern,
                    agent_path_ids=(
                        [p.path_id for p in current.comparison.ordered_paths]
                        if current.comparison
                        else []
                    ),
                    doctor_adopted=review.action in {"accept", "modify"},
                    created_at=datetime.now(UTC),
                )
                changes["doctor_review"] = record.model_dump(mode="json")
                changes["review_history"] = [
                    *current.review_history,
                    record.model_dump(mode="json"),
                ]
            saved = ConsultationSnapshot.model_validate(
                {
                    **current.model_dump(mode="json"),
                    **changes,
                }
            )
            job.status = "failed" if status == "failed" else "succeeded"
            job.lease_until = None
            records = [("job", str(job.run_id), job.model_dump(mode="json"))]
            result = await self.service.repository.save(
                saved,
                expected_turn=current.turn,
                runtime_records=records,
            )
            if status in {"needs_evidence", "awaiting_doctor_review"}:
                await self.store.emit(
                    result.consultation_id,
                    result.turn,
                    "clarification_requested" if status == "needs_evidence" else "review_requested",
                    {"paused_at": result.paused_at, "revision": result.revision},
                )
            return result

    async def close(self) -> None:
        """关闭时释放租约；保留已提交意图供下一实例恢复。"""
        self.stopping = True
        if self.worker:
            self.worker.cancel()
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(
            *list(self.tasks.values()),
            *([self.worker] if self.worker else []),
            return_exceptions=True,
        )
