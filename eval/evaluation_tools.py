"""评测追踪与实验身份工具。

职责：在评测进程观察知识访问、隔离并发病例并计算公开路径的银标指标。
依赖现有 KnowledgeService 与 DTO；不修改医学算法、不把标签传入推理、不保存隐藏推理。
"""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Self, cast

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.schemas.reasoning_api import ReasoningResponse

_TRACE: ContextVar[list[dict[str, object]] | None] = ContextVar("evaluation_trace", default=None)
_STAGE: ContextVar[str] = ContextVar("evaluation_stage", default="clinical_parser")
SEARCH_METHODS = (
    "search_core_framework",
    "search_classic",
    "search_formula",
    "search_differential",
    "search_case",
)
EXACT_METHODS = (
    "get_evidence_by_ids",
    "get_clause_by_id",
    "get_formula_by_name",
    "get_differential_by_id",
)


def digest(value: object) -> str:
    """生成稳定 JSON 摘要；调用者不得传入密钥。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_digest(paths: Sequence[Path]) -> str:
    """按相对顺序和文件内容绑定代码/知识身份，变更后禁止混合续跑。"""
    checksum = sha256()
    for path in sorted(paths):
        checksum.update(path.name.encode("utf-8"))
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def experiment_identity(root: Path, settings: Settings) -> dict[str, object]:
    """记录影响推理的配置与内容身份；服务地址仅保留摘要，凭据永不参与报告。"""
    config = settings.model_dump(
        mode="json",
        exclude={
            "llm_api_key",
            "embedding_api_key",
            "reranker_api_key",
            "milvus_token",
            "database_url",
            "langgraph_database_url",
        },
    )
    # 身份仅覆盖模型/检索/评分策略，服务器和存储连接不属于质量实验参数。
    prefixes = ("llm_", "embedding_", "reranker_", "retrieval_", "fusion_", "topk_", "milvus_")
    selected = {key: value for key, value in config.items() if key.startswith(prefixes)}
    for key in list(selected):
        if key.endswith(("url", "uri", "path")):
            selected[key] = digest(selected[key])
    selected.update(
        {
            "clinical_parser_mode": settings.clinical_parser_mode,
            "reasoning_mode": settings.reasoning_mode,
            "scoring_policy": settings.scoring_policy.model_dump(mode="json"),
            "evidence_gap_policy": settings.evidence_gap_policy.model_dump(mode="json"),
            "max_supplement_rounds": settings.max_supplement_rounds,
            "protocol_version": settings.protocol_version,
        }
    )
    derived_path = settings.knowledge_directory.parent / "derived" / "fpeu_v12.jsonl"
    return {
        "configuration_sha256": digest(selected),
        "comparison_configuration_sha256": digest(
            {
                key: value
                for key, value in selected.items()
                if key
                not in {
                    "clinical_parser_mode",
                    "reasoning_mode",
                    "retrieval_mode",
                    "reranker_enabled",
                }
            }
        ),
        "code_sha256": file_digest(
            list((root / "backend/app").rglob("*.py")) + list((root / "eval").glob("*.py"))
        ),
        "prompt_sha256": file_digest(list((root / "backend/app/prompts").glob("*.md"))),
        "knowledge_sha256": file_digest(list(settings.knowledge_directory.glob("*.jsonl"))),
        "fpeu_branch_sha256": file_digest([derived_path]) if derived_path.exists() else "missing",
        "workflow_version": "six-stage-r1",
        "formula_branch_recall_top_n": settings.formula_branch_recall_top_n,
    }


def set_trace_stage(stage: str) -> None:
    """更新当前任务的公开阶段，子任务继承上下文但不跨病例共享。"""
    _STAGE.set(stage)


@contextmanager
def case_trace(calls: list[dict[str, object]]) -> Iterator[None]:
    """将观察记录绑定到当前病例任务，不依赖全局 case_id。"""
    token = _TRACE.set(calls)
    stage_token = _STAGE.set("clinical_parser")
    try:
        yield
    finally:
        _STAGE.reset(stage_token)
        _TRACE.reset(token)


class KnowledgeObserver:
    """只装饰评测实例的公开知识方法，调用结果原样返回并在退出时完整恢复。"""

    def __init__(self, knowledge: KnowledgeService) -> None:
        """保存服务实例；病例记录由 ContextVar 分配。"""
        self.knowledge = knowledge
        self.original: dict[str, tuple[bool, object]] = {}

    def __enter__(self) -> Self:
        """一次安装观察器，精确访问和检索访问分别记账。"""
        for name in (*SEARCH_METHODS, *EXACT_METHODS):
            self.original[name] = (name in vars(self.knowledge), vars(self.knowledge).get(name))
            method = getattr(self.knowledge, name)
            setattr(self.knowledge, name, self._wrap(name, method))
        return self

    def _wrap(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        """动态签名只在观察适配边界使用，保持生产方法参数与返回契约。"""

        @wraps(original)
        async def observed(*args: object, **kwargs: object) -> object:
            calls = _TRACE.get()
            if calls is None:
                return await original(*args, **kwargs)
            row: dict[str, object] = {
                "order": len(calls) + 1,
                "tool": name,
                "stage": _STAGE.get(),
                "kind": "search" if name in SEARCH_METHODS else "exact",
                "query_sha256": digest([args, kwargs]),
                "status": "running",
                "results": [],
            }
            calls.append(row)
            started = perf_counter()
            try:
                outcome = await original(*args, **kwargs)
                values = (
                    outcome if isinstance(outcome, list) else ([] if outcome is None else [outcome])
                )
                row["results"] = [
                    {
                        key: getattr(item, key)
                        for key in (
                            "evidence_id",
                            "knowledge_type",
                            "source_level",
                            "content",
                            "retrieval_method",
                            "retrieval_score",
                            "rerank_score",
                        )
                    }
                    for item in values
                ]
                row["status"] = "completed"
                return outcome
            except BaseException as exc:
                # 取消/超时也保留调用状态；异常正文可能带请求和密钥，不写入报告。
                row.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                row["duration_seconds"] = perf_counter() - started

        return observed

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """恢复原实例属性；原本未覆盖的方法删除影子属性以恢复类方法。"""
        for name, (existed, value) in self.original.items():
            if existed:
                setattr(self.knowledge, name, value)
            else:
                delattr(self.knowledge, name)


def observed_contexts(calls: list[dict[str, object]]) -> list[dict[str, object]]:
    """只聚合真实 search 返回，精确引用读取单列，不冒充召回。"""
    found: dict[str, dict[str, object]] = {}
    for call in calls:
        if call["kind"] == "search":
            for item in cast(list[dict[str, object]], call["results"]):
                found.setdefault(str(item["evidence_id"]), item)
    return list(found.values())


def domain_metrics(detail: dict[str, object], result: ReasoningResponse) -> dict[str, object]:
    """同一路径判断六经、关系和方证，不把多路径标签并集合成一次命中。"""
    paths = result.reasoning_paths[:3]
    channels = set(cast(list[str], detail.get("reference_channels") or []))
    formulas = set(cast(list[str], detail.get("reference_formulas") or []))
    relation = detail.get("expected_relation")
    matched = [p for p in paths if channels and channels.issubset(p.six_channels)]
    exact = [p for p in paths if channels and set(p.six_channels) == channels]
    forbidden = set(cast(list[str], detail.get("avoid_unjustified_formulas") or []))
    return {
        "label_type": "silver_reference_pending_expert_review",
        "channel_path_top1": bool(matched and matched[0] is paths[0]) if channels else None,
        "channel_path_top3": bool(matched) if channels else None,
        "channel_exact_path_top3": bool(exact) if channels else None,
        "formula_top3_hit": any(p.formula_pattern in formulas for p in paths) if formulas else None,
        "relation_exact_path_top3": (
            any(p.relationship == relation for p in exact) if channels and relation else None
        ),
        "joint_channel_formula_relation_top3": (
            any(p.formula_pattern in formulas and p.relationship == relation for p in exact)
            if channels and formulas and relation
            else None
        ),
        "forbidden_formula_absent": (
            all(p.formula_pattern not in forbidden for p in paths) if forbidden else None
        ),
    }
