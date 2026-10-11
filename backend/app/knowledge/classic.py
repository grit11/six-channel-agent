"""知识条目加载和标准化，仅供 KnowledgeService 内部调用，不进行患者推理。"""

import json
from collections import Counter
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

from backend.app.core.exceptions import KnowledgeRetrievalError
from backend.app.schemas.knowledge import KnowledgeRecord, KnowledgeType

SOURCE_LEVELS = {
    "classic_clause": "L1",
    "formula_pattern": "L3",
    "differential": "L4",
    "medical_case": "L5",
    "six_channel_framework": "L0",
}
REQUIRED_FIELDS = {
    "classic_clause": ["classic_text", "clause_no", "semantic_six_channels"],
    "formula_pattern": [
        "formula_name",
        "association_scope",
        "canonical_six_channels",
        "case_derived_six_channels",
        "key_symptoms_source_grounded",
    ],
    "differential": ["candidates", "key_dimensions", "decision_rules"],
    "medical_case": ["symptoms_source_grounded", "tongue", "pulse"],
    "six_channel_framework": ["content", "title"],
}


def _strings(value: object) -> list[str]:
    """将来源字符串或字符串列表标准化，其他类型显式拒绝。"""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast(list[str], value)
    raise ValueError("来源字段必须是字符串或字符串列表")


def _validate_source_fields(raw: dict[str, object], kind: str) -> None:
    """校验原始字段类型，不将 None、列表或对象隐式转成伪原文。"""
    for field in ("id", "knowledge_type", "text_for_embedding_v2"):
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise ValueError(f"来源字段 {field} 必须是非空字符串")
    if kind == "classic_clause":
        TypeAdapter(list[str]).validate_python(raw["semantic_six_channels"])
        if not isinstance(raw["classic_text"], str) or not raw["classic_text"]:
            raise ValueError("classic_text 必须是非空原文")
    elif kind == "formula_pattern":
        if raw["association_scope"] not in ("classic_clause_based", "case_derived_historical"):
            raise ValueError("未知方证来源范围")
        for field in (
            "canonical_six_channels",
            "case_derived_six_channels",
            "key_symptoms_source_grounded",
        ):
            TypeAdapter(list[str]).validate_python(raw[field])
    elif kind == "differential":
        TypeAdapter(list[str]).validate_python(raw["candidates"])
        TypeAdapter(dict[str, str]).validate_python(raw["key_dimensions"])
        TypeAdapter(list[str]).validate_python(raw["decision_rules"])
    elif kind == "medical_case":
        TypeAdapter(list[str]).validate_python(raw["symptoms_source_grounded"])


def normalize_record(raw: dict[str, object], version: str) -> KnowledgeRecord:
    """优先采用 grounded/v2 字段；不以旧字段补回已清空的审核结果。"""
    kind = str(raw.get("knowledge_type", ""))
    if kind not in SOURCE_LEVELS:
        raise ValueError("未知知识类型")
    for field in REQUIRED_FIELDS[kind]:
        if field not in raw:
            raise ValueError(f"知识记录缺少 {field}")
    _validate_source_fields(raw, kind)
    embedding = raw.get("text_for_embedding_v2")
    if not isinstance(embedding, str) or not embedding.strip():
        raise ValueError("知识记录缺少有效 v2 检索文本")
    channels: list[str] = []
    if kind == "classic_clause":
        channels = _strings(raw["semantic_six_channels"])
    elif kind == "formula_pattern":
        # 医案派生标签不能反向写入经典归属；按本条来源范围选择过滤标签。
        key = (
            "canonical_six_channels"
            if raw["association_scope"] == "classic_clause_based"
            else "case_derived_six_channels"
        )
        channels = _strings(raw[key])
    elif kind == "medical_case":
        channels = _strings(raw.get("six_channels", []))
    source = "；".join(_strings(raw.get("source_local", "")))
    if not source:
        # 部分条文仅有在线校核来源，不虚构本地文档出处。
        source = "；".join(
            _strings(raw.get("source_work", "")) + _strings(raw.get("source_online_primary", ""))
        )
    content = str(raw["classic_text"]) if kind == "classic_clause" else embedding
    if kind == "six_channel_framework":
        content = str(raw["content"])
    return KnowledgeRecord(
        knowledge_id=str(raw.get("id", "")),
        knowledge_type=cast(KnowledgeType, kind),
        knowledge_version=version,
        source=source,
        source_level=SOURCE_LEVELS[kind],
        content=content,
        embedding_text=embedding,
        channels=channels,
        metadata=raw,
    )


def load_records(directory: Path, version: str) -> list[KnowledgeRecord]:
    """加载五类 JSONL 并校验 ID、字段及类型；任一文件失败则整体失败。"""
    records: list[KnowledgeRecord] = []
    try:
        paths = sorted(directory.glob("*.jsonl"))
        if len(paths) != 5:
            raise ValueError("运行时主库必须包含五个 JSONL 文件")
        for path in paths:
            for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{path.name}:{line_no} 必须是 JSON 对象")
                records.append(normalize_record(raw, version))
        counts = Counter(record.knowledge_type for record in records)
        if set(counts) != set(SOURCE_LEVELS):
            raise ValueError("五类知识必须均存在")
        if len({record.knowledge_id for record in records}) != len(records):
            raise ValueError("来源知识 ID 重复")
        clauses = [r.metadata["clause_no"] for r in records if r.knowledge_type == "classic_clause"]
        formulas = [
            r.metadata["formula_name"] for r in records if r.knowledge_type == "formula_pattern"
        ]
        if len(set(map(str, clauses))) != len(clauses) or len(set(map(str, formulas))) != len(
            formulas
        ):
            raise ValueError("条文号或方名重复，无法进行唯一精确查询")
    except Exception as exc:
        raise KnowledgeRetrievalError("知识库加载校验失败，请检查来源文件和字段") from exc
    return records
