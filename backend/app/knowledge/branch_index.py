"""方证派生分支索引：依赖五类只读来源，严格验证父子关系，不负责患者诊断。

机器派生只能标记待专家复核；不把聚合方证症状拆成伪条文，不产生硬禁忌。
"""

import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from backend.app.schemas.knowledge import FormulaBranchUnit, KnowledgeRecord


def derive_branches(records: list[KnowledgeRecord], version: str) -> list[FormulaBranchUnit]:
    """从父方证明确引用且经典明确关联的单条来源生成稳定待审分支。"""
    clauses = {
        str(item.metadata["clause_no"]): item
        for item in records
        if item.knowledge_type == "classic_clause"
    }
    result: list[FormulaBranchUnit] = []
    for parent in records:
        if (
            parent.knowledge_type != "formula_pattern"
            or parent.metadata.get("association_scope") != "classic_clause_based"
        ):
            continue
        name = str(parent.metadata["formula_name"])
        for clause_no in TypeAdapter(list[str]).validate_python(
            TypeAdapter(list[str]).validate_python(parent.metadata.get("source_clause_nos", []))
        ):
            clause = clauses.get(str(clause_no))
            if clause is None or name not in TypeAdapter(list[str]).validate_python(
                TypeAdapter(list[str]).validate_python(clause.metadata.get("related_formulas", []))
            ):
                continue
            positive = [
                term
                for term in TypeAdapter(list[str]).validate_python(
                    TypeAdapter(list[str]).validate_python(
                        clause.metadata.get("symptom_terms_source_grounded", [])
                    )
                )
                if isinstance(term, str) and term in clause.content
            ]
            negative = [
                term
                for term in TypeAdapter(list[str]).validate_python(
                    TypeAdapter(list[str]).validate_python(
                        clause.metadata.get("negative_findings_source_grounded", [])
                    )
                )
                if isinstance(term, str) and term in clause.content
            ]
            identity = f"{version}:{parent.knowledge_id}:{clause.knowledge_id}:0"
            result.append(
                FormulaBranchUnit(
                    branch_id="fpeu:" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    formula_evidence_id=f"{version}:{parent.knowledge_id}",
                    clause_evidence_ids=[f"{version}:{clause.knowledge_id}"],
                    formula_name=name,
                    canonical_channels=parent.channels,
                    positive_conditions=list(dict.fromkeys(positive)),
                    negative_conditions=list(dict.fromkeys(negative)),
                    review_status="source_verified_pending_expert",
                    provenance="five-source-derived/v1.2; single-clause quote; not expert approved",
                    knowledge_version=version,
                    source_quote=clause.content,
                    logical_scope=(
                        "unresolved_multiformula"
                        if len(
                            TypeAdapter(list[str]).validate_python(
                                TypeAdapter(list[str]).validate_python(
                                    clause.metadata.get("related_formulas", [])
                                )
                            )
                        )
                        > 1
                        else "single_clause"
                    ),
                )
            )
    return sorted(result, key=lambda item: item.branch_id)


def validate_branches(
    branches: list[FormulaBranchUnit], records: list[KnowledgeRecord], version: str
) -> list[FormulaBranchUnit]:
    """拒绝伪造 ID、跨版本、错配方名和非连续来源条件；待审不得变成硬排除。"""
    by_id = {f"{version}:{item.knowledge_id}": item for item in records}
    seen: set[str] = set()
    for branch in branches:
        parent = by_id.get(branch.formula_evidence_id)
        if (
            branch.branch_id in seen
            or branch.knowledge_version != version
            or parent is None
            or parent.knowledge_type != "formula_pattern"
            or parent.metadata.get("association_scope") != "classic_clause_based"
            or parent.metadata.get("formula_name") != branch.formula_name
            or branch.canonical_channels != parent.channels
        ):
            raise ValueError("方证派生分支父来源或版本无效")
        if len(branch.clause_evidence_ids) != 1:
            raise ValueError("当前派生索引仅支持单条来源分支")
        identity = (
            f"{version}:{parent.knowledge_id}:{branch.clause_evidence_ids[0].split(':', 1)[1]}:0"
        )
        if branch.branch_id != "fpeu:" + hashlib.sha256(identity.encode()).hexdigest()[:24]:
            raise ValueError("派生分支身份不匹配来源")
        seen.add(branch.branch_id)
        clauses = [by_id.get(ref) for ref in branch.clause_evidence_ids]
        if any(item is None or item.knowledge_type != "classic_clause" for item in clauses):
            raise ValueError("方证派生分支经典引用无效")
        for item in clauses:
            assert item is not None
            if str(item.metadata["clause_no"]) not in TypeAdapter(list[str]).validate_python(
                parent.metadata.get("source_clause_nos", [])
            ) or branch.formula_name not in TypeAdapter(list[str]).validate_python(
                item.metadata.get("related_formulas", [])
            ):
                raise ValueError("方证分支条文未与父方证明确关联")
        texts = [item.content for item in clauses if item is not None]
        if branch.source_quote not in texts or any(
            not condition or condition not in branch.source_quote
            for condition in branch.positive_conditions + branch.negative_conditions
        ):
            raise ValueError("方证分支条件缺少连续原文")
    return branches


def load_branch_index(
    path: Path, records: list[KnowledgeRecord], version: str
) -> list[FormulaBranchUnit]:
    """读取严格派生索引；未发布时从同一只读来源派生，不使用模型补齐。"""
    if not path.exists():
        return validate_branches(derive_branches(records, version), records, version)
    branches = [
        FormulaBranchUnit.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return validate_branches(branches, records, version)


def branch_digest(branches: list[FormulaBranchUnit]) -> str:
    """给分支内容独立版本签名，使重算与知识缓存不会沿用过期条件。"""
    return hashlib.sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in branches],
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
