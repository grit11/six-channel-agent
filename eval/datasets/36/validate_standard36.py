"""六经标准36例 JSONL 结构与来源身份只读校验。

本脚本只验证技术一致性，不认定参考六经或方证临床正确。
用法：python validate_standard36.py /path/to/SixChannel_Standard36_Silver_v1.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

CHANNELS=("太阳","阳明","少阳","太阴","少阴","厥阴")

def validate(path: Path) -> dict:
    """检查固定病例数量、ID、标签、证据绑定、来源版本、重复与文本泄漏。"""
    data=[json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    errors=[]
    if len(data)!=36: errors.append(f"期望36例，实际{len(data)}例")
    counts=collections.Counter(x.get("group") for x in data)
    for c in CHANNELS:
        if counts[c]!=6:errors.append(f"{c} 数量不是6，实际{counts[c]}")
    ids=[x.get("case_id") for x in data]
    if len(ids)!=len(set(ids)):errors.append("case_id重复")
    texts=[x.get("text", "").strip() for x in data]
    if len(texts)!=len(set(texts)):errors.append("病例文本重复")
    for x in data:
        case_id=x.get("case_id","?")
        group=x.get("group")
        if x.get("difficulty")!="standard":errors.append(case_id+" 不是standard")
        if group not in CHANNELS:errors.append(case_id+" 六经类别非法")
        if x.get("reference_channels")!=[group]:errors.append(case_id+" 六经参考未完整或与分类不一致")
        formulas=x.get("reference_formulas") or []
        if len(formulas)!=1 or not isinstance(formulas[0],str) or not formulas[0].strip():
            errors.append(case_id+" 方证参考缺失/不唯一")
            continue
        if formulas[0] in x.get("text",""):errors.append(case_id+" 病例原文含方证参考名称")
        if len(x.get("text","").strip())<20:errors.append(case_id+" 病例信息过短")
        sources=x.get("source_refs") or []
        if not any(s.get("type")=="classic_clause" for s in sources):errors.append(case_id+" 无经典条文来源")
        if not any(s.get("type")=="formula_pattern" and s.get("formula_name")==formulas[0] for s in sources):
            errors.append(case_id+" 无当前方证对应来源")
        for s in sources:
            if s.get("evidence_id")!="1.2:"+s.get("knowledge_id",""):
                errors.append(case_id+" Evidence ID 与知识版本不一致")
        src_ids=[s.get("evidence_id") for s in sources]
        if len(src_ids)!=len(set(src_ids)):errors.append(case_id+" 重复 Evidence ID")
        if any("clause_no" not in s for s in sources if s.get("type")=="classic_clause"):
            errors.append(case_id+" 经典条文缺条号")
        diffs=x.get("priority_differential_rules") or []
        if any(("1.2:"+d) not in src_ids for d in diffs):errors.append(case_id+" 鉴别来源未绑定")
        if x.get("label_type")!="silver_reference_pending_expert_review":
            errors.append(case_id+" 标签未标明专家待审核")
        if x.get("expected_relation")!="single":errors.append(case_id+" 非标准单主路径")
    return {"total":len(data),"distribution":dict(counts),"distinct_formulas":len({f for x in data for f in x.get("reference_formulas",[])}),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"errors":errors,"passed":not errors}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    args=parser.parse_args()
    report=validate(args.dataset)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report["passed"] else 1)
