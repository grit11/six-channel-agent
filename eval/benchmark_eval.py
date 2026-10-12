"""48 条基准的结果汇总工具。

职责：读取现有 system_eval 公开报告，计算银标 Top-K 覆盖并输出分组和成对对照。
依赖 EvaluationCase 与 ReasoningResponse；不运行医学算法、不修改标签、不把标签送入模型。
"""

# 直接执行 eval 脚本前需加入仓库根目录，随后才可导入 backend 和 eval。

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.schemas.reasoning_api import ReasoningResponse
from eval.evaluation_tools import domain_metrics
from eval.system_eval import load_cases, path_label_metrics


def load_details(path: Path) -> dict[str, dict[str, object]]:
    """加载详细银标及分组，重复编号明确拒绝，保留未知标注。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    records = {row["case_id"]: row for row in rows}
    if not rows or len(records) != len(rows):
        raise ValueError("详细集不能为空或包含重复 ID")
    return records


def coverage(groups: list[list[str]], labels: Sequence[str] | None) -> dict[str, object]:
    """只对非空参考标签计分；无标注不强制解释为负例或错误。"""
    return path_label_metrics(groups, labels) if labels else {"status": "unlabelled"}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """分别记录实际样本数与各指标分母，不用总病例数稀释未知标签。"""
    result: dict[str, object] = {"cases": len(rows)}
    for kind in ("channel", "formula"):
        labelled = [cast(dict[str, object], row[kind]) for row in rows]
        labelled = [item for item in labelled if "top1_recall" in item]
        result[kind] = {
            "labelled": len(labelled),
            "unlabelled": sum(
                cast(dict[str, object], row[kind]).get("status") == "unlabelled" for row in rows
            ),
            **(
                {
                    "execution_failed": sum(
                        cast(dict[str, object], row[kind]).get("status") == "execution_failed"
                        for row in rows
                    )
                }
                if any(
                    cast(dict[str, object], row[kind]).get("status") == "execution_failed"
                    for row in rows
                )
                else {}
            ),
            "top1_recall": (
                mean(cast(float, item["top1_recall"]) for item in labelled) if labelled else None
            ),
            "top3_recall": (
                mean(cast(float, item["top3_recall"]) for item in labelled) if labelled else None
            ),
        }
    domain_names = {
        name for row in rows for name in cast(dict[str, object], row.get("domain_metrics", {}))
    }
    result["domain"] = {}
    for name in sorted(domain_names):
        values = [cast(dict[str, object], row.get("domain_metrics", {})).get(name) for row in rows]
        labelled_values = [float(value) for value in values if isinstance(value, bool)]
        cast(dict[str, object], result["domain"])[name] = {
            "mean": mean(labelled_values) if labelled_values else None,
            "scored": len(labelled_values),
            "unavailable": len(values) - len(labelled_values),
        }
    return result


def build_report(
    report_path: Path, folder: Path, *, allow_partial: bool = False
) -> dict[str, object]:
    """严格对应输入和输出身份，生成逐案、分层与八组成对公开对照。"""
    cases = load_cases(folder / "cases_eval_compatible.jsonl")
    details = load_details(folder / "cases_detailed.jsonl")
    original = json.loads(report_path.read_text(encoding="utf-8"))
    records = original["cases"]
    if original.get("status", "completed") != "completed":
        raise ValueError("源报告尚未完成")
    if original.get("expected_sample_count", len(cases)) != len(cases):
        raise ValueError("源报告的期望分母与输入集不一致")
    if allow_partial:
        if original.get("status") != "completed" or original.get("complete_benchmark") is not False:
            raise ValueError("部分报告须已完成且明确标记非完整评测")
        ids = {row["case_id"] for row in records}
        if not ids or len(ids) != len(records) or not ids.issubset(details):
            raise ValueError("部分报告病例身份非法")
        cases = [case for case in cases if case.case_id in ids]
    if len(records) != len(cases) or {row["case_id"] for row in records} != {
        case.case_id for case in cases
    }:
        raise ValueError("报告必须完整包含输入集，禁止部分运行冒充全量")
    results = {row["case_id"]: row for row in records}
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        record = results[case.case_id]
        if record.get("input_text") != case.text:
            raise ValueError(f"{case.case_id}: 报告输入与数据集不一致")
        if record.get("result") is None:
            row = {
                "case_id": case.case_id,
                "group": details[case.case_id]["group"],
                "split": details[case.case_id]["split"],
                "channel": {"status": "execution_failed"},
                "formula": {"status": "execution_failed"},
                "engineering_errors": record["errors"],
                "top3_reference_misses": [],
                "top3": [],
                "runtime_seconds": record.get("runtime_seconds"),
            }
            rows.append(row)
            grouped[f"split:{row['split']}"].append(row)
            grouped[f"group:{row['group']}"].append(row)
            continue
        result = ReasoningResponse.model_validate(record["result"])
        paths = result.reasoning_paths[:3]
        detail = details[case.case_id]
        if detail["text"] != case.text:
            raise ValueError("详细集与兼容集文本不一致")
        channel = coverage([path.six_channels for path in paths], case.expected_channels)
        formula = coverage(
            [[path.formula_pattern] if path.formula_pattern else [] for path in paths],
            case.expected_formulas,
        )
        misses = [
            kind
            for kind, metric in (("channel", channel), ("formula", formula))
            if "top3_recall" in metric and cast(float, metric["top3_recall"]) < 1
        ]
        row = {
            "case_id": case.case_id,
            "group": detail["group"],
            "split": detail["split"],
            "channel": channel,
            "formula": formula,
            "domain_metrics": domain_metrics(detail, result),
            "unknown_preserved": (
                all(
                    any(
                        f.key == key and f.polarity == "unknown"
                        for f in result.clinical_evidence.current_findings()
                    )
                    for key in case.expected_unknown
                )
                if case.expected_unknown
                else None
            ),
            "conflicts_isolated": (
                (
                    set(case.conflicted_keys).issubset(
                        {item.key for item in result.clinical_evidence.contradictions}
                    )
                    and not set(case.conflicted_keys).intersection(
                        item.key for item in result.clinical_evidence.current_findings()
                    )
                )
                if case.conflicted_keys
                else None
            ),
            "top3_reference_misses": misses,
            "engineering_errors": record["errors"],
            "runtime_seconds": record.get("runtime_seconds"),
            "top3": [path.model_dump(mode="json") for path in paths],
            "questions": [gap.model_dump(mode="json") for gap in result.evidence_gaps],
            "contradictions": [
                item.model_dump(mode="json") for item in result.clinical_evidence.contradictions
            ],
            "unknown": [
                item.key
                for item in result.clinical_evidence.current_findings()
                if item.polarity == "unknown"
            ],
        }
        rows.append(row)
        grouped[f"split:{detail['split']}"].append(row)
        grouped[f"group:{detail['group']}"].append(row)
    indexed = {str(row["case_id"]): row for row in rows}
    with (folder / "challenge_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        pairs = [
            {
                "case_a": pair["案例A"],
                "case_b": pair["案例B"],
                "discriminator": pair["必须区分的差异"],
                "source_rule_id": pair["鉴别规则ID"],
                "result_a": indexed[pair["案例A"]],
                "result_b": indexed[pair["案例B"]],
            }
            for pair in csv.DictReader(stream)
            if pair["案例A"] in indexed and pair["案例B"] in indexed
        ]
    return {
        "version": "benchmark48-summary/v1.0",
        "mode": original["mode"],
        "label_type": "silver_reference_pending_expert_review",
        "scope": "同源合成集的候选覆盖，不代表临床准确率；runtime 缺失保持 null",
        "engineering_passed": original["passed"],
        "execution_failed": sum(row.get("result") is None for row in records),
        "expected_sample_count": original.get("expected_sample_count", len(details)),
        "complete_benchmark": original.get("complete_benchmark", len(records) == len(details)),
        "aggregate": summarize(rows),
        "groups": {name: summarize(items) for name, items in grouped.items()},
        "cases": rows,
        "challenge_pairs": pairs,
    }


def save_report(output: Path, report: dict[str, object]) -> None:
    """保存可复算 JSON 与可阅读摘要，不更改原始基线快照。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Benchmark48 评测摘要",
        "",
        str(report["scope"]),
        f"模式：{report['mode']}；标签：{report['label_type']}",
        "",
        "| 分组 | 病例 | 六经标注数 | 六经 Top1/Top3 | 方证标注数 | 方证 Top1/Top3 |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    groups = {"all": report["aggregate"], **cast(dict[str, object], report["groups"])}
    for name, value in groups.items():
        item = cast(dict[str, object], value)
        channel = cast(dict[str, object], item["channel"])
        formula = cast(dict[str, object], item["formula"])
        lines.append(
            f"| {name} | {item['cases']} | {channel['labelled']} | "
            f"{channel['top1_recall']} / {channel['top3_recall']} | {formula['labelled']} | "
            f"{formula['top1_recall']} / {formula['top3_recall']} |"
        )
    lines += [
        "",
        "## 单路径领域指标",
        "",
        "| 指标 | 均值 | 已评分 | 不可评分 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metric in cast(
        dict[str, dict[str, object]], cast(dict[str, object], report["aggregate"])["domain"]
    ).items():
        lines.append(
            f"| {name} | {metric['mean']} | {metric['scored']} | {metric['unavailable']} |"
        )
    lines += ["", "## 逐案失败", ""]
    for row in cast(list[dict[str, object]], report["cases"]):
        if row["engineering_errors"] or row["top3_reference_misses"]:
            lines.append(
                f"- {row['case_id']}: 工程={row['engineering_errors']}；"
                f"Top3 未完整覆盖={row['top3_reference_misses']}"
            )
    lines += ["", "## 八组鉴别对照", ""]
    for pair in cast(list[dict[str, object]], report["challenge_pairs"]):
        names = []
        for side in ("result_a", "result_b"):
            row = cast(dict[str, object], pair[side])
            paths = cast(list[dict[str, object]], row["top3"])
            names.append(" / ".join(str(path["formula_pattern"]) for path in paths))
        lines.append(
            f"- {pair['case_a']} [{names[0]}] → {pair['case_b']} [{names[1]}]；"
            f"{pair['discriminator']}；来源 {pair['source_rule_id']}"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """读取指定基线，输出摘要；参考覆盖不足记录结果，不伪装工程异常。"""
    parser = argparse.ArgumentParser(description="48 条银标评测分组与鉴别对照")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--folder", type=Path, default=ROOT / "eval/datasets/benchmark48")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark48/summary.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = build_report(args.report, args.folder, allow_partial=args.allow_partial)
    save_report(args.output, report)
    print(json.dumps(report["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
