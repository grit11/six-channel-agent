"""同一数据集的消融报告对比。

职责：核验输入、代码、提示和知识身份，按共同成功病例比较领域指标与RAGAs。
依赖现有JSON报告；不运行推理、不把失败或未标注补零、不比较不同评分口径。
"""

# 直接执行入口必须先配置仓库导入路径。

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.schemas.reasoning_api import ReasoningResponse
from eval.benchmark_eval import load_details
from eval.evaluation_tools import domain_metrics


def compare_reports(
    reports: list[Path], folder: Path, ragas_reports: list[Path] | None = None
) -> dict[str, object]:
    """同时报告各组失败和配对有效分母，禁止不同病例集合直接作均值比较。"""
    if len(reports) < 2:
        raise ValueError("至少提供两组系统报告")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    details = load_details(folder / "cases_detailed.jsonl")
    indexed = []
    for source in sources:
        if source.get("status") != "completed":
            raise ValueError("只能比较已完成报告")
        rows = source["cases"]
        records_by_id = {row["case_id"]: row for row in rows}
        if len(records_by_id) != len(rows) or not records_by_id:
            raise ValueError("病例身份为空或重复")
        for identifier, row in records_by_id.items():
            if identifier not in details or row["input_text"] != details[identifier]["text"]:
                raise ValueError("报告原文与数据集不一致")
        indexed.append(records_by_id)
    first_ids = set(indexed[0])
    if any(set(rows) != first_ids for rows in indexed):
        raise ValueError("各组病例集合不一致")
    for source in sources[1:]:
        if source.get("input_digest") != sources[0].get("input_digest"):
            raise ValueError("输入标签摘要不同")
        for key in (
            "code_sha256",
            "prompt_sha256",
            "knowledge_sha256",
            "comparison_configuration_sha256",
        ):
            initial = sources[0]["runtime"].get("experiment_identity", {}).get(key)
            actual = source["runtime"].get("experiment_identity", {}).get(key)
            if not initial or initial != actual:
                raise ValueError("缺少或不一致的代码、提示或知识身份")
    paired = sorted(
        set.intersection(
            *[
                {
                    identifier
                    for identifier, row in rows.items()
                    if row.get("passed") is True and row.get("result") is not None
                }
                for rows in indexed
            ]
        )
    )
    groups: list[dict[str, object]] = []
    for path, source, rows in zip(reports, sources, indexed, strict=True):
        cases = []
        for identifier in paired:
            result = ReasoningResponse.model_validate(rows[identifier]["result"])
            cases.append(
                {"case_id": identifier, "metrics": domain_metrics(details[identifier], result)}
            )
        metric_names = {name for row in cases for name in row["metrics"]}
        metrics = {}
        for name in sorted(metric_names):
            values = [
                row["metrics"][name] for row in cases if isinstance(row["metrics"][name], bool)
            ]
            if values:
                metrics[name] = {"mean": mean(values), "n": len(values)}
        groups.append(
            {
                "report": str(path),
                "ablation": source["runtime"].get("ablation"),
                "attempted": len(rows),
                "execution_failed": sum(row.get("result") is None for row in rows.values()),
                "engineering_failed": sum(row.get("passed") is not True for row in rows.values()),
                "paired_metrics": metrics,
                "paired_cases": cases,
            }
        )
    output: dict[str, object] = {
        "version": "evaluation-comparison/v1.0",
        "paired_case_ids": paired,
        "paired_count": len(paired),
        "groups": groups,
        "complete_benchmark": all(source.get("complete_benchmark") is True for source in sources),
        "notice": "共同成功样本的银标配对比较；各组失败单列，不代表临床质量。",
    }
    if ragas_reports:
        if len(ragas_reports) != len(reports):
            raise ValueError("RAGAs报告数量与系统报告须一致且按同序提供")
        grades = [json.loads(path.read_text(encoding="utf-8")) for path in ragas_reports]
        from hashlib import sha256

        for path, grade in zip(reports, grades, strict=True):
            if grade["source_report"].get("sha256") != sha256(path.read_bytes()).hexdigest():
                raise ValueError("RAGAs报告与系统源报告不匹配")
        policy = grades[0]["source_report"].get("judge_configuration")
        if not policy or any(
            grade["source_report"].get("judge_configuration") != policy
            or grade.get("judge_model") != grades[0].get("judge_model")
            or grade.get("ragas_version") != grades[0].get("ragas_version")
            or grade.get("silver_reference_enabled") != grades[0].get("silver_reference_enabled")
            for grade in grades
        ):
            raise ValueError("裁判或上下文评分口径不同")
        grade_maps = [{row["case_id"]: row for row in grade["cases"]} for grade in grades]
        names = set.intersection(
            *[{name for row in rows.values() for name in row["metrics"]} for rows in grade_maps]
        )
        semantic = {}
        for name in sorted(names):
            ids = [
                identifier
                for identifier in paired
                if all(
                    rows.get(identifier, {}).get("metrics", {}).get(name, {}).get("status")
                    == "scored"
                    for rows in grade_maps
                )
            ]
            semantic[name] = {
                "paired_ids": ids,
                "groups": [
                    {
                        "n": len(ids),
                        "mean": (
                            mean(rows[identifier]["metrics"][name]["score"] for identifier in ids)
                            if ids
                            else None
                        ),
                    }
                    for rows in grade_maps
                ],
            }
        output["ragas_paired_metrics"] = semantic
    return output


def main() -> int:
    """读取显式报告列表，输出可复算JSON与配对指标表。"""
    parser = argparse.ArgumentParser(description="消融实验共同样本对比")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--ragas-reports", type=Path, nargs="+")
    parser.add_argument("--folder", type=Path, default=ROOT / "eval/datasets/benchmark48")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_reports(args.reports, args.folder, args.ragas_reports)
    from eval.ragas_eval import export

    export(args.output, report)
    groups = cast(list[dict[str, object]], report["groups"])
    lines = [
        "# 消融配对对比",
        "",
        str(report["notice"]),
        f"共同成功病例：{report['paired_count']}；完整集：{report['complete_benchmark']}",
        "",
        "| 组 | 尝试 | 执行失败 | 工程失败 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group['ablation'] or group['report']} | {group['attempted']} | "
            f"{group['execution_failed']} | {group['engineering_failed']} |"
        )
    lines += [
        "",
        "| 配对领域指标 | "
        + " | ".join(str(g["ablation"] or i) for i, g in enumerate(groups, 1))
        + " |",
        "| --- | " + " | ".join("---:" for _ in groups) + " |",
    ]
    names = {name for group in groups for name in cast(dict[str, object], group["paired_metrics"])}
    for name in sorted(names):
        cells = []
        for group in groups:
            metric = cast(dict[str, dict[str, object]], group["paired_metrics"]).get(name)
            cells.append(f"{metric['mean']} (n={metric['n']})" if metric else "unavailable")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    if report.get("ragas_paired_metrics"):
        lines += ["", "## RAGAs配对指标", ""]
        for name, metric in cast(
            dict[str, dict[str, object]], report["ragas_paired_metrics"]
        ).items():
            lines.append(f"- {name}: {metric['groups']}；病例={metric['paired_ids']}")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
