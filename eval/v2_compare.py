"""V2 评测结果对照工具：只读计算阶段和条件性研发指标。"""

import argparse
import json
from pathlib import Path


def summarize(path: Path, dataset: Path | None = None) -> dict[str, object]:
    """计算公开运行状态的工程指标，不改写输入报告。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    total = max(1, len(rows))
    references: dict[str, dict[str, list[str]]] = {}
    if dataset is not None:
        for line in dataset.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            references[case["case_id"]] = {
                "channels": case.get("expected_channels") or case.get("reference_channels") or [],
                "formulas": case.get("expected_formulas") or case.get("reference_formulas") or [],
            }
    for row in rows:
        reference = references.get(row.get("case_id"), {})
        row.setdefault("expected_channels", reference.get("channels", []))
        row.setdefault("expected_formulas", reference.get("formulas", []))

    def conditional(key: str, items: list[tuple[str, bool, bool, str]]) -> dict[str, object]:
        eligible = [item for item in items if item[1]]
        excluded = [
            {"case_id": case_id, "reason": reason}
            for case_id, is_eligible, _, reason in items
            if not is_eligible
        ]
        numerator = sum(hit for _, _, hit, _ in eligible)
        return {
            "numerator": numerator,
            "denominator": len(eligible),
            "eligibility": "case-specific reference and stage availability",
            "excluded_case_ids/reasons": excluded,
            "value": numerator / len(eligible) if eligible else None,
            "metric_key": key,
        }

    ranked: list[tuple[str, bool, bool, str]] = []
    provisional: list[tuple[str, bool, bool, str]] = []
    at20: list[tuple[str, bool, bool, str]] = []
    at5: list[tuple[str, bool, bool, str]] = []
    for row in rows:
        result = row.get("result") or {}
        expected = set(row.get("expected_formulas", []))
        candidates = (result.get("retrieval") or {}).get("formula_candidates", [])
        names = [item.get("formula_name") for item in candidates]
        comparison = result.get("comparison") or {}
        paths = comparison.get("ordered_paths", [])
        top = paths[0].get("formula_name") if paths else None
        case_id = str(row.get("case_id"))
        ranked.append(
            (
                case_id,
                bool(expected and comparison.get("status") == "ranked" and paths),
                top in expected,
                "no_reference_or_not_ranked",
            )
        )
        provisional.append(
            (case_id, bool(expected and paths), top in expected, "no_reference_or_no_path")
        )
        trace = (result.get("retrieval") or {}).get("retrieval_trace") or {}
        top20 = trace.get("top20") if isinstance(trace, dict) else None
        has_top20 = isinstance(top20, list)
        top20_names = (
            {item.get("formula_name") for item in top20 if isinstance(item, dict)}
            if has_top20
            else set()
        )
        at20.append(
            (
                case_id,
                bool(expected and has_top20),
                bool(expected & top20_names),
                (
                    "legacy_report_did_not_save_top20"
                    if not has_top20
                    else "no_reference_or_not_in_top20"
                ),
            )
        )
        at5.append(
            (
                case_id,
                bool(expected and candidates),
                bool(expected & set(names[:5])),
                "no_reference_or_not_in_s3",
            )
        )
    calculated = {
        "end_to_end_status_rate": {
            "numerator": sum(bool(row.get("status")) for row in rows),
            "denominator": len(rows),
            "eligibility": "all recorded rows",
            "excluded_case_ids/reasons": [],
        },
        "s4_ranked_top1_hit": conditional("s4_ranked_top1_hit", ranked),
        "s4_provisional_path_hit": conditional("s4_provisional_path_hit", provisional),
        "s3_reference_present_at_20": conditional("s3_reference_present_at_20", at20),
        "s3_reference_present_at_5": conditional("s3_reference_present_at_5", at5),
        "doctor_review_requested": {
            "numerator": sum(row.get("status") == "awaiting_doctor_review" for row in rows),
            "denominator": len(rows),
            "eligibility": "all recorded rows",
            "excluded_case_ids/reasons": [],
        },
    }
    details = (
        payload.get("summary", {}).get("metric_details") if isinstance(payload, dict) else None
    )
    return {
        "cases": float(len(rows)),
        "failed_rate": sum(bool(row.get("error")) for row in rows) / total,
        "clarification_rate": sum((row.get("status") == "needs_evidence") for row in rows) / total,
        "review_rate": sum(row.get("status") == "awaiting_doctor_review" for row in rows) / total,
        "metric_details": details or calculated,
    }


def main() -> None:
    """输出两个 V2 运行结果的工程指标差异。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--dataset", type=Path, help="可选的只读 JSONL 银标映射")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "left": summarize(args.left, args.dataset),
                "right": summarize(args.right, args.dataset),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
