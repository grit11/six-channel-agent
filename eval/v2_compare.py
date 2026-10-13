"""V2 评测结果对照工具：比较暂停率、来源覆盖和延迟，不比较旧规则分数。"""

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict[str, float]:
    """计算公开运行状态的工程指标。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else payload
    total = max(1, len(rows))
    return {
        "cases": float(len(rows)),
        "failed_rate": sum(bool(row.get("error")) for row in rows) / total,
        "clarification_rate": sum((row.get("status") == "needs_evidence") for row in rows) / total,
        "review_rate": sum(row.get("status") == "awaiting_doctor_review" for row in rows) / total,
    }


def main() -> None:
    """输出两个 V2 运行结果的工程指标差异。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {"left": summarize(args.left), "right": summarize(args.right)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
