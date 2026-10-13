"""V2 配置门禁：读取项目 .env 的字段名并通过 Settings 校验，不打印密钥。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings


def env_keys(path: Path) -> set[str]:
    """读取 dotenv 键名，不解析或回显敏感值。"""
    if not path.is_file():
        return set()
    return {
        line.split("=", 1)[0].strip().upper()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }


def check(env_file: Path) -> dict[str, object]:
    """验证 V2 固定协议和 provider 接口。"""
    keys = env_keys(env_file)
    old = sorted(keys & {"CLINICAL_PARSER_MODE", "REASONING_MODE", "SCORING_CONFIG_VERSION"})
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    return {
        "protocol": settings.protocol_version,
        "engine": "clinical-v2.0",
        "old_keys": old,
        "llm_configured": bool(settings.llm_base_url and settings.llm_model),
        "retrieval_mode": settings.retrieval_mode,
        "reranker_enabled": settings.reranker_enabled,
        "checkpoint_backend": settings.checkpoint_backend,
        "consultation_storage": settings.consultation_storage,
        "status": (
            "passed" if settings.protocol_version == "clinical-v2.0" and not old else "failed"
        ),
    }


def main() -> int:
    """执行配置门禁并返回真实退出码。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    try:
        result = check(args.env_file)
    except (ValueError, OSError, RuntimeError) as exc:
        print({"status": "failed", "error": type(exc).__name__})
        return 1
    print(result)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
