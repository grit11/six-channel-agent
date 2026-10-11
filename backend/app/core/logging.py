"""结构化基础设施日志，只记录安全字段，不输出密钥、连接 URL 或病例文本。"""

import json
import logging

logger = logging.getLogger("six_channel")


def log_backend_event(event: str, backend: str, cause_type: str) -> None:
    """记录后端选择及异常类别，不序列化原始异常以防凭据泄露。"""
    logger.warning(
        json.dumps(
            {
                "event": event,
                "backend": backend,
                "cause_type": cause_type,
                "request_id": None,
                "consultation_id": None,
                "thread_id": None,
                "stage": "infrastructure_startup",
            },
            ensure_ascii=False,
        )
    )
