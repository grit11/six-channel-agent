"""硅基流动 Reranker 最小真实验证脚本，不输出密钥、不修改知识库或患者数据。"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService

CANDIDATE_MODELS = (
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "netease-youdao/bce-reranker-base_v1",
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
)


async def verify_model(
    client: httpx.AsyncClient, model: str, documents: list[str]
) -> tuple[int, object]:
    """向单个模型发送最小请求，只返回 HTTP 状态和脱敏响应摘要。"""
    response = await client.post(
        "rerank",
        json={
            "model": model,
            "query": "脉浮恶寒汗出恶风",
            "documents": documents,
            "top_n": len(documents),
        },
    )
    try:
        payload: object = response.json()
    except ValueError:
        payload = response.text[:200]
    return response.status_code, payload


def summarize(status: int, payload: object) -> dict[str, object]:
    """只保留可判断模型兼容性的字段，避免输出服务原始文本或凭据。"""
    if isinstance(payload, dict):
        return {
            "status_code": status,
            "keys": sorted(str(key) for key in payload),
            "error": str(payload.get("error", ""))[:160],
            "result_count": (
                len(payload.get("results", []))
                if isinstance(payload.get("results"), list)
                else None
            ),
        }
    return {"status_code": status, "response_type": type(payload).__name__}


async def main() -> None:
    """验证候选模型并输出脱敏 JSON；不在脚本中自动切换生产配置。"""
    settings = Settings()
    service = await KnowledgeService.from_directory(
        settings.knowledge_directory, settings.knowledge_base_version
    )
    evidence = [await service.get_clause_by_id(str(number)) for number in (1, 2, 35)]
    documents = [item.content for item in evidence if item is not None]
    headers = {"Authorization": f"Bearer {settings.reranker_api_key.get_secret_value()}"}
    results: dict[str, dict[str, object]] = {}
    async with httpx.AsyncClient(
        base_url=settings.reranker_base_url.rstrip("/") + "/",
        headers=headers,
        timeout=settings.reranker_timeout_seconds,
    ) as client:
        for model in CANDIDATE_MODELS:
            try:
                status, payload = await verify_model(client, model, documents)
                results[model] = summarize(status, payload)
                if status == 200 and isinstance(payload, dict) and payload.get("results"):
                    break
            except httpx.HTTPError as exc:
                results[model] = {"error_type": type(exc).__name__}
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
