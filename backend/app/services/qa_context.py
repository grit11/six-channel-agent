"""QA 有界上下文构造；依赖权威历史，不生成医学摘要或复制旧轮来源 JSON。"""

import re

from backend.app.schemas.knowledge import QAMessage


def build_qa_context(messages: list[QAMessage], token_budget: int) -> list[dict[str, str]]:
    """按完整已核验问答裁掉最早轮次；旧编号去掉，新轮必须重新检索。"""
    pairs: list[list[dict[str, str]]] = []
    for index in range(0, len(messages) - 1, 2):
        user, assistant = messages[index : index + 2]
        if (
            user.role != "user"
            or assistant.role != "assistant"
            or assistant.status not in {"completed", "partial"}
        ):
            continue
        if assistant.response is None or assistant.response.status != "answered":
            continue
        pairs.append(
            [
                {"role": "user", "content": user.content},
                {"role": "assistant", "content": re.sub(r"\[\d+\]", "", assistant.content)},
            ]
        )
    # 保守按字符计软预算；不是供应商真实 token 统计，禁止据此伪造 usage。
    while pairs and sum(len(m["content"]) for pair in pairs for m in pair) > token_budget:
        pairs.pop(0)
    return [message for pair in pairs[-6:] for message in pair]
