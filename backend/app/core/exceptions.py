"""
应用异常定义。

职责：提供跨 Adapter、Domain、Application 和 API 层统一使用的异常类型。
边界：本模块不记录日志，也不直接构造 HTTP 响应。
"""


class ApplicationError(Exception):
    """所有可预期应用错误的基类。"""

    code = "APPLICATION_ERROR"

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        """保存对外安全消息和结构化详情，原始异常用异常链保留。"""
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(ApplicationError):
    """配置缺失或不合法。"""

    code = "CONFIGURATION_ERROR"


class ProviderUnavailableError(ApplicationError):
    """外部模型或基础设施 Provider 不可用。"""

    code = "PROVIDER_UNAVAILABLE"


class PersistenceError(ApplicationError):
    """持久化操作失败。"""

    code = "PERSISTENCE_ERROR"


class KnowledgeRetrievalError(ApplicationError):
    """知识召回或证据校验失败。"""

    code = "KNOWLEDGE_RETRIEVAL_ERROR"


class KnowledgeNotFoundError(ApplicationError):
    """精确知识资源不存在，不能以模糊搜索代替。"""

    code = "KNOWLEDGE_NOT_FOUND"


class ConsultationNotFoundError(ApplicationError):
    """指定辨证会话不存在。"""

    code = "CONSULTATION_NOT_FOUND"


class InvalidStateTransitionError(ApplicationError):
    """当前会话状态不允许执行请求动作。"""

    code = "INVALID_STATE_TRANSITION"


class InvalidBusinessRequestError(ApplicationError):
    """Schema 合法，但请求内容不满足业务动作要求。"""

    code = "INVALID_BUSINESS_REQUEST"
