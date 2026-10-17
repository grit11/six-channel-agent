"""QA 归属 API 测试；注入可信服务端身份，绝不把任意请求头当认证。"""

from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.app.agents.knowledge_qa_agent import KnowledgeQAAgent
from backend.app.api.v2.knowledge import router
from backend.app.core.config import Settings
from backend.app.knowledge.service import KnowledgeService
from backend.app.services.knowledge_qa_service import KnowledgeQAService


async def test_production_no_auth_and_cross_owner_fail_closed() -> None:
    """生产无认证 401；已认证医生无法读取/更新其他医生的 UUID。"""
    knowledge = await KnowledgeService.from_directory(
        Settings(_env_file=None).knowledge_directory, "1.2"
    )
    service = KnowledgeQAService(knowledge, KnowledgeQAAgent(knowledge))
    session = await service.repository.create("doctor-a", "测试", "1.2")
    app = FastAPI()
    app.state.container = SimpleNamespace(
        knowledge_qa_service=service, settings=SimpleNamespace(app_env="prod")
    )
    identity = {"owner": None}

    @app.middleware("http")
    async def trusted_fixture(request: Request, call_next):
        """测试夹具由服务端注入主体，不采用客户端 header。"""
        request.state.owner_id = identity["owner"]
        return await call_next(request)

    app.include_router(router, prefix="/api/v2")
    with TestClient(app) as client:
        assert (
            client.get("/api/v2/knowledge/sessions", headers={"X-Owner-Id": "doctor-a"}).status_code
            == 401
        )
        identity["owner"] = "doctor-b"
        assert client.get(f"/api/v2/knowledge/sessions/{session.session_id}").status_code == 404
        assert (
            client.get(f"/api/v2/knowledge/sessions/{session.session_id}/messages").status_code
            == 404
        )
        assert (
            client.patch(
                f"/api/v2/knowledge/sessions/{session.session_id}",
                json={"expected_revision": 0, "title": "恶意修改"},
            ).status_code
            == 404
        )
        assert client.get("/api/v2/knowledge/sessions").json() == []
        identity["owner"] = "doctor-a"
        assert client.get(f"/api/v2/knowledge/sessions/{session.session_id}").status_code == 200
