from httpx import ASGITransport, AsyncClient

from agentdiff.config import Settings
from agentdiff.main import create_app


async def test_healthz_ok(client: AsyncClient) -> None:
    resp = await client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok"}


async def test_healthz_reports_db_down() -> None:
    settings = Settings(
        anthropic_api_key="test-key",
        database_url="postgresql+asyncpg://localhost:59999/agentdiff_down",
    )
    app = create_app(settings)
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 503
        assert resp.json()["detail"] == {"status": "degraded", "db": "error"}
    finally:
        await app.state.engine.dispose()
