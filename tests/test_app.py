from agentdiff.config import Settings
from agentdiff.main import create_app


def test_app_factory_metadata() -> None:
    settings = Settings(anthropic_api_key="test-key", database_url="postgresql+asyncpg://localhost:59999/x")
    app = create_app(settings)

    assert app.title == "agentdiff"
    assert app.state.settings is settings


async def test_lifespan_disposes_engine() -> None:
    settings = Settings(anthropic_api_key="test-key", database_url="postgresql+asyncpg://localhost:59999/x")
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        assert app.state.engine is not None
