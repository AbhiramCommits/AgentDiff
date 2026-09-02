from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from .api.routes import router
from .config import Settings
from .db import create_engine_and_sessionmaker
from .reviewer import Reviewer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.engine.dispose()


def create_app(
    settings: Settings | None = None, client: AsyncAnthropic | None = None
) -> FastAPI:
    settings = settings or Settings()
    engine, session_factory = create_engine_and_sessionmaker(settings.database_url)

    app = FastAPI(title="agentdiff", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.reviewer = Reviewer(settings, client=client)
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail={"status": "degraded", "db": "error"})
        return {"status": "ok", "db": "ok"}

    return app


app = create_app()
