from contextlib import asynccontextmanager
from pathlib import Path

from anthropic import AsyncAnthropic
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .api.routes import router
from .config import Settings
from .db import create_engine_and_sessionmaker, get_session
from .gate import Gate
from .observability import CorrelationIdMiddleware, setup_logging
from .reviewer import Reviewer
from .schemas import StatsResponse
from .stats import compute_stats

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.engine.dispose()


def create_app(
    settings: Settings | None = None, client: AsyncAnthropic | None = None
) -> FastAPI:
    setup_logging()
    settings = settings or Settings()
    engine, session_factory = create_engine_and_sessionmaker(settings.database_url)

    app = FastAPI(title="agentdiff", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    reviewer = Reviewer(settings, client=client)
    app.state.reviewer = reviewer
    app.state.gate = Gate(settings, reviewer)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail={"status": "degraded", "db": "error"})
        return {"status": "ok", "db": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/stats", response_model=StatsResponse)
    async def stats(
        days: int = Query(default=7, ge=1, le=90),
        session: AsyncSession = Depends(get_session),
    ) -> dict:
        return await compute_stats(session, days)

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    return app


app = create_app()
