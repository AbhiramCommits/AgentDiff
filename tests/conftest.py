import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from utils import TEST_DATABASE_URL, make_settings

from agentdiff import models  # noqa: F401
from agentdiff.db import Base
from agentdiff.main import create_app


async def _database_exists(url: str) -> bool:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


async def _ensure_test_database() -> None:
    url = make_url(TEST_DATABASE_URL)
    if await _database_exists(TEST_DATABASE_URL):
        return
    admin_url = url.set(database="postgres")
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    await _ensure_test_database()
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine):
    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    await app.state.engine.dispose()
