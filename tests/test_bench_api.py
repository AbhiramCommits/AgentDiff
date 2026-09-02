from contextlib import asynccontextmanager
from decimal import Decimal

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agentdiff.main import create_app
from agentdiff.models import BenchmarkResult
from utils import TEST_DATABASE_URL, make_settings


@asynccontextmanager
async def api_client():
    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield app, ac
    await app.state.engine.dispose()


def row(**overrides) -> BenchmarkResult:
    base = dict(
        model_id="claude-opus-5",
        prompt_version="v1",
        effort="high",
        true_positives=8,
        false_positives=1,
        false_negatives=0,
        precision=0.889,
        recall=1.0,
        f1=0.941,
        mean_latency_ms=1200.5,
        mean_cost_usd=Decimal("0.050000"),
        clean_false_positive_rate=0.0,
        total_cases=12,
        clean_cases=4,
    )
    base.update(overrides)
    return BenchmarkResult(**base)


async def seed(app: FastAPI) -> None:
    async with app.state.session_factory() as session:
        session.add(
            row(
                model_id="claude-opus-5", prompt_version="v1", effort="high",
                f1=0.941, precision=0.889, true_positives=8, false_positives=1,
            )
        )
        await session.commit()
        import asyncio

        await asyncio.sleep(0.01)
        session.add(
            row(
                model_id="claude-opus-5", prompt_version="v1", effort="high",
                f1=0.700, precision=0.700, true_positives=7, false_positives=3,
            )
        )
        await session.commit()
        await asyncio.sleep(0.01)
        session.add(
            row(
                model_id="claude-sonnet-5", prompt_version="v1", effort="high",
                f1=0.500, precision=0.500, true_positives=5, false_positives=5,
            )
        )
        await session.commit()


async def test_list_benchmarks_returns_rows_desc(db_engine) -> None:
    async with api_client() as (app, ac):
        await seed(app)
        resp = await ac.get("/api/v1/benchmarks")

    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    assert rows[0]["model_id"] == "claude-sonnet-5"
    assert rows[1]["model_id"] == "claude-opus-5"
    assert rows[1]["f1"] == 0.7
    assert rows[2]["f1"] == 0.941


async def test_compare_benchmarks_returns_latest_per_config_and_delta(db_engine) -> None:
    async with api_client() as (app, ac):
        await seed(app)
        resp = await ac.get(
            "/api/v1/benchmarks/compare",
            params={"a": "claude-opus-5@v1@high", "b": "claude-sonnet-5@v1@high"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["a"]["f1"] == 0.7
    assert body["b"]["f1"] == 0.5
    assert body["delta"]["f1"] == round(0.5 - 0.7, 6)
    assert body["delta"]["mean_cost_usd"] == 0.0


async def test_compare_benchmarks_unknown_config_404(db_engine) -> None:
    async with api_client() as (app, ac):
        await seed(app)
        resp = await ac.get(
            "/api/v1/benchmarks/compare",
            params={"a": "claude-opus-5@v1@high", "b": "unknown@v1@high"},
        )

    assert resp.status_code == 404
    assert "unknown@v1@high" in resp.json()["detail"]["error"]


async def test_compare_benchmarks_malformed_config_422(db_engine) -> None:
    async with api_client() as (app, ac):
        resp = await ac.get(
            "/api/v1/benchmarks/compare", params={"a": "not-a-config", "b": "also-bad"}
        )

    assert resp.status_code == 422


async def test_list_benchmarks_respects_limit(db_engine) -> None:
    async with api_client() as (app, ac):
        await seed(app)
        resp = await ac.get("/api/v1/benchmarks", params={"limit": "1"})

    assert resp.status_code == 200
    assert len(resp.json()) == 1
