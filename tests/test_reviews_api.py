from contextlib import asynccontextmanager
from decimal import Decimal
from uuid import uuid4

from anthropic import NotFoundError
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from utils import (
    TEST_DATABASE_URL,
    FakeAnthropic,
    FakeMessages,
    make_settings,
    make_status_error,
    make_validation_error,
)

from agentdiff.main import create_app
from agentdiff.models import Category, Finding, ReviewRun, RunStatus, Severity
from agentdiff.reviewer import PROMPT_VERSION, ReviewFinding, ReviewResult

DIFF = "diff --git a/src/app.py b/src/app.py\n@@ -1,3 +1,4 @@\n foo\n+bar\n"

FINDING = ReviewFinding(
    file_path="src/app.py",
    start_line=10,
    end_line=12,
    severity=Severity.MAJOR,
    category=Category.CORRECTNESS,
    title="Unhandled exception on empty input",
    rationale="Calling parse() with an empty string raises ValueError and kills the worker.",
    confidence=0.9,
    suggested_patch=None,
)


@asynccontextmanager
async def api_client(fake: FakeAnthropic):
    app = create_app(make_settings(database_url=TEST_DATABASE_URL), client=fake)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield app, ac
    await app.state.engine.dispose()


def payload(**overrides):
    base = {
        "repo": f"repo-{uuid4()}",
        "base_sha": "0" * 40,
        "head_sha": "1" * 40,
        "diff": DIFF,
    }
    base.update(overrides)
    return base


async def test_create_review_persists_run_and_findings() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[FINDING]))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload())

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "complete"
    assert body["model_id"] == "claude-opus-5"
    assert body["prompt_version"] == PROMPT_VERSION
    assert body["input_tokens"] == 1000
    assert body["output_tokens"] == 200
    assert body["cache_read_tokens"] == 500
    assert body["latency_ms"] is not None
    assert body["completed_at"] is not None
    assert len(body["findings"]) == 1
    finding = body["findings"][0]
    assert finding["file_path"] == "src/app.py"
    assert finding["severity"] == "major"
    assert finding["category"] == "correctness"
    assert finding["run_id"] == body["id"]

    async with app.state.session_factory() as session:
        run = (
            await session.execute(select(ReviewRun).where(ReviewRun.id == body["id"]))
        ).scalar_one()
        assert run.status == RunStatus.COMPLETE
        assert run.completed_at is not None
        rows = (
            await session.execute(select(Finding).where(Finding.run_id == run.id))
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].title == FINDING.title


async def test_create_review_persists_cost_from_usage() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload())

    assert resp.status_code == 201
    assert float(resp.json()["cost_usd"]) == 0.01

    async with app.state.session_factory() as session:
        run = (
            await session.execute(
                select(ReviewRun).where(ReviewRun.id == resp.json()["id"])
            )
        ).scalar_one()
        assert run.cost_usd == Decimal("0.010000")


async def test_create_review_malformed_response_returns_502() -> None:
    repo = f"repo-{uuid4()}"
    messages = FakeMessages(error=make_validation_error())
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload(repo=repo))

    assert resp.status_code == 502
    assert "malformed" in resp.json()["detail"]["error"]

    async with app.state.session_factory() as session:
        run = (
            await session.execute(select(ReviewRun).where(ReviewRun.repo == repo))
        ).scalar_one()
        assert run.status == RunStatus.FAILED
        assert run.completed_at is not None


async def test_create_review_model_not_found_returns_502() -> None:
    repo = f"repo-{uuid4()}"
    messages = FakeMessages(error=make_status_error(NotFoundError, status_code=404))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload(repo=repo))

    assert resp.status_code == 502
    async with app.state.session_factory() as session:
        run = (
            await session.execute(select(ReviewRun).where(ReviewRun.repo == repo))
        ).scalar_one()
        assert run.status == RunStatus.FAILED


async def test_create_review_rate_limit_returns_429() -> None:
    from anthropic import RateLimitError

    repo = f"repo-{uuid4()}"
    messages = FakeMessages(error=make_status_error(RateLimitError, status_code=429))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload(repo=repo))

    assert resp.status_code == 429
    async with app.state.session_factory() as session:
        run = (
            await session.execute(select(ReviewRun).where(ReviewRun.repo == repo))
        ).scalar_one()
        assert run.status == RunStatus.FAILED


async def test_create_review_internal_error_returns_500_not_traceback() -> None:
    repo = f"repo-{uuid4()}"
    messages = FakeMessages(error=RuntimeError("boom"))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.post("/api/v1/reviews", json=payload(repo=repo))

    assert resp.status_code == 500
    assert resp.json()["detail"] == {"error": "internal error"}
    async with app.state.session_factory() as session:
        run = (
            await session.execute(select(ReviewRun).where(ReviewRun.repo == repo))
        ).scalar_one()
        assert run.status == RunStatus.FAILED


async def test_get_review_returns_created_run() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[FINDING]))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        created = await ac.post("/api/v1/reviews", json=payload())
        run_id = created.json()["id"]
        resp = await ac.get(f"/api/v1/reviews/{run_id}")

    assert resp.status_code == 200
    assert resp.json()["id"] == run_id
    assert len(resp.json()["findings"]) == 1


async def test_get_review_unknown_id_returns_404() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        resp = await ac.get(f"/api/v1/reviews/{uuid4()}")

    assert resp.status_code == 404


async def test_list_reviews_filters_by_repo_and_limit() -> None:
    repo_a = f"repo-a-{uuid4()}"
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    async with api_client(FakeAnthropic(messages)) as (app, ac):
        for _ in range(2):
            assert (await ac.post("/api/v1/reviews", json=payload(repo=repo_a))).status_code == 201
        assert (
            await ac.post("/api/v1/reviews", json=payload(repo=f"repo-b-{uuid4()}"))
        ).status_code == 201

        filtered = await ac.get(f"/api/v1/reviews?repo={repo_a}")
        limited = await ac.get("/api/v1/reviews?limit=2")

    assert filtered.status_code == 200
    assert len(filtered.json()) == 2
    assert all(item["repo"] == repo_a for item in filtered.json())
    assert limited.status_code == 200
    assert len(limited.json()) == 2
