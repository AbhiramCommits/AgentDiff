from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from utils import (
    GENTEST_FAIL_THEN_PASS,
    GENTEST_PASSES_BOTH,
    PATCH_ACCEPT,
    PATCH_COVERAGE_DROP,
    PATCH_DOES_NOT_APPLY,
    PATCH_TESTS_FAIL,
    TEST_DATABASE_URL,
    FakeAnthropic,
    FakeMessages,
    build_fixture_repo,
    make_settings,
)

from agentdiff.gate import GeneratedTest
from agentdiff.main import create_app
from agentdiff.models import Category, Finding, GateResult, ReviewRun, Severity
from agentdiff.reviewer import ReviewFinding, ReviewResult

DIFF = "diff --git a/mathlib/core.py b/mathlib/core.py\n@@ -2,3 +2,3 @@\n foo\n-bar\n+baz\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path / "repo"


@pytest.fixture
def head_sha(repo: Path) -> str:
    return build_fixture_repo(repo)


@asynccontextmanager
async def api_client(fake: FakeAnthropic, repo: Path):
    app = create_app(
        make_settings(
            database_url=TEST_DATABASE_URL,
            workspace_dir=repo,
            coverage_package="mathlib",
        ),
        client=fake,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield app, ac
    await app.state.engine.dispose()


def finding_payload(title: str, **overrides) -> ReviewFinding:
    base = dict(
        file_path="mathlib/core.py",
        start_line=3,
        end_line=3,
        severity=Severity.MAJOR,
        category=Category.CORRECTNESS,
        title=title,
        rationale="divide(1, 0) returns 0.0 instead of raising ValueError.",
        suggested_patch=PATCH_ACCEPT,
        confidence=0.9,
    )
    base.update(overrides)
    return ReviewFinding(**base)


def review_payload(head_sha: str, repo: str = None) -> dict:
    return {
        "repo": repo or f"repo-{uuid4()}",
        "base_sha": "0" * 40,
        "head_sha": head_sha,
        "diff": DIFF,
    }


async def test_gate_endpoint_persists_results_for_all_reasons(head_sha, repo) -> None:
    findings = [
        finding_payload("accept"),
        finding_payload("breaks tests", suggested_patch=PATCH_TESTS_FAIL),
        finding_payload("coverage", suggested_patch=PATCH_COVERAGE_DROP),
        finding_payload("no apply", suggested_patch=PATCH_DOES_NOT_APPLY),
        finding_payload("no patch", suggested_patch=None),
    ]
    messages = FakeMessages(
        parsed_outputs=[
            ReviewResult(findings=findings),
            GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS),
        ]
    )
    async with api_client(FakeAnthropic(messages), repo) as (app, ac):
        created = await ac.post("/api/v1/reviews", json=review_payload(head_sha))
        assert created.status_code == 201
        run_id = created.json()["id"]

        resp = await ac.post(f"/api/v1/reviews/{run_id}/gate")

        assert resp.status_code == 200
        by_title = {f["title"]: f for f in resp.json()["findings"]}
        assert by_title["accept"]["gate_result"]["decision"] == "accepted"
        assert by_title["accept"]["gate_result"]["reason"] == "tests_passed"
        assert by_title["accept"]["gate_result"]["verification"] == "verified"
        assert by_title["breaks tests"]["gate_result"]["reason"] == "tests_failed"
        assert by_title["coverage"]["gate_result"]["reason"] == "coverage_dropped"
        assert by_title["no apply"]["gate_result"]["reason"] == "patch_did_not_apply"
        assert by_title["no patch"]["gate_result"]["reason"] == "no_patch"

        async with app.state.session_factory() as session:
            rows = (
                await session.execute(
                    select(GateResult)
                    .join(Finding, GateResult.finding_id == Finding.id)
                    .where(Finding.run_id == run_id)
                )
            ).scalars().all()
            assert len(rows) == 5


async def test_post_reviews_with_gate_true_runs_review_then_gate(head_sha, repo) -> None:
    messages = FakeMessages(
        parsed_outputs=[
            ReviewResult(findings=[finding_payload("accept")]),
            GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS),
        ]
    )
    async with api_client(FakeAnthropic(messages), repo) as (app, ac):
        resp = await ac.post(
            "/api/v1/reviews", json=review_payload(head_sha), params={"gate": "true"}
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "complete"
    gate_result = body["findings"][0]["gate_result"]
    assert gate_result["decision"] == "accepted"
    assert gate_result["verification"] == "verified"


async def test_gate_downgrades_confidence_when_unverified(head_sha, repo) -> None:
    messages = FakeMessages(
        parsed_outputs=[
            ReviewResult(findings=[finding_payload("accept")]),
            GeneratedTest(test_code=GENTEST_PASSES_BOTH),
        ]
    )
    async with api_client(FakeAnthropic(messages), repo) as (app, ac):
        resp = await ac.post(
            "/api/v1/reviews", json=review_payload(head_sha), params={"gate": "true"}
        )

    assert resp.status_code == 201
    finding = resp.json()["findings"][0]
    assert finding["confidence"] == 0.45
    assert finding["gate_result"]["verification"] == "unverified"

    async with app.state.session_factory() as session:
        run = (
            await session.execute(
                select(ReviewRun)
                .options(selectinload(ReviewRun.findings))
                .where(ReviewRun.id == resp.json()["id"])
            )
        ).scalar_one()
        gate_result = (
            await session.execute(
                select(GateResult).where(
                    GateResult.finding_id == run.findings[0].id
                )
            )
        ).scalar_one()
        assert gate_result.verification == "unverified"
        assert run.findings[0].confidence == 0.45


async def test_gate_unknown_run_returns_404(head_sha, repo) -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    async with api_client(FakeAnthropic(messages), repo) as (app, ac):
        resp = await ac.post(f"/api/v1/reviews/{uuid4()}/gate")

    assert resp.status_code == 404


async def test_gate_failed_run_returns_409(head_sha, repo) -> None:
    repo_name = f"repo-{uuid4()}"
    messages = FakeMessages(error=RuntimeError("boom"))
    async with api_client(FakeAnthropic(messages), repo) as (app, ac):
        await ac.post("/api/v1/reviews", json=review_payload(head_sha, repo=repo_name))
        async with app.state.session_factory() as session:
            run = (
                await session.execute(
                    select(ReviewRun).where(ReviewRun.repo == repo_name)
                )
            ).scalar_one()
            run_id = run.id
        resp = await ac.post(f"/api/v1/reviews/{run_id}/gate")

    assert resp.status_code == 409
    assert "cannot gate" in resp.json()["detail"]["error"]
