from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from utils import TEST_DATABASE_URL, make_settings

from agentdiff.main import create_app
from agentdiff.models import (
    BenchmarkResult,
    Category,
    Decision,
    Finding,
    GateReason,
    GateResult,
    ReviewRun,
    RunStatus,
    Severity,
)
from agentdiff.stats import percentile

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_percentile() -> None:
    assert percentile([1.0], 50.0) == 1.0
    values = [float(v) for v in range(100, 1001, 100)]
    assert percentile(values, 50.0) == 550.0
    assert percentile(values, 95.0) == pytest.approx(955.0)
    with pytest.raises(ValueError):
        percentile([], 50.0)


def benchmark_row(
    model_id, prompt_version, effort, f1, precision, recall, clean_fp_rate, created_at
) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=model_id,
        prompt_version=prompt_version,
        effort=effort,
        true_positives=5,
        false_positives=1,
        false_negatives=3,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_latency_ms=100.0,
        mean_cost_usd=Decimal("0.010000"),
        clean_false_positive_rate=clean_fp_rate,
        total_cases=12,
        clean_cases=4,
        created_at=created_at,
    )


async def seed(session_factory) -> None:
    async with session_factory() as session:
        for model in (GateResult, Finding, ReviewRun, BenchmarkResult):
            await session.execute(delete(model))

        for i in range(5):
            day = NOW - timedelta(days=1, hours=1 + i)
            session.add(
                ReviewRun(
                    repo=f"repo-{i}",
                    base_sha="0" * 40,
                    head_sha="1" * 40,
                    model_id="claude-opus-5",
                    prompt_version="v1",
                    status=RunStatus.COMPLETE,
                    created_at=day,
                    completed_at=day,
                    latency_ms=100 * (i + 1),
                    cost_usd=Decimal(i + 1) / Decimal(100),
                )
            )
        for i in range(5, 10):
            day = NOW - timedelta(days=2, hours=1 + i)
            session.add(
                ReviewRun(
                    repo=f"repo-{i}",
                    base_sha="0" * 40,
                    head_sha="1" * 40,
                    model_id="claude-opus-5",
                    prompt_version="v1",
                    status=RunStatus.COMPLETE,
                    created_at=day,
                    completed_at=day,
                    latency_ms=100 * (i + 1),
                    cost_usd=Decimal(i + 1) / Decimal(100),
                )
            )
        session.add(
            ReviewRun(
                repo="old-run",
                base_sha="0" * 40,
                head_sha="1" * 40,
                model_id="claude-opus-5",
                prompt_version="v1",
                status=RunStatus.COMPLETE,
                created_at=NOW - timedelta(days=10),
                completed_at=NOW - timedelta(days=10),
                latency_ms=5000,
                cost_usd=Decimal("5.000000"),
            )
        )
        session.add(
            ReviewRun(
                repo="running-run",
                base_sha="0" * 40,
                head_sha="1" * 40,
                model_id="claude-opus-5",
                prompt_version="v1",
                status=RunStatus.RUNNING,
                created_at=NOW - timedelta(days=1, hours=7),
            )
        )
        await session.commit()

        completed = (
            await session.execute(
                select(ReviewRun)
                .where(
                    ReviewRun.status == RunStatus.COMPLETE,
                    ReviewRun.created_at >= NOW - timedelta(days=7),
                )
                .order_by(ReviewRun.created_at)
            )
        ).scalars().all()
        for index, run in enumerate(completed):
            finding = Finding(
                run_id=run.id,
                file_path="app.py",
                start_line=1,
                end_line=1,
                severity=Severity.MAJOR,
                category=Category.CORRECTNESS,
                title="t",
                rationale="r",
                confidence=0.9,
            )
            session.add(finding)
            await session.flush()
            if index < 8:
                session.add(
                    GateResult(
                        finding_id=finding.id,
                        decision=Decision.ACCEPTED,
                        reason=GateReason.TESTS_PASSED,
                        tests_passed=3,
                        tests_failed=0,
                        coverage_before=80.0,
                        coverage_after=82.0,
                    )
                )
            elif index == 8:
                session.add(
                    GateResult(
                        finding_id=finding.id,
                        decision=Decision.REJECTED,
                        reason=GateReason.COVERAGE_DROPPED,
                        tests_passed=3,
                        tests_failed=0,
                        coverage_before=80.0,
                        coverage_after=75.0,
                    )
                )
            else:
                session.add(
                    GateResult(
                        finding_id=finding.id,
                        decision=Decision.REJECTED,
                        reason=GateReason.TESTS_FAILED,
                        tests_passed=2,
                        tests_failed=1,
                        coverage_before=80.0,
                        coverage_after=None,
                    )
                )
        await session.commit()

        session.add(
            benchmark_row("claude-opus-5", "v1", "high", 0.5, 0.6, 0.45, 0.1, NOW - timedelta(days=2))
        )
        session.add(
            benchmark_row("claude-sonnet-5", "v2", "low", 0.6, 0.7, 0.55, 0.5, NOW - timedelta(days=1, hours=12))
        )
        session.add(
            benchmark_row("claude-opus-5", "v1", "high", 0.9, 1.0, 0.85, 0.25, NOW - timedelta(days=1))
        )
        await session.commit()


async def test_stats_math_on_seeded_database(db_engine) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    await seed(session_factory)

    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/stats")
    finally:
        await app.state.engine.dispose()
        await engine.dispose()

    assert resp.status_code == 200
    stats = resp.json()

    assert stats["window_days"] == 7
    assert stats["total_runs"] == 11
    assert stats["gated_findings"] == 10
    assert stats["acceptance_rate"] == 0.8
    assert stats["rejection_reasons"] == {"coverage_dropped": 1, "tests_failed": 1}
    assert stats["mean_cost_per_review"] == 0.055
    assert stats["latency"] == {"p50": 550.0, "p95": 955.0}
    assert stats["mean_coverage_delta"] == 1.222222
    assert stats["false_positive_rate"] == 0.25

    cost_dates = sorted(p["date"] for p in stats["cost_over_time"])
    assert len(cost_dates) == 2
    costs = {p["date"]: p["mean_cost_usd"] for p in stats["cost_over_time"]}
    assert sorted(costs.values()) == [0.03, 0.08]

    by_config = {b["config"]: b for b in stats["benchmarks"]}
    assert by_config["claude-opus-5@v1@high"]["f1"] == 0.9
    assert by_config["claude-opus-5@v1@high"]["precision"] == 1.0
    assert by_config["claude-opus-5@v1@high"]["clean_false_positive_rate"] == 0.25
    assert by_config["claude-sonnet-5@v2@low"]["f1"] == 0.6
    assert by_config["claude-sonnet-5@v2@low"]["precision"] == 0.7
    assert by_config["claude-sonnet-5@v2@low"]["clean_false_positive_rate"] == 0.5


async def test_stats_empty_database(db_engine) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        for model in (GateResult, Finding, ReviewRun, BenchmarkResult):
            await session.execute(delete(model))
        await session.commit()

    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/stats")
    finally:
        await app.state.engine.dispose()
        await engine.dispose()

    assert resp.status_code == 200
    stats = resp.json()
    assert stats["total_runs"] == 0
    assert stats["acceptance_rate"] is None
    assert stats["latency"] == {"p50": None, "p95": None}
    assert stats["mean_cost_per_review"] is None
    assert stats["benchmarks"] == []
