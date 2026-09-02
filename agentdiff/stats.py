from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .bench import config_name
from .models import BenchmarkResult, Decision, Finding, GateResult, ReviewRun, RunStatus


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


async def compute_stats(session: AsyncSession, days: int) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)

    total_runs = (
        await session.scalar(
            select(func.count())
            .select_from(ReviewRun)
            .where(ReviewRun.created_at >= since)
        )
        or 0
    )

    completed_rows = (
        await session.execute(
            select(ReviewRun.latency_ms, ReviewRun.cost_usd).where(
                ReviewRun.created_at >= since,
                ReviewRun.status == RunStatus.COMPLETE,
            )
        )
    ).all()
    latencies = sorted(row.latency_ms for row in completed_rows if row.latency_ms is not None)
    costs = [row.cost_usd for row in completed_rows if row.cost_usd is not None]
    mean_cost_per_review = (
        round(float(sum(costs) / len(costs)), 6) if costs else None
    )

    gate_rows = (
        await session.execute(
            select(
                GateResult.decision,
                GateResult.reason,
                GateResult.coverage_before,
                GateResult.coverage_after,
            )
            .join(Finding, GateResult.finding_id == Finding.id)
            .join(ReviewRun, Finding.run_id == ReviewRun.id)
            .where(ReviewRun.created_at >= since)
        )
    ).all()
    accepted = sum(1 for row in gate_rows if row.decision == Decision.ACCEPTED)
    acceptance_rate = round(accepted / len(gate_rows), 6) if gate_rows else None
    rejection_reasons = Counter(
        row.reason.value for row in gate_rows if row.decision == Decision.REJECTED
    )
    deltas = [
        row.coverage_after - row.coverage_before
        for row in gate_rows
        if row.coverage_before is not None and row.coverage_after is not None
    ]
    mean_coverage_delta = round(sum(deltas) / len(deltas), 6) if deltas else None

    cost_day = func.date(func.timezone("UTC", ReviewRun.created_at)).label("day")
    cost_rows = (
        await session.execute(
            select(cost_day, func.avg(ReviewRun.cost_usd))
            .where(
                ReviewRun.created_at >= since,
                ReviewRun.status == RunStatus.COMPLETE,
                ReviewRun.cost_usd.is_not(None),
            )
            .group_by(cost_day)
            .order_by(cost_day)
        )
    ).all()
    cost_over_time = [
        {"date": str(day), "mean_cost_usd": round(float(avg), 6)}
        for day, avg in cost_rows
    ]

    benchmark_rows = (
        await session.execute(
            select(BenchmarkResult).order_by(BenchmarkResult.created_at.desc())
        )
    ).scalars().all()
    latest_by_config: dict[str, BenchmarkResult] = {}
    for row in benchmark_rows:
        key = config_name(row.model_id, row.prompt_version, row.effort)
        latest_by_config.setdefault(key, row)
    benchmarks = [
        {
            "config": key,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "clean_false_positive_rate": row.clean_false_positive_rate,
            "run_at": row.created_at,
        }
        for key, row in latest_by_config.items()
    ]
    false_positive_rate = (
        benchmark_rows[0].clean_false_positive_rate if benchmark_rows else None
    )

    return {
        "window_days": days,
        "total_runs": total_runs,
        "gated_findings": len(gate_rows),
        "acceptance_rate": acceptance_rate,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "mean_cost_per_review": mean_cost_per_review,
        "latency": {
            "p50": round(percentile([float(v) for v in latencies], 50.0), 2)
            if latencies
            else None,
            "p95": round(percentile([float(v) for v in latencies], 95.0), 2)
            if latencies
            else None,
        },
        "mean_coverage_delta": mean_coverage_delta,
        "false_positive_rate": false_positive_rate,
        "cost_over_time": cost_over_time,
        "benchmarks": benchmarks,
    }
