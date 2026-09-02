from decimal import Decimal
from pathlib import Path

import pytest
from utils import FakeReviewer, make_bench_finding, make_bench_outcome

from agentdiff.bench import (
    CorpusCase,
    GroundTruthDefect,
    load_corpus,
    metrics_to_row,
    run_benchmark,
)
from agentdiff.models import Category, Severity
from agentdiff.reviewer import LLMError

CORPUS = Path(__file__).resolve().parents[1] / "bench" / "corpus"


def defect(id: str, *, line_range=(2, 2), category=Category.CORRECTNESS) -> GroundTruthDefect:
    return GroundTruthDefect(
        id=id,
        file="app.py",
        line_range=line_range,
        category=category,
        severity=Severity.MAJOR,
    )


def buggy_case(name: str) -> CorpusCase:
    return CorpusCase(name=name, patch=f"diff for {name}\n", defects=[defect(name)])


def clean_case(name: str) -> CorpusCase:
    return CorpusCase(name=name, patch=f"diff for {name}\n", defects=[])


async def test_run_benchmark_perfect_score() -> None:
    cases = [buggy_case("b1"), buggy_case("b2"), clean_case("c1")]
    reviewer = FakeReviewer(
        [
            make_bench_outcome([make_bench_finding()], latency_ms=120, cost_usd="0.01"),
            make_bench_outcome([make_bench_finding()], latency_ms=180, cost_usd="0.03"),
            make_bench_outcome([], latency_ms=50, cost_usd="0.02"),
        ]
    )
    metrics = await run_benchmark(cases, reviewer, "m", "v1", "high")

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (2, 0, 0)
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert metrics.mean_latency_ms == 116.67
    assert metrics.mean_cost_usd == Decimal("0.020000")
    assert metrics.clean_false_positive_rate == 0.0
    assert (metrics.total_cases, metrics.clean_cases) == (3, 1)


async def test_run_benchmark_all_missed() -> None:
    cases = [buggy_case("b1"), buggy_case("b2"), clean_case("c1")]
    reviewer = FakeReviewer()
    metrics = await run_benchmark(cases, reviewer, "m", "v1", "high")

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 0, 2)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


async def test_run_benchmark_wrong_category_counts_as_false_positive() -> None:
    cases = [buggy_case("b1")]
    reviewer = FakeReviewer(
        [make_bench_outcome([make_bench_finding(category=Category.SECURITY)])]
    )
    metrics = await run_benchmark(cases, reviewer, "m", "v1", "high")

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (0, 1, 1)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


async def test_run_benchmark_clean_case_false_positive_rate() -> None:
    cases = [buggy_case("b1"), clean_case("c1"), clean_case("c2")]
    reviewer = FakeReviewer(
        [
            make_bench_outcome([make_bench_finding()]),
            make_bench_outcome([make_bench_finding()]),
            make_bench_outcome([]),
        ]
    )
    metrics = await run_benchmark(cases, reviewer, "m", "v1", "high")

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 1, 0)
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.clean_false_positive_rate == 0.5


async def test_run_benchmark_failed_case_counts_defects_as_missed() -> None:
    cases = [buggy_case("b1"), buggy_case("b2")]

    class Flaky(FakeReviewer):
        async def review(self, diff, repo_context=None):
            self.calls += 1
            if self.calls == 1:
                return make_bench_outcome(
                    [make_bench_finding()], latency_ms=100, cost_usd="0.01"
                )
            raise LLMError("rate limited", 429)

    metrics = await run_benchmark(cases, Flaky(), "m", "v1", "high")

    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 0, 1)
    assert metrics.mean_latency_ms == 100.0
    assert metrics.mean_cost_usd == Decimal("0.010000")


def test_metrics_to_row_maps_fields() -> None:
    import asyncio

    async def build():
        cases = [buggy_case("b1"), clean_case("c1")]
        metrics = await run_benchmark(
            cases, FakeReviewer([make_bench_outcome([make_bench_finding()])]), "opus", "v2", "high"
        )
        return metrics

    metrics = asyncio.run(build())
    row = metrics_to_row(metrics)

    assert row.model_id == "opus"
    assert row.prompt_version == "v2"
    assert row.effort == "high"
    assert row.true_positives == 1
    assert row.f1 == 1.0
    assert row.mean_cost_usd == Decimal("0.010000")
    assert row.clean_cases == 1
    assert row.total_cases == 2


def test_load_corpus_real_corpus() -> None:
    cases = load_corpus(CORPUS)

    assert len(cases) >= 12
    buggy = [c for c in cases if not c.is_clean]
    clean = [c for c in cases if c.is_clean]
    assert len(buggy) >= 8
    assert len(clean) >= 4
    for case in buggy:
        assert len(case.defects) == 1
    assert all(len(c.patch) > 0 for c in cases)


def test_load_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    case_dir = tmp_path / "dup"
    case_dir.mkdir()
    (case_dir / "patch.diff").write_text("--- a/app.py\n+++ b/app.py\n")
    (case_dir / "expected.json").write_text(
        '{"defects": ['
        '{"id": "x", "file": "app.py", "line_range": [1, 1], "category": "correctness", "severity": "major"},'
        '{"id": "x", "file": "app.py", "line_range": [2, 2], "category": "correctness", "severity": "major"}'
        "]}"
    )
    with pytest.raises(ValueError):
        load_corpus(tmp_path)


def test_load_corpus_rejects_missing_files(tmp_path: Path) -> None:
    case_dir = tmp_path / "broken"
    case_dir.mkdir()
    (case_dir / "patch.diff").write_text("x")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
