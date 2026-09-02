import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agentdiff.cli import _bench, format_table, results_markdown
from agentdiff.models import BenchmarkResult, Category, Severity
from utils import (
    TEST_DATABASE_URL,
    FakeReviewer,
    make_bench_finding,
    make_bench_outcome,
)


def sample_rows() -> list[BenchmarkResult]:
    from decimal import Decimal

    return [
        BenchmarkResult(
            model_id="claude-opus-5", prompt_version="v1", effort="high",
            true_positives=8, false_positives=1, false_negatives=0,
            precision=0.889, recall=1.0, f1=0.941,
            mean_latency_ms=1234.5, mean_cost_usd=Decimal("0.050000"),
            clean_false_positive_rate=0.0, total_cases=12, clean_cases=4,
        ),
        BenchmarkResult(
            model_id="claude-haiku-4-5", prompt_version="v2", effort="low",
            true_positives=3, false_positives=6, false_negatives=5,
            precision=0.333, recall=0.375, f1=0.353,
            mean_latency_ms=300.0, mean_cost_usd=Decimal("0.010000"),
            clean_false_positive_rate=1.0, total_cases=12, clean_cases=4,
        ),
    ]


def test_main_module_importable() -> None:
    import agentdiff.__main__ as main

    assert main.app is not None


def test_format_table_includes_headers_and_rows() -> None:
    table = format_table(sample_rows())

    assert "config" in table
    assert "precision" in table
    assert "claude-opus-5@v1@high" in table
    assert "0.941" in table
    assert "claude-haiku-4-5@v2@low" in table


def test_results_markdown_is_a_scoreboard() -> None:
    md = results_markdown(sample_rows())

    assert md.startswith("# agentdiff benchmark scoreboard")
    assert "| config |" in md
    assert "claude-opus-5@v1@high" in md
    assert "0.889" in md


def test_bench_cli_end_to_end(tmp_path: Path, db_engine) -> None:
    corpus = tmp_path / "corpus"
    for name, defects in [("b1", True), ("c1", False)]:
        case_dir = corpus / name
        (case_dir / "before").mkdir(parents=True)
        (case_dir / "before" / "app.py").write_text("x = 1\n")
        (case_dir / "patch.diff").write_text(f"--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n")
        entries = []
        if defects:
            entries.append(
                {
                    "id": name,
                    "file": "app.py",
                    "line_range": [1, 1],
                    "category": "correctness",
                    "severity": "major",
                }
            )
        (case_dir / "expected.json").write_text(json.dumps({"defects": entries}))

    reviewer = FakeReviewer(
        [
            make_bench_outcome(
                [make_bench_finding(start_line=1, end_line=1, category=Category.CORRECTNESS)],
                latency_ms=200,
                cost_usd="0.02",
            ),
            make_bench_outcome([], latency_ms=100, cost_usd="0.01"),
        ]
    )
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    results_file = tmp_path / "RESULTS.md"

    try:
        metrics = asyncio.run(
            _bench(
                model="claude-opus-5",
                effort="high",
                prompt_version="v1",
                corpus_dir=corpus,
                results_file=results_file,
                reviewer=reviewer,
                session_factory=session_factory,
            )
        )
    finally:
        asyncio.run(engine.dispose())

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.f1 == 1.0
    assert results_file.exists()
    assert "claude-opus-5@v1@high" in results_file.read_text()

    async def check_db():
        engine2 = create_async_engine(TEST_DATABASE_URL)
        sf = async_sessionmaker(engine2, expire_on_commit=False)
        async with sf() as session:
            rows = list(
                (
                    await session.execute(
                        select(BenchmarkResult).where(
                            BenchmarkResult.model_id == "claude-opus-5"
                        )
                    )
                )
                .scalars()
                .all()
            )
        await engine2.dispose()
        return rows

    rows = asyncio.run(check_db())
    assert len(rows) >= 1
    assert rows[-1].f1 == 1.0
    assert rows[-1].clean_cases == 1
