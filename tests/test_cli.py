import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from utils import (
    GENTEST_FAIL_THEN_PASS,
    PATCH_ACCEPT,
    PATCH_DOES_NOT_APPLY,
    TEST_DATABASE_URL,
    FakeAnthropic,
    FakeMessages,
    FakeReviewer,
    build_fixture_repo,
    make_bench_finding,
    make_bench_outcome,
    make_settings,
    run_cmd,
)

from agentdiff.cli import _bench, _gate, _review, format_plain_table, format_table, results_markdown
from agentdiff.gate import GeneratedTest
from agentdiff.models import BenchmarkResult, Category, Severity
from agentdiff.reviewer import Reviewer, ReviewResult


def build_two_commit_repo(root: Path) -> tuple[str, str]:
    root.mkdir()
    (root / "app.py").write_text("x = 1\n")
    run_cmd("git", "init", "-q", cwd=root)
    run_cmd("git", "config", "user.email", "test@example.com", cwd=root)
    run_cmd("git", "config", "user.name", "Test", cwd=root)
    run_cmd("git", "add", ".", cwd=root)
    run_cmd("git", "commit", "-q", "-m", "one", cwd=root)
    base = run_cmd("git", "rev-parse", "HEAD", cwd=root).strip()
    (root / "app.py").write_text("x = 2\n")
    run_cmd("git", "add", ".", cwd=root)
    run_cmd("git", "commit", "-q", "-m", "two", cwd=root)
    head = run_cmd("git", "rev-parse", "HEAD", cwd=root).strip()
    return base, head


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


def test_format_plain_table() -> None:
    table = format_plain_table(
        ["A", "BB"],
        [["x", "yy"], ["longer", "z"]],
    )

    lines = table.splitlines()
    assert lines[0] == "A       BB"
    assert "longer" in table
    assert "yy" in table


async def test_review_command_prints_table(capsys, tmp_path: Path) -> None:
    base, head = build_two_commit_repo(tmp_path / "repo")
    reviewer = FakeReviewer(
        [
            make_bench_outcome(
                [
                    make_bench_finding(
                        start_line=3,
                        end_line=5,
                        severity=Severity.MAJOR,
                        category=Category.SECURITY,
                        title="sql injection",
                    )
                ],
                latency_ms=123,
                cost_usd="0.05",
            )
        ]
    )

    await _review(base=base, head=head, repo=tmp_path / "repo", json_output=False, reviewer=reviewer)

    out = capsys.readouterr().out
    assert "SEVERITY" in out
    assert "major" in out
    assert "security" in out
    assert "sql injection" in out
    assert "app.py:3-5" in out
    assert "tokens in/out/cache_read" in out


async def test_review_command_prints_json(capsys, tmp_path: Path) -> None:
    base, head = build_two_commit_repo(tmp_path / "repo")
    reviewer = FakeReviewer(
        [
            make_bench_outcome(
                [make_bench_finding(severity=Severity.NIT, category=Category.MAINTAINABILITY)]
            )
        ]
    )

    await _review(base=base, head=head, repo=tmp_path / "repo", json_output=True, reviewer=reviewer)

    payload = json.loads(capsys.readouterr().out)
    assert payload["findings"][0]["severity"] == "nit"
    assert payload["findings"][0]["category"] == "maintainability"


def patch_finding(severity: Severity, patch: str | None) -> ReviewResult:
    from agentdiff.reviewer import ReviewFinding

    return ReviewResult(
        findings=[
            ReviewFinding(
                file_path="mathlib/core.py",
                start_line=2,
                end_line=3,
                severity=severity,
                category=Category.CORRECTNESS,
                title="seeded bug",
                rationale="divide by zero",
                suggested_patch=patch,
                confidence=0.9,
            )
        ]
    )


def gate_cli(repo: Path, fail_on: str, findings: ReviewResult, gentest: GeneratedTest | None):
    messages = FakeMessages(
        parsed_outputs=(
            [findings] if gentest is None else [findings, gentest]
        )
    )
    settings = make_settings(workspace_dir=repo, coverage_package="mathlib")
    reviewer = Reviewer(settings, client=FakeAnthropic(messages))
    return _gate(
        base="HEAD",
        head="HEAD",
        repo=repo,
        fail_on=fail_on,
        settings=settings,
        reviewer=reviewer,
    )


async def test_gate_command_accepted_major_fails_on_major(capsys, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    build_fixture_repo(repo)

    code = await gate_cli(
        repo,
        "major",
        patch_finding(Severity.MAJOR, PATCH_ACCEPT),
        GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS),
    )

    out = capsys.readouterr()
    assert "accepted=1 rejected=0" in out.out
    assert "FAIL" in out.err
    assert code == 1


async def test_gate_command_accepted_major_passes_on_blocker(capsys, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    build_fixture_repo(repo)

    code = await gate_cli(
        repo,
        "blocker",
        patch_finding(Severity.MAJOR, PATCH_ACCEPT),
        GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS),
    )

    out = capsys.readouterr()
    assert "accepted=1 rejected=0" in out.out
    assert code == 0


async def test_gate_command_rejected_blocker_never_fails(capsys, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    build_fixture_repo(repo)

    code = await gate_cli(
        repo,
        "minor",
        patch_finding(Severity.BLOCKER, PATCH_DOES_NOT_APPLY),
        None,
    )

    out = capsys.readouterr()
    assert "accepted=0 rejected=1" in out.out
    assert "patch_did_not_apply" in out.out
    assert code == 0


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
        (case_dir / "patch.diff").write_text("--- a/app.py\n+++ b/app.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n")
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
