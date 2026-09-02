from pathlib import Path

from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY
from utils import (
    GENTEST_FAIL_THEN_PASS,
    PATCH_ACCEPT,
    TEST_DATABASE_URL,
    FakeAnthropic,
    FakeMessages,
    build_fixture_repo,
    make_settings,
    make_status_error,
)

from agentdiff.gate import GeneratedTest
from agentdiff.main import create_app
from agentdiff.models import Category, Severity
from agentdiff.reviewer import ReviewFinding, ReviewResult

ALL_METRIC_NAMES = [
    "agentdiff_review_latency_seconds",
    "agentdiff_gate_duration_seconds",
    "agentdiff_tokens_total",
    "agentdiff_cost_usd_total",
    "agentdiff_findings_total",
    "agentdiff_suggestions_total",
    "agentdiff_coverage_delta",
    "agentdiff_reviews_in_flight",
    "agentdiff_llm_errors_total",
]


def counter_value(name: str, labels: dict[str, str] | None = None) -> float:
    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name == name or metric.name + "_total" == name:
            for sample in metric.samples:
                if sample.name.endswith(("_total", "_count")):
                    if labels is None or all(
                        sample.labels.get(key) == value for key, value in labels.items()
                    ):
                        total += sample.value
    return total


def gauge_value(name: str) -> float:
    for metric in REGISTRY.collect():
        if metric.name == name:
            for sample in metric.samples:
                if sample.name == name:
                    return sample.value
    return 0.0


def patch_finding() -> ReviewFinding:
    return ReviewFinding(
        file_path="mathlib/core.py",
        start_line=2,
        end_line=3,
        severity=Severity.MAJOR,
        category=Category.CORRECTNESS,
        title="silent zero on divide by zero",
        rationale="divide(1, 0) returns 0.0 instead of raising.",
        suggested_patch=PATCH_ACCEPT,
        confidence=0.9,
    )


def no_patch_finding() -> ReviewFinding:
    return ReviewFinding(
        file_path="mathlib/core.py",
        start_line=1,
        end_line=1,
        severity=Severity.NIT,
        category=Category.MAINTAINABILITY,
        title="missing docstring",
        rationale="No docstring on the module function.",
        suggested_patch=None,
        confidence=0.5,
    )


async def test_metrics_expose_every_metric_after_review_and_gate(tmp_path: Path, db_engine) -> None:
    repo = tmp_path / "repo"
    head_sha = build_fixture_repo(repo)
    messages = FakeMessages(
        parsed_outputs=[
            ReviewResult(findings=[patch_finding(), no_patch_finding()]),
            GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS),
        ]
    )
    app = create_app(
        make_settings(
            database_url=TEST_DATABASE_URL,
            workspace_dir=repo,
            coverage_package="mathlib",
        ),
        client=FakeAnthropic(messages),
    )
    transport = ASGITransport(app=app)

    model_labels = {"model": "claude-opus-5"}
    before = {
        "tokens_input": counter_value("agentdiff_tokens_total", {**model_labels, "kind": "input"}),
        "tokens_output": counter_value("agentdiff_tokens_total", {**model_labels, "kind": "output"}),
        "tokens_cache": counter_value("agentdiff_tokens_total", {**model_labels, "kind": "cache_read"}),
        "cost": counter_value("agentdiff_cost_usd_total", model_labels),
        "finding_major": counter_value(
            "agentdiff_findings_total", {"severity": "major", "category": "correctness"}
        ),
        "finding_nit": counter_value(
            "agentdiff_findings_total", {"severity": "nit", "category": "maintainability"}
        ),
        "latency": counter_value(
            "agentdiff_review_latency_seconds",
            {"model": "claude-opus-5", "prompt_version": "v1"},
        ),
        "gate": counter_value("agentdiff_gate_duration_seconds"),
        "accepted": counter_value(
            "agentdiff_suggestions_total", {"decision": "accepted", "reason": "tests_passed"}
        ),
        "no_patch": counter_value(
            "agentdiff_suggestions_total", {"decision": "rejected", "reason": "no_patch"}
        ),
        "coverage_delta": counter_value("agentdiff_coverage_delta"),
        "in_flight": gauge_value("agentdiff_reviews_in_flight"),
        "llm_not_found": counter_value("agentdiff_llm_errors_total", {"error_type": "not_found"}),
    }

    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.post(
                "/api/v1/reviews",
                params={"gate": "true"},
                json={
                    "repo": "metrics-repo",
                    "base_sha": "0" * 40,
                    "head_sha": head_sha,
                    "diff": "diff --git a/mathlib/core.py b/mathlib/core.py\n",
                },
            )
        assert resp.status_code == 201
        gate_results = {f["title"]: f["gate_result"] for f in resp.json()["findings"]}
        assert gate_results["silent zero on divide by zero"]["decision"] == "accepted"
        assert gate_results["missing docstring"]["reason"] == "no_patch"

        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            metrics_resp = await ac.get("/metrics")
    finally:
        await app.state.engine.dispose()

    assert metrics_resp.status_code == 200
    assert metrics_resp.headers["content-type"].startswith("text/plain")
    body = metrics_resp.text
    for name in ALL_METRIC_NAMES:
        assert name in body, f"missing metric {name}"

    assert counter_value("agentdiff_tokens_total", {**model_labels, "kind": "input"}) > before["tokens_input"]
    assert counter_value("agentdiff_tokens_total", {**model_labels, "kind": "output"}) > before["tokens_output"]
    assert counter_value("agentdiff_tokens_total", {**model_labels, "kind": "cache_read"}) > before["tokens_cache"]
    assert counter_value("agentdiff_cost_usd_total", model_labels) > before["cost"]
    assert counter_value("agentdiff_findings_total", {"severity": "major", "category": "correctness"}) == before["finding_major"] + 1
    assert counter_value("agentdiff_findings_total", {"severity": "nit", "category": "maintainability"}) == before["finding_nit"] + 1
    assert counter_value("agentdiff_review_latency_seconds", {"model": "claude-opus-5", "prompt_version": "v1"}) == before["latency"] + 1
    assert counter_value("agentdiff_gate_duration_seconds") == before["gate"] + 2
    assert counter_value("agentdiff_suggestions_total", {"decision": "accepted", "reason": "tests_passed"}) == before["accepted"] + 1
    assert counter_value("agentdiff_suggestions_total", {"decision": "rejected", "reason": "no_patch"}) == before["no_patch"] + 1
    assert counter_value("agentdiff_coverage_delta") == before["coverage_delta"] + 1
    assert gauge_value("agentdiff_reviews_in_flight") == before["in_flight"]
    assert counter_value("agentdiff_llm_errors_total", {"error_type": "not_found"}) == before["llm_not_found"]


async def test_llm_error_counter_incremented(db_engine) -> None:
    from anthropic import NotFoundError

    before = counter_value("agentdiff_llm_errors_total", {"error_type": "not_found"})
    messages = FakeMessages(error=make_status_error(NotFoundError, status_code=404))
    app = create_app(
        make_settings(database_url=TEST_DATABASE_URL),
        client=FakeAnthropic(messages),
    )
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.post(
                "/api/v1/reviews",
                json={"repo": "r", "base_sha": "0" * 40, "head_sha": "1" * 40, "diff": "d\n"},
            )
        assert resp.status_code == 502
    finally:
        await app.state.engine.dispose()

    assert counter_value("agentdiff_llm_errors_total", {"error_type": "not_found"}) == before + 1


async def test_dashboard_renders(db_engine) -> None:
    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/")
    finally:
        await app.state.engine.dispose()

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    assert "agentdiff dashboard" in body
    assert 'id="acceptance"' in body
    assert 'id="reasons"' in body
    assert 'id="cost"' in body
    assert 'id="latency"' in body
    assert 'id="bench"' in body
    assert "fetch(\"/stats\")" in body
    assert "prefers-color-scheme: dark" in body
