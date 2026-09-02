import asyncio
from pathlib import Path

import pytest

from agentdiff.gate import (
    Gate,
    GeneratedTest,
    PytestOutcome,
)
from agentdiff.models import Category, Decision, Finding, GateReason, Severity
from agentdiff.reviewer import Reviewer
from utils import (
    GENTEST_FAIL_THEN_PASS,
    GENTEST_PASSES_BOTH,
    PATCH_ACCEPT,
    PATCH_COVERAGE_DROP,
    PATCH_DOES_NOT_APPLY,
    PATCH_TESTS_FAIL,
    FakeAnthropic,
    FakeMessages,
    build_fixture_repo,
    build_slow_fixture_repo,
    make_settings,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path / "repo"


@pytest.fixture
def head_sha(repo: Path) -> str:
    return build_fixture_repo(repo)


def make_finding(**overrides) -> Finding:
    base = dict(
        file_path="mathlib/core.py",
        start_line=3,
        end_line=3,
        severity=Severity.MAJOR,
        category=Category.CORRECTNESS,
        title="Divide by zero silently returns zero",
        rationale="divide(1, 0) returns 0.0 instead of raising ValueError.",
        suggested_patch=PATCH_ACCEPT,
        confidence=0.9,
    )
    base.update(overrides)
    return Finding(**base)


def make_gate(repo: Path, messages: FakeMessages, **settings_overrides) -> Gate:
    settings = make_settings(
        workspace_dir=repo,
        coverage_package="mathlib",
        **settings_overrides,
    )
    reviewer = Reviewer(settings, client=FakeAnthropic(messages))
    return Gate(settings, reviewer)


def worktree_base(repo: Path) -> Path:
    return repo / "agentdiff-worktrees"


def assert_no_worktrees(repo: Path) -> None:
    base = worktree_base(repo)
    assert not base.exists() or not any(base.iterdir())


async def test_gate_accepts_clean_patch_and_verifies(head_sha, repo) -> None:
    messages = FakeMessages(
        parsed_outputs=[GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS)]
    )
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(make_finding(), head_sha)

    assert decision.decision == Decision.ACCEPTED
    assert decision.reason == GateReason.TESTS_PASSED
    assert decision.tests_passed == 3
    assert decision.tests_failed == 0
    assert decision.verification == "verified"
    assert decision.coverage_before is not None
    assert decision.coverage_after is not None
    assert decision.coverage_after >= decision.coverage_before
    assert_no_worktrees(repo)


async def test_gate_rejects_when_patch_breaks_tests(head_sha, repo) -> None:
    messages = FakeMessages()
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(
        make_finding(suggested_patch=PATCH_TESTS_FAIL), head_sha
    )

    assert decision.decision == Decision.REJECTED
    assert decision.reason == GateReason.TESTS_FAILED
    assert decision.tests_failed >= 1
    assert decision.coverage_after is not None
    assert_no_worktrees(repo)


async def test_gate_rejects_when_coverage_drops(head_sha, repo) -> None:
    messages = FakeMessages()
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(
        make_finding(suggested_patch=PATCH_COVERAGE_DROP), head_sha
    )

    assert decision.decision == Decision.REJECTED
    assert decision.reason == GateReason.COVERAGE_DROPPED
    assert decision.tests_failed == 0
    assert decision.coverage_before is not None
    assert decision.coverage_after is not None
    assert decision.coverage_after < decision.coverage_before
    assert_no_worktrees(repo)


async def test_gate_tolerance_allows_small_coverage_drop(head_sha, repo) -> None:
    messages = FakeMessages(
        parsed_outputs=[GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS)]
    )
    gate = make_gate(repo, messages, coverage_tolerance=100.0)
    decision = await gate.gate_finding(
        make_finding(suggested_patch=PATCH_COVERAGE_DROP), head_sha
    )

    assert decision.decision == Decision.ACCEPTED
    assert decision.reason == GateReason.TESTS_PASSED
    assert_no_worktrees(repo)


async def test_gate_rejects_patch_that_does_not_apply(head_sha, repo) -> None:
    messages = FakeMessages()
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(
        make_finding(suggested_patch=PATCH_DOES_NOT_APPLY), head_sha
    )

    assert decision.decision == Decision.REJECTED
    assert decision.reason == GateReason.PATCH_DID_NOT_APPLY
    assert decision.tests_passed == 0
    assert_no_worktrees(repo)


async def test_gate_rejects_finding_without_patch(head_sha, repo) -> None:
    messages = FakeMessages()
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(
        make_finding(suggested_patch=None), head_sha
    )

    assert decision.decision == Decision.REJECTED
    assert decision.reason == GateReason.NO_PATCH
    assert_no_worktrees(repo)
    assert messages.calls == []


async def test_gate_marks_unverified_when_generated_test_does_not_fail_then_pass(
    head_sha, repo
) -> None:
    messages = FakeMessages(
        parsed_outputs=[GeneratedTest(test_code=GENTEST_PASSES_BOTH)]
    )
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(make_finding(), head_sha)

    assert decision.decision == Decision.ACCEPTED
    assert decision.reason == GateReason.TESTS_PASSED
    assert decision.verification == "unverified"
    assert_no_worktrees(repo)


async def test_gate_marks_unverified_when_llm_call_fails(head_sha, repo) -> None:
    from anthropic import NotFoundError

    from utils import make_status_error

    messages = FakeMessages(error=make_status_error(NotFoundError, status_code=404))
    gate = make_gate(repo, messages)
    decision = await gate.gate_finding(make_finding(), head_sha)

    assert decision.decision == Decision.ACCEPTED
    assert decision.verification == "unverified"
    assert_no_worktrees(repo)


async def test_gate_tears_down_worktree_on_timeout(tmp_path: Path) -> None:
    slow_repo = tmp_path / "slow-repo"
    slow_head = build_slow_fixture_repo(slow_repo)
    messages = FakeMessages()
    gate = make_gate(slow_repo, messages, gate_timeout_seconds=2)
    decision = await gate.gate_finding(make_finding(), slow_head)

    assert decision.decision == Decision.REJECTED
    assert decision.reason == GateReason.TESTS_FAILED
    assert_no_worktrees(slow_repo)


async def test_gate_runs_concurrently_with_bounded_semaphore(head_sha, repo) -> None:
    gate = make_gate(repo, FakeMessages(), gate_concurrency=2)

    state = {"active": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_pytest(worktree, *, extra_args=None, with_coverage=True):
        async with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        await asyncio.sleep(0.05)
        async with lock:
            state["active"] -= 1
        return PytestOutcome(passed=3, failed=0, coverage=100.0, duration_ms=1)

    async def fake_verify(worktree, finding, patch):
        return "verified"

    gate._run_pytest = fake_pytest
    gate._verify_with_generated_test = fake_verify

    findings = [make_finding() for _ in range(6)]
    pairs = await gate.gate_all(findings, head_sha)

    assert len(pairs) == 6
    assert all(d.decision == Decision.ACCEPTED for _, d in pairs)
    assert state["max"] == 2
    assert_no_worktrees(repo)


async def test_gate_all_fires_every_reason(head_sha, repo) -> None:
    messages = FakeMessages(
        parsed_outputs=[GeneratedTest(test_code=GENTEST_FAIL_THEN_PASS)]
    )
    gate = make_gate(repo, messages)
    findings = [
        make_finding(),
        make_finding(suggested_patch=PATCH_TESTS_FAIL),
        make_finding(suggested_patch=PATCH_COVERAGE_DROP),
        make_finding(suggested_patch=PATCH_DOES_NOT_APPLY),
        make_finding(suggested_patch=None),
    ]
    pairs = await gate.gate_all(findings, head_sha)

    reasons = {decision.reason for _, decision in pairs}
    assert reasons == {
        GateReason.TESTS_PASSED,
        GateReason.TESTS_FAILED,
        GateReason.COVERAGE_DROPPED,
        GateReason.PATCH_DID_NOT_APPLY,
        GateReason.NO_PATCH,
    }
    assert_no_worktrees(repo)
