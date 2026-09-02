import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from .config import Settings
from .metrics import COVERAGE_DELTA, GATE_DURATION, SUGGESTIONS
from .models import Decision, Finding, GateReason
from .observability import review_log_context
from .reviewer import LLMError, PROMPT_VERSION, Reviewer

logger = logging.getLogger(__name__)

GENERATED_TEST_PROMPT_VERSION = "v1"
GENERATED_TEST_MAX_TOKENS = 8000
CONFIDENCE_DOWNGRADE_FACTOR = 0.5

GENERATED_TEST_SYSTEM_PROMPT = f"""You are an expert test author verifying a bug-fix patch.
PROMPT_VERSION: {GENERATED_TEST_PROMPT_VERSION}

Given a patch and its rationale, write a single pytest test function that:
- fails when run against the ORIGINAL, unpatched code
- passes when run against the PATCHED code
- reproduces the concrete failure scenario described in the rationale

Requirements:
- Use plain pytest; import the project module under test directly.
- Do not import anything from outside the repository.
- The test must be self-contained in one function.
Return a JSON object with a single key "test_code" containing the test source as a
string. Emit no prose outside the JSON object."""


class GeneratedTest(BaseModel):
    test_code: str


class GateTimeoutError(Exception):
    pass


class GateExecutionError(Exception):
    pass


@dataclass(frozen=True)
class PytestOutcome:
    passed: int
    failed: int
    coverage: float | None
    duration_ms: int


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    reason: GateReason
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_before: float | None = None
    coverage_after: float | None = None
    duration_ms: int = 0
    verification: str | None = None


def _parse_count(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


class Gate:
    def __init__(self, settings: Settings, reviewer: Reviewer):
        self.settings = settings
        self.reviewer = reviewer
        self._semaphore = asyncio.Semaphore(settings.gate_concurrency)

    async def gate_all(
        self, findings: list[Finding], head_sha: str
    ) -> list[tuple[Finding, GateDecision]]:
        async def run_one(finding: Finding) -> tuple[Finding, GateDecision]:
            async with self._semaphore:
                return finding, await self.gate_finding(finding, head_sha)

        return list(await asyncio.gather(*(run_one(f) for f in findings)))

    async def gate_finding(self, finding: Finding, head_sha: str) -> GateDecision:
        started = time.monotonic()
        with review_log_context(
            run_id=finding.run_id,
            model=self.settings.model_id,
            prompt_version=PROMPT_VERSION,
        ):
            decision = await self._gate_finding(finding, head_sha)
        GATE_DURATION.observe(time.monotonic() - started)
        SUGGESTIONS.labels(
            decision=decision.decision.value, reason=decision.reason.value
        ).inc()
        if decision.coverage_before is not None and decision.coverage_after is not None:
            COVERAGE_DELTA.observe(
                decision.coverage_after - decision.coverage_before
            )
        return decision

    async def _gate_finding(self, finding: Finding, head_sha: str) -> GateDecision:
        patch = finding.suggested_patch
        if not patch:
            return GateDecision(
                decision=Decision.REJECTED, reason=GateReason.NO_PATCH
            )

        worktree = self.settings.workspace_dir / "agentdiff-worktrees" / uuid4().hex
        started = time.monotonic()
        try:
            worktree.parent.mkdir(parents=True, exist_ok=True)
            await self._git(
                self.settings.workspace_dir,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                head_sha,
            )
            baseline = await self._run_pytest(worktree)
            if baseline.failed > 0:
                return GateDecision(
                    decision=Decision.REJECTED,
                    reason=GateReason.TESTS_FAILED,
                    tests_passed=baseline.passed,
                    tests_failed=baseline.failed,
                    coverage_before=baseline.coverage,
                    duration_ms=self._elapsed(started),
                )

            patch_file = worktree / "__agentdiff.patch"
            patch_file.write_text(patch)
            try:
                await self._git(worktree, "apply", "--3way", "__agentdiff.patch")
            except GateExecutionError:
                return GateDecision(
                    decision=Decision.REJECTED,
                    reason=GateReason.PATCH_DID_NOT_APPLY,
                    coverage_before=baseline.coverage,
                    duration_ms=self._elapsed(started),
                )

            after = await self._run_pytest(worktree)
            duration_ms = self._elapsed(started)
            if after.failed > 0:
                return GateDecision(
                    decision=Decision.REJECTED,
                    reason=GateReason.TESTS_FAILED,
                    tests_passed=after.passed,
                    tests_failed=after.failed,
                    coverage_before=baseline.coverage,
                    coverage_after=after.coverage,
                    duration_ms=duration_ms,
                )

            coverage_before = baseline.coverage if baseline.coverage is not None else 0.0
            coverage_after = after.coverage if after.coverage is not None else 0.0
            if coverage_after < coverage_before - self.settings.coverage_tolerance:
                return GateDecision(
                    decision=Decision.REJECTED,
                    reason=GateReason.COVERAGE_DROPPED,
                    tests_passed=after.passed,
                    tests_failed=after.failed,
                    coverage_before=baseline.coverage,
                    coverage_after=after.coverage,
                    duration_ms=duration_ms,
                )

            verification = await self._verify_with_generated_test(
                worktree, finding, patch
            )
            return GateDecision(
                decision=Decision.ACCEPTED,
                reason=GateReason.TESTS_PASSED,
                tests_passed=after.passed,
                tests_failed=after.failed,
                coverage_before=baseline.coverage,
                coverage_after=after.coverage,
                duration_ms=duration_ms,
                verification=verification,
            )
        except GateTimeoutError:
            return GateDecision(
                decision=Decision.REJECTED,
                reason=GateReason.TESTS_FAILED,
                duration_ms=self._elapsed(started),
            )
        finally:
            await self._worktree_remove(worktree)

    async def _verify_with_generated_test(
        self, worktree: Path, finding: Finding, patch: str
    ) -> str:
        try:
            generated, _ = await self.reviewer.structured_call(
                GeneratedTest,
                GENERATED_TEST_SYSTEM_PROMPT,
                self._verification_prompt(finding, patch),
                GENERATED_TEST_MAX_TOKENS,
            )
        except LLMError:
            return "unverified"

        try:
            test_file = worktree / "tests" / "test_agentdiff_generated.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(generated.test_code)
            test_arg = str(test_file.relative_to(worktree))
            patched = await self._run_pytest(
                worktree, extra_args=[test_arg], with_coverage=False
            )
            await self._git(worktree, "checkout", "HEAD", "--", ".")
            original = await self._run_pytest(
                worktree, extra_args=[test_arg], with_coverage=False
            )
        except (GateTimeoutError, GateExecutionError):
            return "unverified"

        if patched.failed == 0 and original.failed > 0:
            return "verified"
        return "unverified"

    def _verification_prompt(self, finding: Finding, patch: str) -> str:
        return (
            f"File: {finding.file_path}\n"
            f"Title: {finding.title}\n"
            f"Rationale: {finding.rationale}\n\n"
            f"Patch:\n{patch}\n\n"
            "Write the pytest test that demonstrates this fix."
        )

    async def _run_pytest(
        self,
        worktree: Path,
        *,
        extra_args: list[str] | None = None,
        with_coverage: bool = True,
    ) -> PytestOutcome:
        for cache_dir in worktree.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)

        cmd = [sys.executable, "-m", "pytest", "-q"]
        if with_coverage:
            cmd += [
                f"--cov={self.settings.coverage_package}",
                "--cov-report=json:cov.json",
            ]
        if extra_args:
            cmd += extra_args

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=worktree,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.settings.gate_timeout_seconds
            )
        except asyncio.TimeoutError:
            self._kill_process(proc)
            await proc.wait()
            raise GateTimeoutError(
                f"pytest exceeded {self.settings.gate_timeout_seconds}s in {worktree}"
            ) from None

        text = stdout.decode(errors="replace")
        passed = _parse_count(text, r"(\d+) passed")
        failed = _parse_count(text, r"(\d+) failed")
        if proc.returncode != 0 and failed == 0:
            failed = 1

        coverage: float | None = None
        if with_coverage:
            cov_file = worktree / "cov.json"
            if cov_file.exists():
                try:
                    coverage = float(
                        json.loads(cov_file.read_text())["totals"]["percent_covered"]
                    )
                except (KeyError, TypeError, ValueError):
                    coverage = None

        return PytestOutcome(
            passed=passed,
            failed=failed,
            coverage=coverage,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    @staticmethod
    def _kill_process(proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.monotonic() - started) * 1000)

    async def _git(self, repo_dir: Path, *args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo_dir),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise GateExecutionError(
                f"git {' '.join(args)} failed: {stdout.decode(errors='replace').strip()}"
            )

    async def _worktree_remove(self, worktree: Path) -> None:
        try:
            await self._git(
                self.settings.workspace_dir, "worktree", "remove", "--force", str(worktree)
            )
        except Exception as exc:
            logger.warning("failed to remove worktree %s: %s", worktree, exc)
        if worktree.exists():
            shutil.rmtree(worktree, ignore_errors=True)
