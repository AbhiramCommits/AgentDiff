import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from pydantic import ValidationError

from agentdiff.config import Settings
from agentdiff.reviewer import ReviewResult

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://localhost:5432/agentdiff_test",
)

FIXTURE_CORE = """\
def divide(a, b):
    if b == 0:
        return 0.0
    return a / b
"""

FIXTURE_TESTS = """\
from mathlib.core import divide


def test_divide_basic():
    assert divide(10, 2) == 5.0


def test_divide_negative():
    assert divide(-10, 2) == -5.0


def test_divide_by_one():
    assert divide(7, 1) == 7.0
"""

PATCH_ACCEPT = """\
--- a/mathlib/core.py
+++ b/mathlib/core.py
@@ -2,3 +2,3 @@
     if b == 0:
-        return 0.0
+        raise ValueError("division by zero")
     return a / b
"""

PATCH_TESTS_FAIL = """\
--- a/mathlib/core.py
+++ b/mathlib/core.py
@@ -3,2 +3,2 @@
         return 0.0
-    return a / b
+    return b / a
"""

PATCH_COVERAGE_DROP = """\
--- a/mathlib/core.py
+++ b/mathlib/core.py
@@ -3,2 +3,5 @@
         return 0.0
     return a / b
+
+def never_called_helper():
+    return 123
"""

PATCH_DOES_NOT_APPLY = """\
--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1 +1 @@
-old
+new
"""

GENTEST_FAIL_THEN_PASS = """\
import pytest

from mathlib.core import divide


def test_generated_divide_by_zero_raises():
    with pytest.raises(ValueError):
        divide(1, 0)
"""

GENTEST_PASSES_BOTH = """\
from mathlib.core import divide


def test_generated_smoke():
    assert divide(4, 2) == 2.0
"""


def make_settings(**overrides: Any) -> Settings:
    return Settings(anthropic_api_key="test-key", **overrides)


def default_usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500)


def make_validation_error() -> ValidationError:
    try:
        ReviewResult.model_validate({"findings": "nope"})
    except ValidationError as exc:
        return exc
    raise AssertionError("unreachable")


def make_status_error(
    error_cls: type[httpx.HTTPError], status_code: int = 404
) -> Exception:
    request = httpx.Request("POST", "http://api.test/v1/messages")
    return error_cls(
        f"status {status_code}",
        response=httpx.Response(status_code, request=request),
        body=None,
    )


def run_cmd(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def build_fixture_repo(root: Path) -> str:
    (root / "mathlib").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "mathlib" / "__init__.py").write_text("")
    (root / "mathlib" / "core.py").write_text(FIXTURE_CORE)
    (root / "tests" / "test_core.py").write_text(FIXTURE_TESTS)
    (root / "conftest.py").write_text("")
    run_cmd("git", "init", "-q", cwd=root)
    run_cmd("git", "config", "user.email", "test@example.com", cwd=root)
    run_cmd("git", "config", "user.name", "Test", cwd=root)
    run_cmd("git", "add", ".", cwd=root)
    run_cmd("git", "commit", "-q", "-m", "initial", cwd=root)
    return run_cmd("git", "rev-parse", "HEAD", cwd=root).strip()


def build_slow_fixture_repo(root: Path) -> str:
    build_fixture_repo(root)
    (root / "tests" / "test_slow.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"
    )
    run_cmd("git", "add", ".", cwd=root)
    run_cmd("git", "commit", "-q", "-m", "slow test", cwd=root)
    return run_cmd("git", "rev-parse", "HEAD", cwd=root).strip()


class FakeMessages:
    def __init__(
        self,
        *,
        parsed_output: Any = None,
        parsed_outputs: list[Any] | None = None,
        message_text: str | None = None,
        error: Exception | None = None,
        usage: SimpleNamespace | None = None,
        counts: int = 42,
    ) -> None:
        self._parsed_output = parsed_output
        self._queue: list[Any] = list(parsed_outputs or [])
        self._message_text = message_text
        self._error = error
        self._usage = usage if usage is not None else default_usage()
        self._counts = counts
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.parse = self._parse
        self.create = self._create
        self.count_tokens = self._count_tokens

    def _next_parsed(self) -> Any:
        if self._queue:
            return self._queue.pop(0)
        return self._parsed_output

    async def _parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("parse", kwargs))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(parsed_output=self._next_parsed(), usage=self._usage)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("create", kwargs))
        if self._error is not None:
            raise self._error
        content = [SimpleNamespace(type="text", text=self._message_text or "")]
        return SimpleNamespace(content=content, usage=self._usage)

    async def _count_tokens(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("count_tokens", kwargs))
        return SimpleNamespace(input_tokens=self._counts)


class FakeAnthropic:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages
