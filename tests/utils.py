import os
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


class FakeMessages:
    def __init__(
        self,
        *,
        parsed_output: Any = None,
        message_text: str | None = None,
        error: Exception | None = None,
        usage: SimpleNamespace | None = None,
        counts: int = 42,
    ) -> None:
        self._parsed_output = parsed_output
        self._message_text = message_text
        self._error = error
        self._usage = usage if usage is not None else default_usage()
        self._counts = counts
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.parse = self._parse
        self.create = self._create
        self.count_tokens = self._count_tokens

    async def _parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("parse", kwargs))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(parsed_output=self._parsed_output, usage=self._usage)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("create", kwargs))
        if self._error is not None:
            raise self._error
        content = [SimpleNamespace(type="text", text=self._message_text or "")]
        return SimpleNamespace(content=content, usage=self._usage)

    async def _count_tokens(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(("count_tokens", kwargs))
        if self._error is not None:
            raise self._error
        return SimpleNamespace(input_tokens=self._counts)


class FakeAnthropic:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages
