import json
from decimal import Decimal

import httpx
import pytest
from anthropic import APIConnectionError, APIStatusError, NotFoundError, RateLimitError

from agentdiff.config import Settings
from agentdiff.models import Category, Severity
from agentdiff.reviewer import (
    LLMError,
    MAX_TOKENS,
    PROMPT_VERSION,
    ReviewFinding,
    ReviewOutcome,
    ReviewResult,
    Reviewer,
    compute_cost_usd,
)
from utils import (
    FakeAnthropic,
    FakeMessages,
    make_settings,
    make_status_error,
    make_validation_error,
)

DIFF = "diff --git a/src/app.py b/src/app.py\n@@ -1,3 +1,4 @@\n foo\n+bar\n"

FINDING = ReviewFinding(
    file_path="src/app.py",
    start_line=10,
    end_line=12,
    severity=Severity.MAJOR,
    category=Category.CORRECTNESS,
    title="Unhandled exception on empty input",
    rationale="Calling parse() with an empty string raises ValueError and kills the worker.",
    confidence=0.9,
    suggested_patch=None,
)


async def run_review(fake: FakeAnthropic, **settings_overrides) -> ReviewOutcome:
    reviewer = Reviewer(make_settings(**settings_overrides), client=fake)
    outcome = await reviewer.review(diff=DIFF)
    return outcome


async def test_review_returns_validated_findings_and_usage() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[FINDING]))
    outcome = await run_review(FakeAnthropic(messages))

    assert outcome.result.findings == [FINDING]
    assert outcome.input_tokens == 1000
    assert outcome.output_tokens == 200
    assert outcome.cache_read_tokens == 500
    assert outcome.estimated_input_tokens == 42
    assert outcome.model_id == "claude-opus-5"


def test_compute_cost_usd() -> None:
    assert compute_cost_usd("claude-opus-5", 1000, 200) == Decimal("0.010000")
    assert compute_cost_usd("claude-sonnet-5", 0, 0) == Decimal("0.000000")
    assert compute_cost_usd("unknown-model", 1_000_000, 1_000_000) == Decimal("0.000000")


async def test_review_uses_settings_model_id() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    outcome = await run_review(
        FakeAnthropic(messages), model_id="claude-sonnet-5"
    )

    assert outcome.model_id == "claude-sonnet-5"
    assert messages.calls[-1][1]["model"] == "claude-sonnet-5"


async def test_review_sends_structured_request_shape() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    await run_review(FakeAnthropic(messages))

    kwargs = [kw for name, kw in messages.calls if name == "parse"][0]
    assert kwargs["max_tokens"] == MAX_TOKENS
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in kwargs["thinking"]
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["output_format"] is ReviewResult
    assert kwargs["messages"] == [{"role": "user", "content": DIFF}]
    system_block = kwargs["system"][0]
    assert system_block["cache_control"] == {"type": "ephemeral"}
    assert f"PROMPT_VERSION: {PROMPT_VERSION}" in system_block["text"]


async def test_review_preflight_count_tokens_runs_first() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    await run_review(FakeAnthropic(messages))

    assert messages.calls[0][0] == "count_tokens"
    count_kwargs = messages.calls[0][1]
    assert count_kwargs["model"] == "claude-opus-5"
    assert count_kwargs["messages"] == [{"role": "user", "content": DIFF}]
    assert count_kwargs["thinking"] == {"type": "adaptive"}


async def test_review_includes_repo_context_in_system_prompt() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    reviewer = Reviewer(make_settings(), client=FakeAnthropic(messages))
    await reviewer.review(diff=DIFF, repo_context="Always use tabs for indentation.")

    kwargs = [kw for name, kw in messages.calls if name == "parse"][0]
    assert "Always use tabs for indentation." in kwargs["system"][0]["text"]


async def test_review_unknown_model_cost_is_zero() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    outcome = await run_review(FakeAnthropic(messages), model_id="not-priced-model")

    assert outcome.cost_usd == Decimal("0.000000")


async def test_review_model_not_found_raises_llm_error() -> None:
    messages = FakeMessages(error=make_status_error(NotFoundError, status_code=404))
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502


async def test_review_rate_limit_raises_429() -> None:
    messages = FakeMessages(error=make_status_error(RateLimitError, status_code=429))
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 429


async def test_review_api_status_error_raises_502() -> None:
    messages = FakeMessages(error=make_status_error(APIStatusError, status_code=500))
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502


async def test_review_connection_error_raises_502() -> None:
    error = APIConnectionError(
        request=httpx.Request("POST", "http://api.test/v1/messages")
    )
    messages = FakeMessages(error=error)
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502


async def test_review_malformed_output_raises_502() -> None:
    messages = FakeMessages(error=make_validation_error())
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502
    assert "malformed" in str(excinfo.value)


async def test_review_no_structured_output_raises_502() -> None:
    messages = FakeMessages(parsed_output=None)
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502


async def test_review_fallback_without_parse_uses_create() -> None:
    payload = ReviewResult(findings=[FINDING]).model_dump_json()
    messages = FakeMessages(message_text=payload)
    del messages.parse
    outcome = await run_review(FakeAnthropic(messages))

    assert outcome.result.findings == [FINDING]
    create_kwargs = [kw for name, kw in messages.calls if name == "create"][0]
    assert create_kwargs["output_config"]["effort"] == "high"
    assert create_kwargs["output_config"]["format"]["type"] == "json_schema"
    assert create_kwargs["output_config"]["format"]["schema"]["title"] == "ReviewResult"


async def test_review_fallback_empty_response_raises() -> None:
    messages = FakeMessages(message_text="   ")
    del messages.parse
    with pytest.raises(LLMError):
        await run_review(FakeAnthropic(messages))


async def test_review_fallback_malformed_json_raises_502() -> None:
    messages = FakeMessages(message_text='{"findings": "not-a-list"}')
    del messages.parse
    with pytest.raises(LLMError) as excinfo:
        await run_review(FakeAnthropic(messages))
    assert excinfo.value.status_code == 502


async def test_review_latency_is_measured_with_monotonic_clock() -> None:
    messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
    outcome = await run_review(FakeAnthropic(messages))

    assert isinstance(outcome.latency_ms, int)
    assert outcome.latency_ms >= 0


async def test_reviewer_lazy_client_uses_env_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    settings = Settings()
    reviewer = Reviewer(settings)

    client = reviewer._get_client()
    assert client.api_key == "env-key"


async def test_review_against_real_sdk_with_mocked_transport() -> None:
    import httpx2
    from anthropic import AsyncAnthropic

    captured: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 123}, request=request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": '{"findings": []}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 500,
                },
            },
            request=request,
        )

    real_client = AsyncAnthropic(
        api_key="k",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    reviewer = Reviewer(make_settings(), client=real_client)
    outcome = await reviewer.review(diff=DIFF)

    assert outcome.result == ReviewResult(findings=[])
    assert outcome.input_tokens == 1000
    assert outcome.output_tokens == 200
    assert outcome.cache_read_tokens == 500
    assert outcome.estimated_input_tokens == 123

    messages_request = next(r for r in captured if r.url.path == "/v1/messages")
    body = json.loads(messages_request.content)
    assert body["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in body["thinking"]
    assert body["max_tokens"] == MAX_TOKENS
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
