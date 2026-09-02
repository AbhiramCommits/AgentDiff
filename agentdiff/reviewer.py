import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import TypeVar

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    NotFoundError,
    RateLimitError,
)
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .models import Category, Severity

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)

PROMPT_VERSION = "v1"
MAX_TOKENS = 16000

MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

SYSTEM_PROMPT = f"""You are an expert software engineer performing a rigorous code review of a git diff.
PROMPT_VERSION: {PROMPT_VERSION}

Analyze the provided unified git diff for concrete, actionable defects only.
Return a JSON object containing a "findings" array. Each finding must contain:
- "file_path": the path of the affected file
- "start_line" and "end_line": the line range in the new file (omit when not applicable)
- "severity": one of "blocker", "major", "minor", "nit"
- "category": one of "correctness", "security", "performance", "maintainability", "test-gap"
- "title": a one-sentence title
- "rationale": an explanation naming the concrete failure scenario this defect causes
- "confidence": a float between 0.0 and 1.0
- "suggested_patch": only when the fix is mechanical, a minimal unified diff with correct
  hunk headers that applies cleanly to the head SHA; omit it otherwise

Report only findings you are reasonably confident about. An empty "findings" array is a
valid response. Do not restate the diff and do not emit any prose outside the JSON object."""


class ReviewFinding(BaseModel):
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    severity: Severity
    category: Category
    title: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_patch: str | None = None


class ReviewResult(BaseModel):
    findings: list[ReviewFinding]


class LLMError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ReviewOutcome:
    result: ReviewResult
    model_id: str
    estimated_input_tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: Decimal
    latency_ms: int


def compute_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> Decimal:
    input_per_mtok, output_per_mtok = MODEL_PRICING.get(model_id, (0.0, 0.0))
    total_mtok_usd = Decimal(input_tokens) * Decimal(str(input_per_mtok)) + Decimal(
        output_tokens
    ) * Decimal(str(output_per_mtok))
    return (total_mtok_usd / Decimal(1_000_000)).quantize(Decimal("0.000001"))


class Reviewer:
    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> AsyncAnthropic:
        return self._get_client()

    def _get_client(self) -> AsyncAnthropic:
        if self._client is not None:
            return self._client
        if self.settings.anthropic_api_key:
            return AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return AsyncAnthropic()

    def _build_system_text(self, repo_context: str | None) -> str:
        text = SYSTEM_PROMPT
        if repo_context:
            text += (
                "\n\nRepository conventions provided by the caller. Take them into account "
                f"when judging findings and writing patches:\n{repo_context}"
            )
        return text

    def _system_blocks(self, text: str) -> list[dict[str, object]]:
        return [
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def _estimate_tokens(
        self,
        client: AsyncAnthropic,
        model_id: str,
        system_blocks: list[dict[str, object]],
        messages: list[dict[str, str]],
    ) -> int:
        count = await client.messages.count_tokens(
            model=model_id,
            system=system_blocks,
            messages=messages,
            thinking={"type": "adaptive"},
        )
        return count.input_tokens

    async def structured_call(
        self,
        response_model: type[ResponseT],
        system_text: str,
        user_content: str,
        max_tokens: int,
        effort: str | None = None,
    ) -> tuple[ResponseT, object]:
        model_id = self.settings.model_id
        client = self.client
        system_blocks = self._system_blocks(system_text)
        messages: list[dict[str, str]] = [{"role": "user", "content": user_content}]
        thinking = {"type": "adaptive"}
        effort = effort or self.settings.effort
        try:
            if hasattr(client.messages, "parse"):
                parsed = await client.messages.parse(
                    model=model_id,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    output_config={"effort": effort},
                    output_format=response_model,
                    system=system_blocks,
                    messages=messages,
                )
                result = parsed.parsed_output
                if result is None:
                    raise LLMError("model returned no structured output")
                return result, parsed.usage

            response = await client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                thinking=thinking,
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": response_model.model_json_schema(),
                    },
                    "effort": effort,
                },
                system=system_blocks,
                messages=messages,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            if not text.strip():
                raise LLMError("model returned an empty response")
            result = response_model.model_validate_json(text)
            return result, response.usage
        except NotFoundError as exc:
            raise LLMError(f"model '{model_id}' was not found by the API", 502) from exc
        except RateLimitError as exc:
            raise LLMError(f"Anthropic API rate limit exceeded: {exc}", 429) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"Anthropic API error {exc.status_code}: {exc.message}", 502
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(f"could not connect to the Anthropic API: {exc}", 502) from exc
        except ValidationError as exc:
            raise LLMError("model returned malformed structured output", 502) from exc

    async def review(self, diff: str, repo_context: str | None = None) -> ReviewOutcome:
        started = time.monotonic()
        model_id = self.settings.model_id
        system_text = self._build_system_text(repo_context)
        client = self.client
        system_blocks = self._system_blocks(system_text)
        messages: list[dict[str, str]] = [{"role": "user", "content": diff}]

        try:
            estimated_input_tokens = await self._estimate_tokens(
                client, model_id, system_blocks, messages
            )
        except NotFoundError as exc:
            raise LLMError(f"model '{model_id}' was not found by the API", 502) from exc
        except RateLimitError as exc:
            raise LLMError(f"Anthropic API rate limit exceeded: {exc}", 429) from exc
        except APIStatusError as exc:
            raise LLMError(
                f"Anthropic API error {exc.status_code}: {exc.message}", 502
            ) from exc
        except APIConnectionError as exc:
            raise LLMError(f"could not connect to the Anthropic API: {exc}", 502) from exc

        result, usage = await self.structured_call(ReviewResult, system_text, diff, MAX_TOKENS)

        latency_ms = int((time.monotonic() - started) * 1000)
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read_tokens = usage.cache_read_input_tokens
        cost_usd = compute_cost_usd(model_id, input_tokens, output_tokens)

        logger.info(
            "review complete model=%s findings=%d tokens_in=%d tokens_out=%d cache_read=%d cost_usd=%s latency_ms=%d",
            model_id,
            len(result.findings),
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cost_usd,
            latency_ms,
        )
        return ReviewOutcome(
            result=result,
            model_id=model_id,
            estimated_input_tokens=estimated_input_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
