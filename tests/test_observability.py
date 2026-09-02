import io
import json
import logging
import uuid

from httpx import ASGITransport, AsyncClient

from agentdiff.main import create_app
from agentdiff.observability import (
    _correlation_id,
    JsonFormatter,
    review_log_context,
)
from agentdiff.reviewer import ReviewResult, Reviewer
from utils import (
    TEST_DATABASE_URL,
    FakeAnthropic,
    FakeMessages,
    make_settings,
)


def make_record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        "agentdiff.test", logging.INFO, "path", 10, message, None, None
    )


def test_json_formatter_includes_context_fields() -> None:
    formatter = JsonFormatter()
    with review_log_context(
        run_id="run-1", model="claude-opus-5", prompt_version="v1"
    ):
        entry = json.loads(formatter.format(make_record("hello world")))

    assert entry["message"] == "hello world"
    assert entry["run_id"] == "run-1"
    assert entry["model"] == "claude-opus-5"
    assert entry["prompt_version"] == "v1"
    assert entry["level"] == "INFO"
    assert "timestamp" in entry


def test_json_formatter_includes_correlation_id() -> None:
    formatter = JsonFormatter()
    token = _correlation_id.set("cid-123")
    try:
        entry = json.loads(formatter.format(make_record("with cid")))
    finally:
        _correlation_id.reset(token)

    assert entry["correlation_id"] == "cid-123"


def test_json_formatter_includes_exception() -> None:
    import sys

    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "agentdiff.test", logging.ERROR, "path", 10, "failed", None, None
        )
        record.exc_info = sys.exc_info()
        entry = json.loads(formatter.format(record))

    assert "exception" in entry
    assert "ValueError: boom" in entry["exception"]


async def test_middleware_echoes_correlation_id(db_engine) -> None:
    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/healthz", headers={"x-correlation-id": "abc123"})
    finally:
        await app.state.engine.dispose()

    assert resp.status_code == 200
    assert resp.headers["x-correlation-id"] == "abc123"


async def test_middleware_generates_correlation_id(db_engine) -> None:
    app = create_app(make_settings(database_url=TEST_DATABASE_URL))
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.get("/healthz")
    finally:
        await app.state.engine.dispose()

    generated = resp.headers["x-correlation-id"]
    assert generated
    assert len(generated) >= 12


async def test_review_log_line_carries_run_id_and_model() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("agentdiff")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        messages = FakeMessages(parsed_output=ReviewResult(findings=[]))
        reviewer = Reviewer(make_settings(), client=FakeAnthropic(messages))
        run_id = uuid.uuid4()
        await reviewer.review(diff="diff --git a/x b/x\n", run_id=run_id)

        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        review_lines = [l for l in lines if "review complete" in l.get("message", "")]
        assert review_lines
        assert review_lines[0]["run_id"] == str(run_id)
        assert review_lines[0]["model"] == "claude-opus-5"
        assert review_lines[0]["prompt_version"] == "v1"
    finally:
        logger.removeHandler(handler)
