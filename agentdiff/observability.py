import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterator

from starlette.middleware.base import BaseHTTPMiddleware

CORRELATION_ID_HEADER = "x-correlation-id"

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_model_id: ContextVar[str | None] = ContextVar("model", default=None)
_prompt_version: ContextVar[str | None] = ContextVar("prompt_version", default=None)

_CONTEXT_FIELDS = {
    "correlation_id": _correlation_id,
    "run_id": _run_id,
    "model": _model_id,
    "prompt_version": _prompt_version,
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, context_var in _CONTEXT_FIELDS.items():
            value = getattr(record, key, None)
            if value is None:
                value = context_var.get()
            if value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("agentdiff")
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def review_log_context(
    run_id: object | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> Iterator[None]:
    run_token = _run_id.set(str(run_id) if run_id is not None else None)
    model_token = _model_id.set(model)
    prompt_token = _prompt_version.set(prompt_version)
    try:
        yield
    finally:
        _run_id.reset(run_token)
        _model_id.reset(model_token)
        _prompt_version.reset(prompt_token)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or uuid.uuid4().hex[:12]
        token = _correlation_id.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            _correlation_id.reset(token)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
