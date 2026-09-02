FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY agentdiff ./agentdiff
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn agentdiff.main:app --host 0.0.0.0 --port 8000"]
