VENV ?= .venv
PYTEST ?= $(VENV)/bin/pytest

.PHONY: install up down test cov migrate

install:
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

up:
	docker compose up --build

down:
	docker compose down

test:
	$(PYTEST)

cov:
	$(PYTEST) --cov=agentdiff --cov-report=term-missing --cov-fail-under=80

migrate:
	$(VENV)/bin/alembic upgrade head
