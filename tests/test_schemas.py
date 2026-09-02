import pytest
from pydantic import ValidationError

from agentdiff.models import Category, Decision, GateReason, Severity
from agentdiff.schemas import FindingCreate, GateResultCreate, ReviewRunCreate


def test_review_run_create_defaults() -> None:
    run = ReviewRunCreate(repo="acme/widgets", base_sha="0" * 40, head_sha="1" * 40)

    assert run.model_id is None
    assert run.prompt_version == "1"


def test_finding_create_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        FindingCreate(
            file_path="src/app.py",
            severity=Severity.NIT,
            category=Category.MAINTAINABILITY,
            title="Nit",
            rationale="Whatever",
            confidence=1.5,
        )


def test_gate_result_create_defaults() -> None:
    gate = GateResultCreate(decision=Decision.ACCEPTED, reason=GateReason.TESTS_PASSED)

    assert gate.tests_passed == 0
    assert gate.tests_failed == 0
    assert gate.coverage_before is None
    assert gate.coverage_after is None
    assert gate.duration_ms is None
