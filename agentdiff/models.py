import enum
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Severity(str, enum.Enum):
    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    NIT = "nit"


class Category(str, enum.Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TEST_GAP = "test-gap"


class Decision(str, enum.Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class GateReason(str, enum.Enum):
    TESTS_PASSED = "tests_passed"
    TESTS_FAILED = "tests_failed"
    COVERAGE_DROPPED = "coverage_dropped"
    PATCH_DID_NOT_APPLY = "patch_did_not_apply"
    NO_PATCH = "no_patch"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


def _enum(enum_cls: type[enum.Enum], name: str, length: int) -> SAEnum:
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=False,
        length=length,
        create_constraint=False,
        values_callable=_enum_values,
    )


class ReviewRun(Base):
    __tablename__ = "review_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    repo: Mapped[str] = mapped_column(String(255))
    base_sha: Mapped[str] = mapped_column(String(64))
    head_sha: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(64), server_default=text("'1'"))
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "runstatus", 32), default=RunStatus.PENDING, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    findings: Mapped[list["Finding"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Finding.start_line"
    )


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("review_runs.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(512))
    start_line: Mapped[int | None] = mapped_column(Integer)
    end_line: Mapped[int | None] = mapped_column(Integer)
    severity: Mapped[Severity] = mapped_column(_enum(Severity, "severity", 16))
    category: Mapped[Category] = mapped_column(_enum(Category, "category", 32))
    title: Mapped[str] = mapped_column(String(512))
    rationale: Mapped[str] = mapped_column(Text)
    suggested_patch: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["ReviewRun"] = relationship(back_populates="findings")
    gate_result: Mapped["GateResult | None"] = relationship(
        back_populates="finding", uselist=False, cascade="all, delete-orphan"
    )


class GateResult(Base):
    __tablename__ = "gate_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), unique=True
    )
    decision: Mapped[Decision] = mapped_column(_enum(Decision, "decision", 16))
    reason: Mapped[GateReason] = mapped_column(_enum(GateReason, "gatereason", 32))
    tests_passed: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    tests_failed: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    coverage_before: Mapped[float | None] = mapped_column(Float)
    coverage_after: Mapped[float | None] = mapped_column(Float)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    verification: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    finding: Mapped["Finding"] = relationship(back_populates="gate_result")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    model_id: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(64))
    effort: Mapped[str] = mapped_column(String(32))
    true_positives: Mapped[int] = mapped_column(Integer)
    false_positives: Mapped[int] = mapped_column(Integer)
    false_negatives: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)
    f1: Mapped[float] = mapped_column(Float)
    mean_latency_ms: Mapped[float] = mapped_column(Float)
    mean_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    clean_false_positive_rate: Mapped[float] = mapped_column(Float)
    total_cases: Mapped[int] = mapped_column(Integer)
    clean_cases: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
