from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .models import Category, Decision, GateReason, RunStatus, Severity


class HealthResponse(BaseModel):
    status: str
    db: str


class ReviewCreateRequest(BaseModel):
    repo: str
    base_sha: str
    head_sha: str
    diff: str
    repo_context: str | None = None


class ReviewRunCreate(BaseModel):
    repo: str
    base_sha: str
    head_sha: str
    model_id: str | None = None
    prompt_version: str = "1"


class FindingCreate(BaseModel):
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    severity: Severity
    category: Category
    title: str
    rationale: str
    suggested_patch: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class FindingRead(FindingCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    run_id: UUID
    created_at: datetime
    gate_result: "GateResultRead | None" = None


class ReviewRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    repo: str
    base_sha: str
    head_sha: str
    model_id: str
    prompt_version: str
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cost_usd: Decimal | None = None
    findings: list[FindingRead] = Field(default_factory=list)


class GateResultCreate(BaseModel):
    decision: Decision
    reason: GateReason
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_before: float | None = None
    coverage_after: float | None = None
    duration_ms: int | None = None


class GateResultRead(GateResultCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    finding_id: UUID
    verification: str | None = None


class BenchmarkResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_id: str
    prompt_version: str
    effort: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_latency_ms: float
    mean_cost_usd: Decimal | None = None
    clean_false_positive_rate: float
    total_cases: int
    clean_cases: int
    created_at: datetime


class BenchmarkCompareResponse(BaseModel):
    a: BenchmarkResultRead
    b: BenchmarkResultRead
    delta: dict[str, float]


class StatsLatency(BaseModel):
    p50: float | None = None
    p95: float | None = None


class StatsCostPoint(BaseModel):
    date: str
    mean_cost_usd: float


class StatsBenchmark(BaseModel):
    config: str
    precision: float
    recall: float
    f1: float
    clean_false_positive_rate: float
    run_at: datetime


class StatsResponse(BaseModel):
    window_days: int
    total_runs: int
    gated_findings: int
    acceptance_rate: float | None = None
    rejection_reasons: dict[str, int] = {}
    mean_cost_per_review: float | None = None
    latency: StatsLatency
    mean_coverage_delta: float | None = None
    false_positive_rate: float | None = None
    cost_over_time: list[StatsCostPoint] = []
    benchmarks: list[StatsBenchmark] = []
