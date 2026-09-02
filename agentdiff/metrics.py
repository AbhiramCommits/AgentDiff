from prometheus_client import Counter, Gauge, Histogram

REVIEW_LATENCY = Histogram(
    "agentdiff_review_latency_seconds",
    "End-to-end review latency",
    ["model", "prompt_version"],
    buckets=(1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0),
)

GATE_DURATION = Histogram(
    "agentdiff_gate_duration_seconds",
    "Gate duration per finding",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0),
)

TOKENS = Counter(
    "agentdiff_tokens_total",
    "Tokens consumed by the LLM",
    ["model", "kind"],
)

COST = Counter(
    "agentdiff_cost_usd_total",
    "LLM spend in USD",
    ["model"],
)

FINDINGS = Counter(
    "agentdiff_findings_total",
    "Findings emitted by reviews",
    ["severity", "category"],
)

SUGGESTIONS = Counter(
    "agentdiff_suggestions_total",
    "Gate decisions per finding",
    ["decision", "reason"],
)

COVERAGE_DELTA = Histogram(
    "agentdiff_coverage_delta",
    "Coverage after minus coverage before, per gated finding",
    buckets=(-50.0, -20.0, -10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0),
)

REVIEWS_IN_FLIGHT = Gauge(
    "agentdiff_reviews_in_flight",
    "Reviews currently running",
)

LLM_ERRORS = Counter(
    "agentdiff_llm_errors_total",
    "LLM API and validation errors",
    ["error_type"],
)
