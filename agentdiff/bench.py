import json
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .models import BenchmarkResult, Category, Severity
from .reviewer import LLMError, ReviewFinding, Reviewer

logger = logging.getLogger(__name__)

CONFIG_SEPARATOR = "@"


@dataclass(frozen=True)
class GroundTruthDefect:
    id: str
    file: str
    line_range: tuple[int, int]
    category: Category
    severity: Severity


@dataclass(frozen=True)
class CorpusCase:
    name: str
    patch: str
    defects: list[GroundTruthDefect]

    @property
    def is_clean(self) -> bool:
        return not self.defects


def config_name(model_id: str, prompt_version: str, effort: str) -> str:
    return CONFIG_SEPARATOR.join([model_id, prompt_version, effort])


def parse_config_name(name: str) -> tuple[str, str, str]:
    parts = name.split(CONFIG_SEPARATOR)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"invalid config name '{name}': expected model@{CONFIG_SEPARATOR}"
            f"prompt_version@{CONFIG_SEPARATOR}effort"
        )
    return parts[0], parts[1], parts[2]


def load_corpus(corpus_dir: Path) -> list[CorpusCase]:
    cases: list[CorpusCase] = []
    for case_dir in sorted(corpus_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        patch_file = case_dir / "patch.diff"
        expected_file = case_dir / "expected.json"
        if not patch_file.exists() or not expected_file.exists():
            raise ValueError(
                f"corpus case '{case_dir.name}' is missing patch.diff or expected.json"
            )
        raw = json.loads(expected_file.read_text())
        defects: list[GroundTruthDefect] = []
        ids: set[str] = set()
        for entry in raw.get("defects", []):
            start, end = int(entry["line_range"][0]), int(entry["line_range"][1])
            if start > end:
                raise ValueError(f"corpus case '{case_dir.name}': line_range start > end")
            defect_id = str(entry["id"])
            if defect_id in ids:
                raise ValueError(f"corpus case '{case_dir.name}': duplicate defect id")
            ids.add(defect_id)
            defects.append(
                GroundTruthDefect(
                    id=defect_id,
                    file=str(entry["file"]),
                    line_range=(start, end),
                    category=Category(entry["category"]),
                    severity=Severity(entry["severity"]),
                )
            )
        cases.append(
            CorpusCase(
                name=case_dir.name,
                patch=patch_file.read_text(),
                defects=defects,
            )
        )
    return cases


def _normalize_path(path: str) -> str:
    cleaned = path.strip()
    for prefix in ("a/", "b/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.strip("./")


def _paths_match(finding_path: str, defect_path: str) -> bool:
    finding = _normalize_path(finding_path)
    defect = _normalize_path(defect_path)
    if finding == defect:
        return True
    return finding.endswith("/" + defect) or defect.endswith("/" + finding)


def _lines_overlap(
    finding_start: int | None,
    finding_end: int | None,
    defect_start: int,
    defect_end: int,
) -> bool:
    if finding_start is None or finding_end is None:
        return False
    return finding_start <= defect_end and defect_start <= finding_end


@dataclass(frozen=True)
class MatchOutcome:
    true_positives: int
    false_positives: int
    false_negatives: int


def match_findings(
    findings: list[ReviewFinding], defects: list[GroundTruthDefect]
) -> MatchOutcome:
    unmatched = list(defects)
    true_positives = 0
    false_positives = 0
    for finding in findings:
        matched_index: int | None = None
        for index, defect in enumerate(unmatched):
            if (
                _paths_match(finding.file_path, defect.file)
                and finding.category == defect.category
                and _lines_overlap(
                    finding.start_line,
                    finding.end_line,
                    defect.line_range[0],
                    defect.line_range[1],
                )
            ):
                matched_index = index
                break
        if matched_index is None:
            false_positives += 1
        else:
            true_positives += 1
            unmatched.pop(matched_index)
    return MatchOutcome(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=len(unmatched),
    )


@dataclass(frozen=True)
class CaseScore:
    case: str
    clean: bool
    failed: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    latency_ms: int
    cost_usd: Decimal


@dataclass(frozen=True)
class BenchMetrics:
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
    mean_cost_usd: Decimal
    clean_false_positive_rate: float
    total_cases: int
    clean_cases: int
    per_case: list[CaseScore]

    @property
    def config(self) -> str:
        return config_name(self.model_id, self.prompt_version, self.effort)


def _aggregate(scores: list[CaseScore]) -> tuple[int, int, int]:
    return (
        sum(s.true_positives for s in scores),
        sum(s.false_positives for s in scores),
        sum(s.false_negatives for s in scores),
    )


async def run_benchmark(
    cases: list[CorpusCase],
    reviewer: Reviewer,
    model_id: str,
    prompt_version: str,
    effort: str,
) -> BenchMetrics:
    scores: list[CaseScore] = []
    for case in cases:
        started = time.monotonic()
        try:
            outcome = await reviewer.review(diff=case.patch)
            matched = match_findings(outcome.result.findings, case.defects)
            scores.append(
                CaseScore(
                    case=case.name,
                    clean=case.is_clean,
                    failed=False,
                    true_positives=matched.true_positives,
                    false_positives=matched.false_positives,
                    false_negatives=matched.false_negatives,
                    latency_ms=outcome.latency_ms,
                    cost_usd=outcome.cost_usd,
                )
            )
        except LLMError as exc:
            logger.warning("benchmark case '%s' failed: %s", case.name, exc)
            scores.append(
                CaseScore(
                    case=case.name,
                    clean=case.is_clean,
                    failed=True,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=len(case.defects),
                    latency_ms=int((time.monotonic() - started) * 1000),
                    cost_usd=Decimal("0.0"),
                )
            )

    true_positives, false_positives, false_negatives = _aggregate(scores)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    successful = [s for s in scores if not s.failed]
    mean_latency_ms = (
        sum(s.latency_ms for s in successful) / len(successful) if successful else 0.0
    )
    mean_cost_usd = (
        sum((s.cost_usd for s in successful), start=Decimal("0")) / len(successful)
        if successful
        else Decimal("0")
    ).quantize(Decimal("0.000001"))

    clean_scores = [s for s in scores if s.clean]
    clean_false_positive_rate = (
        sum(s.false_positives for s in clean_scores) / len(clean_scores)
        if clean_scores
        else 0.0
    )

    return BenchMetrics(
        model_id=model_id,
        prompt_version=prompt_version,
        effort=effort,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        mean_latency_ms=round(mean_latency_ms, 2),
        mean_cost_usd=mean_cost_usd,
        clean_false_positive_rate=round(clean_false_positive_rate, 6),
        total_cases=len(cases),
        clean_cases=len(clean_scores),
        per_case=scores,
    )


def metrics_to_row(metrics: BenchMetrics) -> BenchmarkResult:
    return BenchmarkResult(
        model_id=metrics.model_id,
        prompt_version=metrics.prompt_version,
        effort=metrics.effort,
        true_positives=metrics.true_positives,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        mean_latency_ms=metrics.mean_latency_ms,
        mean_cost_usd=metrics.mean_cost_usd,
        clean_false_positive_rate=metrics.clean_false_positive_rate,
        total_cases=metrics.total_cases,
        clean_cases=metrics.clean_cases,
    )
