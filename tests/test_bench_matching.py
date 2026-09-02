from utils import make_bench_finding

from agentdiff.bench import (
    GroundTruthDefect,
    config_name,
    match_findings,
    parse_config_name,
)
from agentdiff.models import Category, Severity


def defect(**overrides) -> GroundTruthDefect:
    base = dict(
        id="d1",
        file="app.py",
        line_range=(2, 2),
        category=Category.CORRECTNESS,
        severity=Severity.MAJOR,
    )
    base.update(overrides)
    return GroundTruthDefect(**base)


def test_exact_match_is_true_positive() -> None:
    outcome = match_findings([make_bench_finding()], [defect()])
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (1, 0, 0)


def test_right_line_wrong_category_is_not_true_positive() -> None:
    finding = make_bench_finding(category=Category.MAINTAINABILITY)
    outcome = match_findings([finding], [defect()])
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (0, 1, 1)


def test_two_findings_on_same_defect_count_once() -> None:
    outcome = match_findings(
        [make_bench_finding(), make_bench_finding()],
        [defect()],
    )
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (1, 1, 0)


def test_two_defects_can_each_be_matched_once() -> None:
    defects = [defect(id="d1"), defect(id="d2", line_range=(5, 5))]
    outcome = match_findings(
        [
            make_bench_finding(),
            make_bench_finding(start_line=5, end_line=5),
            make_bench_finding(),
        ],
        defects,
    )
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (2, 1, 0)


def test_overlapping_line_ranges_match() -> None:
    outcome = match_findings(
        [make_bench_finding(start_line=1, end_line=3)],
        [defect(line_range=(2, 2))],
    )
    assert outcome.true_positives == 1


def test_adjacent_non_overlapping_ranges_do_not_match() -> None:
    outcome = match_findings(
        [make_bench_finding(start_line=3, end_line=4)],
        [defect(line_range=(1, 2))],
    )
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (0, 1, 1)


def test_wrong_file_is_not_true_positive() -> None:
    outcome = match_findings(
        [make_bench_finding(file_path="other.py")],
        [defect()],
    )
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (0, 1, 1)


def test_diff_prefixes_are_normalized() -> None:
    outcome = match_findings(
        [make_bench_finding(file_path="a/app.py")],
        [defect(file="app.py")],
    )
    assert outcome.true_positives == 1


def test_nested_path_matches_suffix() -> None:
    outcome = match_findings(
        [make_bench_finding(file_path="src/app.py")],
        [defect(file="app.py")],
    )
    assert outcome.true_positives == 1


def test_different_subdirectories_do_not_match() -> None:
    outcome = match_findings(
        [make_bench_finding(file_path="lib/app.py")],
        [defect(file="src/app.py")],
    )
    assert outcome.true_positives == 0


def test_finding_without_line_range_never_matches() -> None:
    outcome = match_findings(
        [make_bench_finding(start_line=None, end_line=None)],
        [defect()],
    )
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (0, 1, 1)


def test_severity_difference_does_not_affect_matching() -> None:
    outcome = match_findings(
        [make_bench_finding(severity=Severity.NIT)],
        [defect(severity=Severity.BLOCKER)],
    )
    assert outcome.true_positives == 1


def test_no_findings_and_no_defects_is_clean_zero() -> None:
    outcome = match_findings([], [])
    assert (outcome.true_positives, outcome.false_positives, outcome.false_negatives) == (0, 0, 0)


def test_config_name_roundtrip() -> None:
    name = config_name("claude-opus-5", "v1", "high")
    assert name == "claude-opus-5@v1@high"
    assert parse_config_name(name) == ("claude-opus-5", "v1", "high")


def test_parse_config_name_rejects_malformed() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_config_name("no-separators")
    with pytest.raises(ValueError):
        parse_config_name("a@@high")
