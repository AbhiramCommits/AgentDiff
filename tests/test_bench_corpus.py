import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agentdiff.bench import load_corpus

CORPUS = Path(__file__).resolve().parents[1] / "bench" / "corpus"


def test_corpus_composition() -> None:
    cases = load_corpus(CORPUS)

    assert len(cases) >= 12
    assert sum(1 for c in cases if not c.is_clean) >= 8
    assert sum(1 for c in cases if c.is_clean) >= 4


def test_every_patch_applies_and_ranges_are_in_bounds() -> None:
    for case_dir in sorted(CORPUS.iterdir()):
        if not case_dir.is_dir():
            continue
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td)
            for source in (case_dir / "before").iterdir():
                if source.is_file():
                    shutil.copy(source, tree / source.name)
            shutil.copy(case_dir / "patch.diff", tree / "patch.diff")
            result = subprocess.run(
                ["git", "apply", "patch.diff"],
                cwd=tree,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, (
                f"{case_dir.name}: patch does not apply: {result.stderr}"
            )
            expected = json.loads((case_dir / "expected.json").read_text())
            for entry in expected["defects"]:
                target = tree / entry["file"]
                assert target.exists(), f"{case_dir.name}: missing {entry['file']}"
                nlines = len(target.read_text().splitlines())
                start, end = entry["line_range"]
                assert 1 <= start <= end <= nlines, (
                    f"{case_dir.name}: line_range {entry['line_range']} out of bounds"
                )


def test_every_before_suite_passes() -> None:
    for case_dir in sorted(CORPUS.iterdir()):
        if not case_dir.is_dir():
            continue
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "before"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, (
            f"{case_dir.name}: before/ tests failed:\n{result.stdout}\n{result.stderr}"
        )
