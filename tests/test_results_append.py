"""Tests for the experiment scripts' results.md-append logic.

Each `append_results` locates its insertion point by searching for literal
substrings in results.md. Run against a temp copy of the *real* template so
a change to that template's headings/separators is caught here rather than
silently landing a row in the wrong section (or crashing with a bare
"substring not found"). See remediation.md R12 / roadmap Job 6.5.4.

Requires tests/conftest.py to put experiments/ on sys.path.
"""

import shutil
from pathlib import Path

import pytest

import exp_accuracy_vs_k
import exp_baseline
import exp_reconstruction

REAL_RESULTS_MD = Path(__file__).resolve().parent.parent / "results" / "results.md"


@pytest.fixture
def results_copy(tmp_path):
    """A throwaway copy of the real results.md template."""
    copy_path = tmp_path / "results.md"
    shutil.copy(REAL_RESULTS_MD, copy_path)
    return copy_path


def section(text, heading, next_heading):
    return text[text.index(heading) : text.index(next_heading)]


# --- exp_baseline --------------------------------------------------------


def test_baseline_row_lands_in_e1_section_only(results_copy, monkeypatch):
    monkeypatch.setattr(exp_baseline, "RESULTS_PATH", results_copy)
    exp_baseline.append_results(0.9999, 4096, 1.234, "unit-test row")

    text = results_copy.read_text()
    e1 = section(text, "## E1 -- Raw-pixel baseline", "## E2 -- Accuracy vs. k")
    assert "| 0.9999 | 4096 | 1.234 | unit-test row |" in e1
    # Doesn't bleed into the next section.
    assert "unit-test row" not in text[text.index("## E2 -- Accuracy vs. k") :]


def test_baseline_missing_marker_names_the_script_and_marker(
    results_copy, monkeypatch
):
    monkeypatch.setattr(exp_baseline, "RESULTS_PATH", results_copy)
    results_copy.write_text("no E1 heading in this file")
    with pytest.raises(RuntimeError, match="exp_baseline.*## E1"):
        exp_baseline.append_results(0.5, 4096, 1.0, "")


# --- exp_reconstruction ---------------------------------------------------


def test_reconstruction_rows_land_in_e3_section_only(results_copy, monkeypatch):
    monkeypatch.setattr(exp_reconstruction, "RESULTS_PATH", results_copy)
    exp_reconstruction.append_results([(1, 0.01234), (279, 0.00056)])

    text = results_copy.read_text()
    e3 = section(text, "## E3 -- Reconstruction vs. k", "## E4 -- Robustness suite")
    assert "| 1 | 0.012340 |" in e3
    assert "| 279 | 0.000560 |" in e3
    assert "0.012340" not in text[: text.index("## E3 -- Reconstruction vs. k")]


def test_reconstruction_missing_marker_names_the_script_and_marker(
    results_copy, monkeypatch
):
    monkeypatch.setattr(exp_reconstruction, "RESULTS_PATH", results_copy)
    results_copy.write_text("no E3 heading in this file")
    with pytest.raises(RuntimeError, match="exp_reconstruction.*## E3"):
        exp_reconstruction.append_results([(1, 0.01)])


# --- exp_accuracy_vs_k ----------------------------------------------------


def test_accuracy_vs_k_rows_land_in_e2_section_and_fill_saturation_note(
    results_copy, monkeypatch
):
    monkeypatch.setattr(exp_accuracy_vs_k, "RESULTS_PATH", results_copy)
    rows = [(1, 0.11), (5, 0.51), (20, 0.9999)]
    exp_accuracy_vs_k.append_results(rows, saturation_k=20, baseline_accuracy=0.9250)

    text = results_copy.read_text()
    e2 = section(text, "## E2 -- Accuracy vs. k", "## E3 -- Reconstruction vs. k")
    assert "| 20 | 0.9999 |" in e2
    assert "Saturation point:" in e2
    assert "k=20" in e2


def test_accuracy_vs_k_rerun_appends_a_dated_note_without_losing_the_first(
    results_copy, monkeypatch
):
    """The template's saturation placeholder is filled on the first run
    (`)_` present); a second run must not find it again, so it appends a
    dated re-run note instead of silently overwriting the first one."""
    monkeypatch.setattr(exp_accuracy_vs_k, "RESULTS_PATH", results_copy)
    exp_accuracy_vs_k.append_results(
        [(1, 0.11)], saturation_k=20, baseline_accuracy=0.9250
    )
    first_text = results_copy.read_text()
    assert "no k in" not in first_text  # sanity: the filled-in branch ran

    exp_accuracy_vs_k.append_results(
        [(1, 0.12)], saturation_k=40, baseline_accuracy=0.9250
    )
    second_text = results_copy.read_text()
    assert "k=20" in second_text  # original note preserved
    assert "Re-run" in second_text  # new note appended, not overwritten


def test_accuracy_vs_k_missing_marker_names_the_script_and_marker(
    results_copy, monkeypatch
):
    monkeypatch.setattr(exp_accuracy_vs_k, "RESULTS_PATH", results_copy)
    results_copy.write_text("no E2 heading in this file")
    with pytest.raises(RuntimeError, match="exp_accuracy_vs_k.*## E2"):
        exp_accuracy_vs_k.append_results([(1, 0.1)], None, 0.9250)
