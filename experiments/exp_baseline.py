"""E1 -- Raw-pixel baseline.

Claim tested: what does PCA need to beat? 1-NN directly on raw 4096-d pixel
vectors (no PCA).
Measures: accuracy, dimensionality, query time.
Output: results row(s) in results/results.md.

This baseline is deliberately the "dumb" comparison: raw flattened pixels,
no PCA and no centering, so E2's eigenspace accuracies are measured against
the same split and the same 1-NN classifier -- classify.nearest_neighbor is
dimension-agnostic precisely so those two numbers are comparable. Olivetti's
identities are cleanly separated, so this should land far above the 1/40 =
2.5% chance level; decisions.md notes a trivial method gets ~90% here.
"""

from datetime import date
from pathlib import Path
import time

import numpy as np

from src import classify, data

N_IDENTITIES = 40
#: R13: an independent rerun of this experiment measured query latency ~30%
#: lower than the committed number -- wall-clock timing shares the machine
#: with whatever else is running, unlike every other number in results.md,
#: which reproduces exactly. Repeating the full sweep and reporting the
#: spread across repeats, rather than a single pass's mean, makes that
#: variance visible instead of quietly reporting one point estimate.
N_TIMING_REPEATS = 5
#: Chance level for 1-NN: each test face has one correct identity among the
#: 40, and 1-NN always returns one of them.
CHANCE_LEVEL = 1.0 / N_IDENTITIES
#: Floor for "the wiring works". A misaligned label array or a broken
#: distance computation lands near chance, so anything at or below this is
#: an anomaly to report rather than a result to record.
ACCURACY_FLOOR = 0.5

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.md"


def load_split():
    images, labels = data.load_olivetti()
    (train_images, y_train), (test_images, y_test) = data.stratified_split(
        images, labels
    )
    return data.flatten(train_images), y_train, data.flatten(test_images), y_test


def measure_query_times(X_train, y_train, queries):
    """Per-query 1-NN latency in ms, one query per call, for a single sweep.

    The demo path classifies a single uploaded image at a time, so the
    number E1 exists to record is the latency of a one-query call, not the
    amortized cost of a batch matrix product. Returns (n_query,) times.
    """
    times_ms = np.empty(queries.shape[0])
    for i in range(queries.shape[0]):
        start = time.perf_counter()
        classify.nearest_neighbor(X_train, y_train, queries[i : i + 1])
        times_ms[i] = (time.perf_counter() - start) * 1e3
    return times_ms


def repeated_query_time_stats(X_train, y_train, queries, n_repeats=N_TIMING_REPEATS):
    """Mean per-query latency of `n_repeats` independent full sweeps.

    A single sweep's mean is one point estimate of a wall-clock measurement
    that shares the machine with everything else running (R13). Repeating
    the whole sweep surfaces run-to-run variance that a single pass hides;
    within a sweep, the per-query median (not returned here, but cheap to
    recompute from `measure_query_times`) already guards against one slow
    outlier call dragging that sweep's mean.

    Returns
    -------
    repeat_means_ms : (n_repeats,) array, one mean ms/query per sweep
    """
    return np.array(
        [measure_query_times(X_train, y_train, queries).mean() for _ in range(n_repeats)]
    )


def check_accuracy_sanity(accuracy):
    """Guard against a wiring bug being recorded as an experimental finding."""
    if accuracy <= ACCURACY_FLOOR:
        raise RuntimeError(
            f"raw-pixel 1-NN accuracy {accuracy:.4f} is at or below "
            f"{ACCURACY_FLOOR:.0%}, near the {CHANCE_LEVEL:.4f} chance level. "
            "Olivetti identities are cleanly separated and this baseline "
            "should land near the ~90% decisions.md notes, so a number this "
            "low means a bug (misaligned labels, a broken distance "
            "computation) -- report it, don't record it as a result"
        )


def _find(text, marker, start=0):
    """`text.index`, but a miss names this script and the marker it wanted.

    A bare `ValueError: substring not found` gives no clue which of several
    lookups failed or why -- this is the difference between "results.md's
    template changed under exp_baseline.py" and a silent no-op.
    """
    idx = text.find(marker, start)
    if idx == -1:
        raise RuntimeError(
            f"exp_baseline.append_results: expected marker {marker!r} not "
            f"found in {RESULTS_PATH} -- has the results.md template changed?"
        )
    return idx


def append_results(accuracy, dimensions, ms_per_query, notes):
    """Append one row to E1's table in results.md, without overwriting."""
    today = date.today().isoformat()
    new_line = (
        f"| {today} | {accuracy:.4f} | {dimensions} | {ms_per_query:.3f} | {notes} |"
    )

    text = RESULTS_PATH.read_text()
    section_idx = _find(text, "## E1 -- Raw-pixel baseline")
    # E1's own separator is the first table rule after its heading. Searching
    # from section_idx (not the file start) means a `|---|` in an earlier
    # section can't be picked up, and the prefix `|---|` rather than a full
    # dash count keeps this working if the table gains a column.
    sep_idx = _find(text, "|---|", section_idx)
    insert_at = _find(text, "\n", sep_idx) + 1
    new_text = text[:insert_at] + new_line + "\n" + text[insert_at:]
    RESULTS_PATH.write_text(new_text)


def main():
    X_train, y_train, X_test, y_test = load_split()

    # No PCA, no centering: the classifier consumes the raw flattened pixels.
    predicted = classify.nearest_neighbor(X_train, y_train, X_test)
    accuracy = float((predicted == y_test).mean())
    check_accuracy_sanity(accuracy)
    print(f"raw-pixel 1-NN accuracy: {accuracy:.4f} (chance {CHANCE_LEVEL:.4f})")

    repeat_means = repeated_query_time_stats(X_train, y_train, X_test)
    ms_per_query = float(repeat_means.mean())
    ms_stdev = float(repeat_means.std(ddof=1))
    print(
        f"query latency: {ms_per_query:.3f} +/- {ms_stdev:.3f} ms/query "
        f"({N_TIMING_REPEATS} sweeps of {X_test.shape[0]} single-query calls "
        f"each; per-sweep means: {', '.join(f'{m:.3f}' for m in repeat_means)})"
    )

    append_results(
        accuracy,
        X_train.shape[1],
        ms_per_query,
        f"no PCA, no centering; mean of {N_TIMING_REPEATS} repeated sweeps of "
        f"{X_test.shape[0]} single-query calls each, stdev {ms_stdev:.3f} ms "
        "across sweeps (wall-clock timing, machine-load sensitive -- see "
        "decisions.md R13)",
    )
    print("appended 1 row to results/results.md")


if __name__ == "__main__":
    main()
