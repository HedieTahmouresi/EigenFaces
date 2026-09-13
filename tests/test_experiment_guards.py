"""Tests for the experiment scripts' own correctness-checking logic.

`check_curve_shape`/`find_saturation_k` (exp_accuracy_vs_k.py),
`check_monotonic` (exp_reconstruction.py), and `check_accuracy_sanity`
(exp_baseline.py) are the guards meant to catch a broken decomposition
before it reaches results.md as a "finding" -- pure functions of
`(k, metric)` sequences, cheap to test with synthetic data rather than a
real model fit. See remediation.md R11 / roadmap Job 6.5.3.

Requires tests/conftest.py to put experiments/ on sys.path.
"""

import numpy as np
import pytest

import exp_accuracy_vs_k
import exp_baseline
import exp_reconstruction


# --- exp_baseline.check_accuracy_sanity ---------------------------------


def test_accuracy_sanity_passes_above_floor():
    exp_baseline.check_accuracy_sanity(0.9250)  # must not raise


def test_accuracy_sanity_raises_at_the_floor():
    with pytest.raises(RuntimeError, match="chance"):
        exp_baseline.check_accuracy_sanity(exp_baseline.ACCURACY_FLOOR)


def test_accuracy_sanity_raises_below_floor():
    with pytest.raises(RuntimeError, match="chance"):
        exp_baseline.check_accuracy_sanity(0.03)


# --- exp_baseline.repeated_query_time_stats (R13) -----------------------


def test_repeated_query_time_stats_returns_one_mean_per_repeat():
    """Tiny synthetic data, not real Olivetti -- this checks the repeat/mean
    bookkeeping, not actual latency numbers."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(10, 8))
    y_train = np.arange(10)
    queries = X_train[:4]

    repeat_means = exp_baseline.repeated_query_time_stats(
        X_train, y_train, queries, n_repeats=3
    )
    assert repeat_means.shape == (3,)
    assert np.all(np.isfinite(repeat_means))
    assert np.all(repeat_means >= 0)


# --- exp_accuracy_vs_k.check_curve_shape --------------------------------

N_TEST = 120  # matches the real split, so NOISE_FACES/N_TEST is the real band


def test_curve_shape_clean_monotonic_case_passes(capsys):
    rows = [(1, 0.10), (5, 0.50), (10, 0.80), (20, 0.9250)]
    exp_accuracy_vs_k.check_curve_shape(rows, n_test=N_TEST)  # must not raise
    assert "note:" not in capsys.readouterr().out


def test_curve_shape_in_band_wiggle_prints_a_note_but_does_not_raise(capsys):
    # A one-test-face dip (1/120) is inside the 2-face noise band, per
    # NOISE_FACES -- this is the exact shape E2's real k=40/279 rows have.
    one_face = 1 / N_TEST
    rows = [(1, 0.90), (5, 0.90 - one_face), (10, 0.95)]
    exp_accuracy_vs_k.check_curve_shape(rows, n_test=N_TEST)  # must not raise
    assert "note:" in capsys.readouterr().out


def test_curve_shape_out_of_band_drop_raises():
    rows = [(1, 0.90), (5, 0.50)]  # a 0.40 drop, far past the 2-face band
    with pytest.raises(RuntimeError, match="noise band"):
        exp_accuracy_vs_k.check_curve_shape(rows, n_test=N_TEST)


def test_curve_shape_floor_level_peak_raises_even_if_monotonic():
    rows = [(1, 0.02), (5, 0.03)]  # non-decreasing, but never above the floor
    with pytest.raises(RuntimeError, match="chance level"):
        exp_accuracy_vs_k.check_curve_shape(rows, n_test=N_TEST)


# --- exp_accuracy_vs_k.find_saturation_k --------------------------------


def test_find_saturation_k_returns_the_first_k_that_reaches_target():
    rows = [(1, 0.10), (5, 0.50), (10, 0.93), (20, 0.95)]
    assert exp_accuracy_vs_k.find_saturation_k(rows, target=0.9250) == 10


def test_find_saturation_k_returns_none_when_never_reached():
    rows = [(1, 0.10), (5, 0.50)]
    assert exp_accuracy_vs_k.find_saturation_k(rows, target=0.9250) is None


# --- exp_reconstruction.check_monotonic ---------------------------------


def test_check_monotonic_passes_on_a_falling_sequence():
    rows = [(1, 0.0149), (5, 0.0090), (10, 0.0071), (279, 0.0017)]
    exp_reconstruction.check_monotonic(rows)  # must not raise


def test_check_monotonic_passes_on_a_plateau():
    rows = [(80, 0.0028), (160, 0.0028), (279, 0.0028)]
    exp_reconstruction.check_monotonic(rows)  # must not raise


def test_check_monotonic_raises_on_any_rise():
    rows = [(1, 0.0149), (5, 0.0200)]  # MSE going up as k grows
    with pytest.raises(RuntimeError, match="increased"):
        exp_reconstruction.check_monotonic(rows)
