"""Tests for src/classify.py -- 1-NN and distance-from-facespace rejection.

The structural tests run on synthetic clusters and synthetic residual
distributions so they stay fast and are checkable by hand; the real-data
tests at the bottom confirm the two pieces work on actual eigenspace
weights and actual model residuals.
"""

import numpy as np
import pytest

from src import classify, data
from src.model import EigenfaceModel


@pytest.fixture
def two_clusters():
    """Two well-separated 5-d clusters, labelled 'A' and 'B'.

    Deliberately string labels: the classifier returns labels it was given,
    it does not assume they are integer class indices.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.2, size=(20, 5))
    b = rng.normal(loc=10.0, scale=0.2, size=(20, 5))
    weights = np.vstack([a, b])
    labels = np.array(["A"] * 20 + ["B"] * 20)
    return weights, labels


@pytest.fixture(scope="session")
def olivetti_split():
    images, labels = data.load_olivetti()
    (train_images, y_train), (test_images, y_test) = data.stratified_split(
        images, labels
    )
    return data.flatten(train_images), y_train, data.flatten(test_images), y_test


# --- 3.1 nearest_neighbor ----------------------------------------------


def test_query_near_a_cluster_gets_that_clusters_label(two_clusters):
    weights, labels = two_clusters
    queries = np.array([[0.1] * 5, [9.9] * 5])
    assert list(classify.nearest_neighbor(weights, labels, queries)) == ["A", "B"]


def test_training_points_classify_as_themselves(two_clusters):
    """A query identical to a training vector has distance 0 to it."""
    weights, labels = two_clusters
    predicted = classify.nearest_neighbor(weights, labels, weights)
    assert np.array_equal(predicted, labels)


def test_matches_a_brute_force_distance_computation(two_clusters):
    """Guards the ||q||^2-dropping shortcut in the distance expansion."""
    weights, labels = two_clusters
    rng = np.random.default_rng(1)
    queries = rng.normal(loc=5.0, scale=4.0, size=(30, 5))

    brute_force = np.array(
        [labels[np.argmin(np.linalg.norm(weights - q, axis=1))] for q in queries]
    )
    assert np.array_equal(
        classify.nearest_neighbor(weights, labels, queries), brute_force
    )


def test_output_shape_and_dtype(two_clusters):
    weights, labels = two_clusters
    predicted = classify.nearest_neighbor(weights, labels, weights[:7])
    assert predicted.shape == (7,)
    assert predicted.dtype == labels.dtype


def test_works_at_raw_pixel_dimensionality():
    """E1 runs this on 4096-d vectors; E2 on k-d weights. Same function."""
    rng = np.random.default_rng(2)
    train = rng.normal(size=(40, 4096))
    labels = np.arange(40)
    queries = train[[3, 17]] + rng.normal(scale=1e-3, size=(2, 4096))
    assert np.array_equal(classify.nearest_neighbor(train, labels, queries), [3, 17])


def test_ties_resolve_to_the_lowest_training_index():
    train = np.array([[0.0], [0.0], [5.0]])
    labels = np.array([10, 20, 30])
    assert classify.nearest_neighbor(train, labels, np.array([[0.0]]))[0] == 10


def test_dimension_mismatch_raises(two_clusters):
    weights, labels = two_clusters
    with pytest.raises(ValueError, match="dimension mismatch"):
        classify.nearest_neighbor(weights, labels, np.zeros((2, 4)))


def test_label_count_mismatch_raises(two_clusters):
    weights, labels = two_clusters
    with pytest.raises(ValueError, match="labels"):
        classify.nearest_neighbor(weights, labels[:10], weights[:2])


def test_empty_training_set_raises():
    with pytest.raises(ValueError, match="empty training set"):
        classify.nearest_neighbor(np.zeros((0, 5)), np.array([]), np.zeros((1, 5)))


# --- 3.2 compute_threshold / reject ------------------------------------


def test_threshold_flags_the_top_5_percent_of_a_known_distribution():
    residuals = np.arange(1000, dtype=np.float64)
    flagged = classify.reject(residuals, classify.compute_threshold(residuals))
    assert 45 <= flagged.sum() <= 55


def test_threshold_percentile_is_configurable():
    residuals = np.arange(1000, dtype=np.float64)
    strict = classify.compute_threshold(residuals, percentile=50)
    lenient = classify.compute_threshold(residuals, percentile=99)
    assert strict < lenient
    assert classify.reject(residuals, strict).sum() > classify.reject(
        residuals, lenient
    ).sum()


def test_reject_returns_a_boolean_mask_of_matching_shape():
    residuals = np.array([1.0, 5.0, 9.0])
    mask = classify.reject(residuals, 4.0)
    assert mask.dtype == bool
    assert np.array_equal(mask, [False, True, True])


def test_a_value_exactly_at_the_threshold_is_kept():
    """Rejection is strictly greater-than, so the threshold itself passes."""
    assert not classify.reject(np.array([3.0]), 3.0)[0]


def test_empty_residuals_raise():
    with pytest.raises(ValueError, match="empty"):
        classify.compute_threshold(np.array([]))


def test_out_of_range_percentile_raises():
    with pytest.raises(ValueError, match="percentile"):
        classify.compute_threshold(np.arange(10.0), percentile=150)


# --- real data ---------------------------------------------------------


def test_1nn_in_eigenspace_recognizes_real_faces(olivetti_split):
    """Smoke test of the full path, not a result.

    E2 is what measures accuracy properly; this only asserts the wiring
    produces something far above the 1/40 = 2.5% chance level, so a broken
    projection or a label misalignment would fail here rather than silently
    becoming an experimental "finding".
    """
    X_train, y_train, X_test, y_test = olivetti_split
    model = EigenfaceModel().fit(X_train, k=40)
    predicted = classify.nearest_neighbor(
        model.transform(X_train), y_train, model.transform(X_test)
    )
    assert (predicted == y_test).mean() > 0.5


def test_noise_images_are_rejected_but_real_faces_are_not(olivetti_split):
    """Distance from face space separates faces from non-faces."""
    X_train, _, X_test, _ = olivetti_split
    model = EigenfaceModel().fit(X_train, k=40)
    threshold = classify.compute_threshold(model.residual(X_train))

    rng = np.random.default_rng(0)
    noise = rng.uniform(*data.PIXEL_RANGE, size=(20, data.N_PIXELS))

    assert classify.reject(model.residual(noise), threshold).all()
    assert classify.reject(model.residual(X_test), threshold).mean() < 0.5


# --- 6.5.2 compute_threshold_calibrated ---------------------------------


@pytest.mark.parametrize("k", [10, 40, 80, 160])
def test_calibrated_threshold_keeps_test_rejection_low_across_k(olivetti_split, k):
    """Regression for R7/R10: the naive threshold rejects 78-100% of real
    test faces at k=80-160, exactly where E2 says accuracy peaks. The
    calibrated threshold must stay well below that at every k E2 considered,
    not just the k=40 value the old pinned test happened to still pass at.
    """
    X_train, y_train, X_test, _ = olivetti_split
    model = EigenfaceModel().fit(X_train, k=k)
    threshold = classify.compute_threshold_calibrated(X_train, y_train, k=k)
    assert classify.reject(model.residual(X_test), threshold).mean() < 0.2


def test_calibrated_threshold_still_rejects_noise(olivetti_split):
    X_train, y_train, _, _ = olivetti_split
    k = 160
    model = EigenfaceModel().fit(X_train, k=k)
    threshold = classify.compute_threshold_calibrated(X_train, y_train, k=k)

    rng = np.random.default_rng(0)
    noise = rng.uniform(*data.PIXEL_RANGE, size=(20, data.N_PIXELS))
    assert classify.reject(model.residual(noise), threshold).all()


def test_calibrated_threshold_is_deterministic(olivetti_split):
    X_train, y_train, _, _ = olivetti_split
    first = classify.compute_threshold_calibrated(X_train, y_train, k=40)
    second = classify.compute_threshold_calibrated(X_train, y_train, k=40)
    assert first == second


def test_calibrated_threshold_rejects_k_past_calibration_rank(olivetti_split):
    """k=279 spans the full training set, so holding back even one image per
    identity to calibrate on necessarily drops the calibration model's rank
    below it -- this must fail loudly, not silently calibrate at a lower k.
    """
    X_train, y_train, _, _ = olivetti_split
    with pytest.raises(ValueError, match="exceeds the rank"):
        classify.compute_threshold_calibrated(X_train, y_train, k=279)


def test_calibrated_threshold_rejects_uneven_class_counts():
    y_train = np.array([0, 0, 0, 1, 1])
    X_train = np.zeros((5, 10))
    with pytest.raises(ValueError, match="equal number of images"):
        classify.compute_threshold_calibrated(X_train, y_train, k=1)
