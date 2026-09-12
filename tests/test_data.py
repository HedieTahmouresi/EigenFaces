"""Tests for src/data.py -- loading, splitting, flattening, centering.

The structural tests (split/flatten/center) run on synthetic arrays so they
stay fast and offline; only the loader tests touch the real dataset, via a
session-scoped fixture so it is fetched at most once per run.
"""

import numpy as np
import pytest

from src import data


@pytest.fixture(scope="session")
def olivetti():
    return data.load_olivetti()


@pytest.fixture
def synthetic():
    """40 identities x 10 images of 64x64, each identity a distinct constant."""
    labels = np.repeat(np.arange(40), 10)
    images = np.empty((400, 64, 64), dtype=np.float64)
    for i, label in enumerate(labels):
        images[i] = label + i / 1000.0
    return images, labels


# --- 1.1 load_olivetti -------------------------------------------------


def test_load_shapes(olivetti):
    images, labels = olivetti
    assert images.shape == (400, 64, 64)
    assert labels.shape == (400,)


def test_load_has_40_identities(olivetti):
    _, labels = olivetti
    assert len(np.unique(labels)) == 40
    counts = np.bincount(labels)
    assert np.all(counts == 10), "Olivetti should have exactly 10 images per identity"


def test_load_pixel_range_matches_declared_constant(olivetti):
    """PIXEL_RANGE is what corrupt.py will clip against -- keep them in sync."""
    images, _ = olivetti
    low, high = data.PIXEL_RANGE
    assert images.min() >= low
    assert images.max() <= high
    # Guard against a silently rescaled dataset (e.g. 0-255) still passing.
    assert images.max() > 0.5 * high


# --- 1.2 stratified_split ----------------------------------------------


def test_split_counts_per_class(synthetic):
    images, labels = synthetic
    (X_train, y_train), (X_test, y_test) = data.stratified_split(images, labels)

    assert X_train.shape[0] == 280 and y_train.shape[0] == 280
    assert X_test.shape[0] == 120 and y_test.shape[0] == 120
    assert np.all(np.bincount(y_train) == 7)
    assert np.all(np.bincount(y_test) == 3)


def test_split_is_disjoint(synthetic):
    """No image may appear in both halves -- identified by its unique value."""
    images, labels = synthetic
    (X_train, _), (X_test, _) = data.stratified_split(images, labels)

    train_ids = {float(img[0, 0]) for img in X_train}
    test_ids = {float(img[0, 0]) for img in X_test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == 280 and len(test_ids) == 120


def test_split_is_reproducible(synthetic):
    images, labels = synthetic
    first = data.stratified_split(images, labels, seed=0)
    second = data.stratified_split(images, labels, seed=0)

    for (Xa, ya), (Xb, yb) in zip(first, second):
        assert np.array_equal(Xa, Xb)
        assert np.array_equal(ya, yb)


def test_split_varies_with_seed(synthetic):
    images, labels = synthetic
    (X_a, _), _ = data.stratified_split(images, labels, seed=0)
    (X_b, _), _ = data.stratified_split(images, labels, seed=1)
    assert not np.array_equal(X_a, X_b)


def test_split_accepts_flattened_input(synthetic):
    """Splitting is axis-0 only, so it works before or after flatten()."""
    images, labels = synthetic
    X = data.flatten(images)
    (X_train, y_train), (X_test, _) = data.stratified_split(X, labels)

    assert X_train.shape == (280, 4096)
    assert X_test.shape == (120, 4096)
    assert np.all(np.bincount(y_train) == 7)


def test_split_rejects_too_few_images_per_class():
    images = np.zeros((6, 64, 64))
    labels = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        data.stratified_split(images, labels, n_train_per_class=7)


# --- 1.3 flatten -------------------------------------------------------


def test_flatten_shape(synthetic):
    images, _ = synthetic
    X = data.flatten(images)
    assert X.shape == (400, data.N_PIXELS) == (400, 4096)


def test_flatten_roundtrips_exactly(synthetic):
    images, _ = synthetic
    X = data.flatten(images)
    assert np.array_equal(X.reshape(-1, *data.IMAGE_SHAPE), images)


# --- 1.4 center --------------------------------------------------------


def test_center_computes_mean_when_none():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4096))
    Xc, mean = data.center(X)

    assert mean.shape == (4096,)
    assert np.allclose(Xc.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(mean, X.mean(axis=0))


def test_center_uses_supplied_mean_verbatim():
    """The data-leakage guard: test data is centered on the TRAIN mean."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 4096))
    X_test = rng.normal(loc=5.0, size=(20, 4096))

    _, mu_train = data.center(X_train)
    Xc_test, mean_used = data.center(X_test, mean=mu_train)

    assert np.array_equal(mean_used, mu_train)
    assert np.allclose(Xc_test, X_test - mu_train)
    # Test data centered on the train mean is NOT zero-mean -- if it were,
    # the test set's own mean had been recomputed.
    assert not np.allclose(Xc_test.mean(axis=0), 0.0, atol=1e-6)


def test_center_does_not_mutate_input():
    X = np.ones((10, 4096))
    original = X.copy()
    data.center(X)
    assert np.array_equal(X, original)
