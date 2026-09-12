"""Tests for src/model.py -- fit, transform, reconstruct, residual.

Most tests run on a small synthetic dataset built to have a known low-rank
structure, so correctness is checkable without the real dataset. Two tests
use real Olivetti data to confirm the pipeline holds at full scale.
"""

import numpy as np
import pytest

from src import data
from src.model import EigenfaceModel


@pytest.fixture
def low_rank():
    """60 samples in 200-d that genuinely live on a 3-d plane, plus an offset.

    Because the data is exactly rank 3, a model fitted with k=3 must
    reconstruct it essentially perfectly -- a strong, unambiguous target.
    """
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(200, 3)))[0]  # 3 orthonormal directions
    coords = rng.normal(size=(60, 3)) * np.array([10.0, 5.0, 1.0])
    offset = rng.normal(size=200) * 3.0
    return coords @ basis.T + offset


@pytest.fixture(scope="session")
def olivetti_train():
    images, labels = data.load_olivetti()
    (train_images, y_train), _ = data.stratified_split(images, labels)
    return data.flatten(train_images), y_train


# --- 2.3 fit -----------------------------------------------------------


def test_fit_shapes(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    assert model.components_.shape == (3, 200)
    assert model.eigenvalues_.shape == (3,)
    assert model.mean_.shape == (200,)


def test_components_are_unit_norm(low_rank):
    """X~^T v is not unit norm by default -- forgetting to normalize is one
    of the two documented ways this project's eigenfaces come out wrong."""
    model = EigenfaceModel().fit(low_rank, k=3)
    norms = np.linalg.norm(model.components_, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_components_are_mutually_orthogonal(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    gram = model.components_ @ model.components_.T
    assert np.allclose(gram, np.eye(3), atol=1e-8)


def test_eigenvalues_descending_and_non_negative(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    assert np.all(np.diff(model.eigenvalues_) <= 1e-12)
    assert np.all(model.eigenvalues_ >= -1e-12)


def test_mean_is_the_training_mean(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    assert np.allclose(model.mean_, low_rank.mean(axis=0))


def test_eigenvalues_match_the_true_covariance(low_rank):
    """The snapshot trick's whole claim: the small matrix gives the same
    eigenvalues as the big one it stands in for."""
    Xc, _ = data.center(low_rank)
    n = Xc.shape[0]
    true_cov = (Xc.T @ Xc) / n  # the 200x200 matrix we are avoiding
    true_eigenvalues = np.sort(np.linalg.eigvalsh(true_cov))[::-1][:3]

    model = EigenfaceModel().fit(low_rank, k=3)
    assert np.allclose(model.eigenvalues_, true_eigenvalues, rtol=1e-8)


def test_components_are_eigenvectors_of_the_true_covariance(low_rank):
    """C u = lambda u, for the covariance matrix we never actually built."""
    Xc, _ = data.center(low_rank)
    true_cov = (Xc.T @ Xc) / Xc.shape[0]

    model = EigenfaceModel().fit(low_rank, k=3)
    for i in range(3):
        u = model.components_[i]
        assert np.allclose(true_cov @ u, model.eigenvalues_[i] * u, atol=1e-8)


def test_fit_rejects_k_larger_than_training_set(low_rank):
    with pytest.raises(ValueError):
        EigenfaceModel().fit(low_rank, k=61)


def test_fit_rejects_k_past_the_rank_of_the_data(low_rank):
    """60 samples, but they lie on a 3-d plane: there is no 4th direction."""
    with pytest.raises(ValueError, match="rank"):
        EigenfaceModel().fit(low_rank, k=4)


# --- 2.4 transform / reconstruct ---------------------------------------


def test_transform_shape(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    assert model.transform(low_rank).shape == (60, 3)


def test_reconstruct_shape(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    W = model.transform(low_rank)
    assert model.reconstruct(W).shape == (60, 200)


def test_rank_3_data_is_recovered_exactly_at_k3(low_rank):
    """Data that truly lives on a 3-d plane loses nothing at k=3."""
    model = EigenfaceModel().fit(low_rank, k=3)
    X_hat = model.reconstruct(model.transform(low_rank))
    assert np.allclose(X_hat, low_rank, atol=1e-8)


def test_reconstruction_error_decreases_with_k(low_rank):
    """Monotonicity is the E3 checkpoint: a rising curve means a bug."""
    errors = []
    for k in [1, 2, 3]:
        model = EigenfaceModel().fit(low_rank, k=k)
        X_hat = model.reconstruct(model.transform(low_rank))
        errors.append(np.mean((low_rank - X_hat) ** 2))
    assert errors[0] > errors[1] > errors[2]


def test_transform_uses_the_training_mean_on_unseen_data(low_rank):
    """The leakage guard again: projecting new data must not recentre it."""
    model = EigenfaceModel().fit(low_rank, k=3)
    new = low_rank[:5] + 100.0  # deliberately far from the training mean

    expected = (new - model.mean_) @ model.components_.T
    assert np.allclose(model.transform(new), expected)


# --- 2.5 residual ------------------------------------------------------


def test_residual_shape_and_sign(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    residuals = model.residual(low_rank)
    assert residuals.shape == (60,)
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0)


def test_residual_is_tiny_on_data_the_basis_spans(low_rank):
    model = EigenfaceModel().fit(low_rank, k=3)
    assert np.all(model.residual(low_rank) < 1e-6)


def test_residual_is_large_off_the_face_plane(low_rank):
    """Distance from face space -- what the rejection threshold will use."""
    model = EigenfaceModel().fit(low_rank, k=3)
    rng = np.random.default_rng(1)
    junk = rng.normal(size=(5, 200)) * 50.0

    assert np.all(model.residual(junk) > model.residual(low_rank).max())


def test_unfitted_model_raises():
    model = EigenfaceModel()
    with pytest.raises(RuntimeError):
        model.transform(np.zeros((2, 200)))


# --- real data ---------------------------------------------------------


def test_fit_on_real_faces(olivetti_train):
    X_train, _ = olivetti_train
    model = EigenfaceModel().fit(X_train, k=10)

    assert model.components_.shape == (10, 4096)
    assert np.allclose(np.linalg.norm(model.components_, axis=1), 1.0, atol=1e-10)
    assert np.all(np.diff(model.eigenvalues_) <= 1e-12)
    assert np.all(model.eigenvalues_ > 0)


def test_rank_ceiling_is_n_train_minus_one(olivetti_train):
    """279 meaningful directions from 280 images, despite 4096 dimensions.

    Centering costs one degree of freedom, so the 280th direction does not
    exist -- the rank consequence the plan calls out, asserted both ways.
    """
    X_train, _ = olivetti_train

    model = EigenfaceModel().fit(X_train, k=279)
    assert model.eigenvalues_.shape == (279,)
    assert np.all(model.eigenvalues_ > 0)

    with pytest.raises(ValueError, match="rank"):
        EigenfaceModel().fit(X_train, k=280)
