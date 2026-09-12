"""Tests for src/eigensolver.py.

Only the `eigh` path is implemented (Phase 2); `solve_power` lands in Phase 11
and its tests belong with it.
"""

import numpy as np
import pytest

from src import eigensolver


@pytest.fixture
def symmetric():
    """A random symmetric positive-semidefinite matrix, like a real L."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 12))
    return (A @ A.T) / 12


# --- 2.1 solve_eigh ----------------------------------------------------


def test_eigenvalues_are_descending(symmetric):
    eigenvalues, _ = eigensolver.solve_eigh(symmetric)
    assert np.all(np.diff(eigenvalues) <= 1e-12)


def test_eigenpairs_satisfy_the_defining_equation(symmetric):
    """The only test that really matters: L v == lambda v."""
    eigenvalues, eigenvectors = eigensolver.solve_eigh(symmetric)
    for i in range(5):
        v = eigenvectors[:, i]
        assert np.allclose(symmetric @ v, eigenvalues[i] * v, atol=1e-10)


def test_eigenvectors_are_orthonormal(symmetric):
    _, eigenvectors = eigensolver.solve_eigh(symmetric)
    gram = eigenvectors.T @ eigenvectors
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


def test_rank_deficient_matrix_has_near_zero_tail(symmetric):
    """A 30x30 built from 12 columns has only 12 non-zero eigenvalues.

    This is the rank limit the snapshot trick exploits: n points span at
    most n directions, however many dimensions they live in.
    """
    eigenvalues, _ = eigensolver.solve_eigh(symmetric)
    assert np.all(eigenvalues[:12] > 1e-8)
    assert np.allclose(eigenvalues[12:], 0.0, atol=1e-10)


# --- 2.2 solve dispatch ------------------------------------------------


def test_solve_matches_solve_eigh_truncated(symmetric):
    full_values, full_vectors = eigensolver.solve_eigh(symmetric)
    values, vectors = eigensolver.solve(symmetric, k=5, method="eigh")

    assert np.allclose(values, full_values[:5])
    assert np.allclose(vectors, full_vectors[:, :5])
    assert vectors.shape == (30, 5)


def test_solve_without_k_returns_everything(symmetric):
    values, vectors = eigensolver.solve(symmetric, method="eigh")
    assert values.shape == (30,)
    assert vectors.shape == (30, 30)


def test_solve_power_branch_is_not_implemented_yet(symmetric):
    """Phase 11. Guards against the branch silently falling through to eigh."""
    with pytest.raises(NotImplementedError):
        eigensolver.solve(symmetric, k=5, method="power")


def test_solve_rejects_unknown_method(symmetric):
    with pytest.raises(ValueError):
        eigensolver.solve(symmetric, k=5, method="svd")
