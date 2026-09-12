"""Eigendecomposition of the snapshot covariance matrix L = (1/n) X~ X~^T.

Two interchangeable solvers for the same eigenvectors/eigenvalues of the
snapshot matrix: `solve_eigh` (default, exact) and `solve_power` (from
scratch, optional). Neither should be able to hurt accuracy relative to the
other -- they solve the same problem.
"""

import numpy as np


def solve_eigh(L):
    """Exact eigendecomposition via numpy.linalg.eigh, sorted descending.

    Returns
    -------
    eigenvalues : (n,) sorted descending
    eigenvectors : (n, n) columns are eigenvectors of L
    """
    raise NotImplementedError


def solve_power(L, k, n_iter=1000, tol=1e-10, seed=0):
    """From-scratch power iteration + deflation for the top k eigenpairs.

    Re-orthogonalize each new vector against all previous ones (Gram-Schmidt)
    and track the orthogonality error -- deflation drift is expected by
    roughly the 50th component and should be reported, not hidden.

    Returns
    -------
    eigenvalues : (k,) sorted descending
    eigenvectors : (n, k) columns are eigenvectors of L
    orthogonality_errors : (k,) running |V^T V - I| metric per added vector
    """
    raise NotImplementedError


def solve(L, k=None, method="eigh"):
    """Dispatch to solve_eigh or solve_power by name."""
    raise NotImplementedError
