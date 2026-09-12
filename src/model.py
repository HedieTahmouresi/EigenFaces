"""EigenfaceModel: the change-of-basis at the center of the project.

Uses the snapshot trick to decompose the (n_train, n_train) matrix instead of
the (4096, 4096) pixel covariance matrix. If `L v = lambda v` for the
snapshot matrix, then `u = X~^T v` (normalized to unit length) is an
eigenvector of the true covariance matrix with the same eigenvalue.
"""

import numpy as np

from src import eigensolver
from src.data import center


class EigenfaceModel:
    def __init__(self, solver="eigh"):
        self.solver = solver
        self.mean_ = None
        self.components_ = None  # (k, 4096)
        self.eigenvalues_ = None  # (k,)

    def fit(self, X_train, k):
        """Fit mean and top-k eigenfaces on X_train (n_train, 4096).

        CHECKPOINT: rendered eigenfaces should look like ghostly faces. If
        they look like noise, the bug is in the X~^T v step or in centering.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        n_train = X_train.shape[0]
        if k > n_train:
            raise ValueError(
                f"k={k} exceeds n_train={n_train}; the training cloud spans at "
                f"most {n_train - 1} meaningful directions"
            )

        Xc, self.mean_ = center(X_train)

        # Snapshot trick: decompose the (n_train, n_train) Gram matrix rather
        # than the (4096, 4096) covariance matrix. Same non-zero eigenvalues.
        L = (Xc @ Xc.T) / n_train
        self.eigenvalues_, V = eigensolver.solve(L, k=k, method=self.solver)

        # Back-project snapshot eigenvectors into pixel space. X~^T v is an
        # eigenvector of the true covariance matrix but is NOT unit norm, so
        # normalizing here is mandatory, not cosmetic.
        U = Xc.T @ V  # (4096, k)
        norms = np.linalg.norm(U, axis=0)
        # A component whose norm has collapsed relative to the first one is a
        # direction the training cloud does not actually span -- normalizing
        # it would amplify float noise into a meaningless "eigenface".
        # The threshold is relative because pixel scale is arbitrary.
        if np.any(norms < 1e-9 * norms[0]):
            n_usable = int(np.count_nonzero(norms >= 1e-9 * norms[0]))
            raise ValueError(
                f"k={k} reaches past the rank of the training data: only "
                f"{n_usable} directions are actually spanned (centering costs "
                f"one degree of freedom, so the ceiling is n_train - 1 = "
                f"{n_train - 1})"
            )
        self.components_ = (U / norms).T  # (k, 4096), unit-norm rows

        return self

    def transform(self, X):
        """Project centered images onto the k-dim eigenspace.

        w = U_k (x - mean_)  -> (n, k) weight vectors.
        """
        self._check_fitted()
        Xc, _ = center(np.asarray(X, dtype=np.float64), mean=self.mean_)
        return Xc @ self.components_.T

    def reconstruct(self, W):
        """Weights back to 4096-d images: x_hat = mean_ + U_k^T w."""
        self._check_fitted()
        W = np.asarray(W, dtype=np.float64)
        return self.mean_ + W @ self.components_

    def residual(self, X):
        """||x - x_hat|| -- distance from face space, used for rejection."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_hat = self.reconstruct(self.transform(X))
        return np.linalg.norm(X - X_hat, axis=1)

    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("model is not fitted; call fit() first")
