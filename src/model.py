"""EigenfaceModel: the change-of-basis at the center of the project.

Uses the snapshot trick to decompose the (n_train, n_train) matrix instead of
the (4096, 4096) pixel covariance matrix. If `L v = lambda v` for the
snapshot matrix, then `u = X~^T v` (normalized to unit length) is an
eigenvector of the true covariance matrix with the same eigenvalue.
"""

import numpy as np


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
        raise NotImplementedError

    def transform(self, X):
        """Project centered images onto the k-dim eigenspace.

        w = U_k (x - mean_)  -> (n, k) weight vectors.
        """
        raise NotImplementedError

    def reconstruct(self, W):
        """Weights back to 4096-d images: x_hat = mean_ + U_k^T w."""
        raise NotImplementedError

    def residual(self, X):
        """||x - x_hat|| -- distance from face space, used for rejection."""
        raise NotImplementedError
