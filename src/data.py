"""Olivetti faces loading, stratified train/test split, flattening, and centering.

Fit the mean face on TRAIN ONLY, then apply it to test. Fitting on all 400
images before splitting is data leakage.
"""

import numpy as np


def load_olivetti():
    """Load the Olivetti faces dataset via sklearn.

    Returns
    -------
    images : (400, 64, 64) float array
    labels : (400,) int array, 40 identities
    """
    raise NotImplementedError


def stratified_split(images, labels, n_train_per_class=7, seed=0):
    """Split into train/test with a fixed number of images per identity.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    raise NotImplementedError


def flatten(images):
    """(n, 64, 64) -> (n, 4096)."""
    raise NotImplementedError


def center(X, mean=None):
    """Subtract a mean face. If mean is None, compute it from X (train only).

    Returns
    -------
    X_centered, mean
    """
    raise NotImplementedError
