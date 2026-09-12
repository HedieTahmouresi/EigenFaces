"""Olivetti faces loading, stratified train/test split, flattening, and centering.

Fit the mean face on TRAIN ONLY, then apply it to test. Fitting on all 400
images before splitting is data leakage.
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

IMAGE_SHAPE = (64, 64)
N_PIXELS = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]  # 4096

#: Valid intensity range of the loaded images. Olivetti arrives as float32
#: already scaled to [0, 1], so corruption functions that brighten or add
#: noise (src/corrupt.py) clip to these bounds rather than to 0-255.
PIXEL_RANGE = (0.0, 1.0)


def load_olivetti():
    """Load the Olivetti faces dataset via sklearn.

    Returns
    -------
    images : (400, 64, 64) float array
    labels : (400,) int array, 40 identities
    """
    dataset = fetch_olivetti_faces(shuffle=False)
    images = np.asarray(dataset.images, dtype=np.float64)
    labels = np.asarray(dataset.target, dtype=int)
    return images, labels


def stratified_split(images, labels, n_train_per_class=7, seed=0):
    """Split into train/test with a fixed number of images per identity.

    Splits along axis 0 only, so it works on either (n, 64, 64) images or
    (n, 4096) flattened vectors. The same seed always yields the same split.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    images = np.asarray(images)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < n_train_per_class:
            raise ValueError(
                f"identity {label} has {len(idx)} images, "
                f"fewer than n_train_per_class={n_train_per_class}"
            )
        shuffled = rng.permutation(idx)
        train_idx.append(shuffled[:n_train_per_class])
        test_idx.append(shuffled[n_train_per_class:])

    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.sort(np.concatenate(test_idx))

    return (images[train_idx], labels[train_idx]), (images[test_idx], labels[test_idx])


def flatten(images):
    """(n, 64, 64) -> (n, 4096)."""
    images = np.asarray(images)
    return images.reshape(images.shape[0], -1)


def center(X, mean=None):
    """Subtract a mean face. If mean is None, compute it from X (train only).

    Pass the *training* mean when centering test data -- recomputing a mean
    from the test set leaks information the model should not have.

    Returns
    -------
    X_centered, mean
    """
    X = np.asarray(X, dtype=np.float64)
    if mean is None:
        mean = X.mean(axis=0)
    else:
        mean = np.asarray(mean, dtype=np.float64)
    return X - mean, mean
