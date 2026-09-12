"""1-NN classification in eigenspace, plus distance-from-facespace rejection.

1-NN always returns *something*; the rejection threshold is what lets the
demo say "not a face" / "unknown" instead of a confident wrong guess.
"""

import numpy as np


def nearest_neighbor(weights_train, labels_train, weights_query):
    """Euclidean 1-NN in weight space. Returns predicted labels.

    Dimension-agnostic: the same function classifies 4096-d raw pixel
    vectors (E1's baseline) and k-d eigenspace weights (E2), which is what
    makes those two numbers comparable.

    Distances come from the expansion ||q - t||^2 = ||q||^2 - 2 q.t + ||t||^2.
    The ||q||^2 term is constant across the training set for a given query,
    so it is dropped: it cannot change which training vector is closest, and
    leaving it out turns the whole comparison into one matrix product
    instead of an (n_query, n_train, d) broadcast that would need ~1 GB at
    raw-pixel dimensionality.

    Ties go to the lowest training index, so the result is deterministic.

    Returns
    -------
    labels : (n_query,) array of predicted labels, same dtype as labels_train
    """
    weights_train = np.asarray(weights_train, dtype=np.float64)
    weights_query = np.asarray(weights_query, dtype=np.float64)
    labels_train = np.asarray(labels_train)

    if weights_train.ndim != 2 or weights_query.ndim != 2:
        raise ValueError(
            "expected 2-D (n, d) arrays, got shapes "
            f"{weights_train.shape} and {weights_query.shape}"
        )
    if weights_train.shape[1] != weights_query.shape[1]:
        raise ValueError(
            f"dimension mismatch: train vectors are {weights_train.shape[1]}-d, "
            f"query vectors are {weights_query.shape[1]}-d"
        )
    if weights_train.shape[0] != labels_train.shape[0]:
        raise ValueError(
            f"{weights_train.shape[0]} training vectors but "
            f"{labels_train.shape[0]} labels"
        )
    if weights_train.shape[0] == 0:
        raise ValueError("cannot classify against an empty training set")

    train_sq_norms = np.einsum("ij,ij->i", weights_train, weights_train)
    scores = train_sq_norms - 2.0 * (weights_query @ weights_train.T)
    return labels_train[np.argmin(scores, axis=1)]


def compute_threshold(residuals_train, percentile=95):
    """Set a rejection threshold from the training residual distribution.

    Calibrating on training residuals means the threshold is expressed in
    the units the model actually produces, rather than a guessed absolute
    distance. At the default percentile, roughly 5% of training faces would
    themselves be rejected -- that is the deliberate cost of rejecting
    non-faces.
    """
    residuals_train = np.asarray(residuals_train, dtype=np.float64)
    if residuals_train.size == 0:
        raise ValueError("cannot compute a threshold from an empty residual set")
    if not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    return float(np.percentile(residuals_train, percentile))


def reject(residuals, threshold):
    """Boolean mask: True where the residual exceeds the threshold.

    True means "too far from face space to trust" -- report unknown rather
    than whatever label 1-NN would have returned.
    """
    return np.asarray(residuals, dtype=np.float64) > threshold
