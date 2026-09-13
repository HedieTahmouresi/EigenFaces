"""1-NN classification in eigenspace, plus distance-from-facespace rejection.

1-NN always returns *something*; the rejection threshold is what lets the
demo say "not a face" / "unknown" instead of a confident wrong guess.
"""

import numpy as np

from src.data import stratified_split
from src.model import EigenfaceModel


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


def compute_threshold_calibrated(
    X_train, y_train, k, percentile=95, n_calib_per_class=1, seed=1, solver="eigh"
):
    """Calibrate a rejection threshold on held-out training images, not on
    the production model's own training residuals.

    `compute_threshold(model.residual(X_train))` degenerates as k grows:
    training residuals shrink toward zero because the fitted basis spans
    the training set itself, so a threshold set from them collapses well
    below where genuine *held-out* residuals sit. Measured on Olivetti at
    the 95th percentile: 8% of test faces rejected at k=10, rising to 78%
    at k=80 and 100% at k=160 -- exactly the k range E2 identifies as the
    accuracy peak.

    The fix is to calibrate against residuals that are held out from
    *some* model the same way test images are held out from the production
    one. This holds back `n_calib_per_class` images per identity from
    `X_train`, fits a separate calibration-only `EigenfaceModel` on the
    rest, and takes the percentile from that calibration model's residuals
    on the held-back images. The returned threshold is then applied to the
    production model's residuals by the caller. Measured false-reject rate
    on the real test set with this method: ~4-8% at k=10-160 (vs. the
    production model's own accuracy peak), against a ~5% target -- see
    `.agents/decisions.md`.

    Because holding back any images reduces the calibration model's rank
    below `n_train - 1`, this cannot calibrate at the maximum k (279): a
    ValueError is raised naming the ceiling instead. That is expected --
    279 is not where E2 puts the accuracy peak anyway.

    Returns
    -------
    threshold : float
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    counts = np.unique(y_train, return_counts=True)[1]
    if counts.min() != counts.max():
        raise ValueError(
            "compute_threshold_calibrated expects an equal number of "
            f"images per identity in X_train/y_train; got counts ranging "
            f"{counts.min()}-{counts.max()}"
        )
    n_per_class = int(counts[0])
    n_fit_per_class = n_per_class - n_calib_per_class
    if n_fit_per_class < 1:
        raise ValueError(
            f"n_calib_per_class={n_calib_per_class} leaves no images to fit "
            f"the calibration model ({n_per_class} images per identity)"
        )

    n_classes = len(counts)
    max_calibratable_k = n_fit_per_class * n_classes - 1
    if k > max_calibratable_k:
        raise ValueError(
            f"k={k} exceeds the rank available to the calibration model "
            f"({max_calibratable_k}, from holding back {n_calib_per_class} "
            f"image(s)/identity to fit it on {n_fit_per_class}/identity "
            f"across {n_classes} identities); lower n_calib_per_class or "
            f"use a smaller k"
        )

    (X_fit, _), (X_calib, _) = stratified_split(
        X_train, y_train, n_train_per_class=n_fit_per_class, seed=seed
    )
    calibration_model = EigenfaceModel(solver=solver).fit(X_fit, k=k)
    calibration_residuals = calibration_model.residual(X_calib)
    return compute_threshold(calibration_residuals, percentile=percentile)


def reject(residuals, threshold):
    """Boolean mask: True where the residual exceeds the threshold.

    True means "too far from face space to trust" -- report unknown rather
    than whatever label 1-NN would have returned.
    """
    return np.asarray(residuals, dtype=np.float64) > threshold
