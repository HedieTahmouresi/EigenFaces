"""1-NN classification in eigenspace, plus distance-from-facespace rejection.

1-NN always returns *something*; the rejection threshold is what lets the
demo say "not a face" / "unknown" instead of a confident wrong guess.
"""

import numpy as np


def nearest_neighbor(weights_train, labels_train, weights_query):
    """Euclidean 1-NN in weight space. Returns predicted labels."""
    raise NotImplementedError


def compute_threshold(residuals_train, percentile=95):
    """Set a rejection threshold from the training residual distribution."""
    raise NotImplementedError


def reject(residuals, threshold):
    """Boolean mask: True where the residual exceeds the threshold."""
    raise NotImplementedError
