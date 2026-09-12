"""DEMO ONLY. Eye-point similarity transform and lighting normalization for
own-photo enrollment.

Not used by src/ training or evaluation code -- only by the demo path
(app.py, gallery.py). Canonical eye positions are computed once by clicking
~5 training faces and averaging; lighting normalization must be applied
identically to training and demo images.
"""

import numpy as np


CANONICAL_EYE_POINTS = None  # set once, from averaging clicks on training faces


def similarity_transform(image, eye_points, canonical_eye_points=CANONICAL_EYE_POINTS):
    """Rotation + uniform scale + translation mapping eye_points onto canonical_eye_points."""
    raise NotImplementedError


def normalize_lighting(image):
    """Apply the same lighting normalization used on training images."""
    raise NotImplementedError
