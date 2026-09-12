"""Synthetic perturbations for the robustness suite (E4).

Apply to the TEST set only -- training stays clean, or the experiment
measures something other than robustness.
"""

import numpy as np


def shift(image, dx, dy):
    """Translate by (dx, dy) pixels."""
    raise NotImplementedError


def rotate(image, degrees):
    """Rotate about the image center."""
    raise NotImplementedError


def brightness(image, factor):
    """Scale pixel intensity by factor."""
    raise NotImplementedError


def occlude(image, patch_size, rng):
    """Black out a random square patch of the given size."""
    raise NotImplementedError


def add_noise(image, sigma, rng):
    """Add zero-mean Gaussian noise with the given sigma."""
    raise NotImplementedError
