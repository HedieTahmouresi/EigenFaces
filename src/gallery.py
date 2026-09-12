"""DEMO ONLY. Enroll new identities and persist their weight vectors.

Depends on align.py for a consistent pipeline (alignment + lighting norm)
between enrollment and query time.
"""

import numpy as np


class Gallery:
    def __init__(self):
        self.names = []
        self.weights = []  # list of (k,) weight vectors

    def enroll(self, name, weight_vector):
        raise NotImplementedError

    def match(self, weight_vector):
        """1-NN against enrolled identities. Returns (name, distance)."""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
