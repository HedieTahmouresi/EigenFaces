"""E4 -- Robustness suite (test set only).

Claim tested: how brittle is eigenfaces recognition, per perturbation axis?
Varies: translation (1,2,4,8 px), rotation (2,5,10,20 deg),
brightness (0.5-2x), occlusion (8,16,24 px patch), Gaussian noise (sigma).
Uses src/corrupt.py; corruption is applied to the TEST set only.
Output: figures/robustness_*.png, results rows in results/results.md.

Key linking observation for the README: if accuracy collapses past 4 px of
shift, eye-clicking in align.py needs ~2 px precision -- a measured
requirement, not a guess.
"""

if __name__ == "__main__":
    raise NotImplementedError
