"""Regenerate the two human checkpoint figures and print the math cross-check.

Not an experiment (no `exp_` prefix, no row in results/results.md): the two
checkpoints in `.agents/status.md` are human-judged, and this script exists
so the figures that ship in the application -- and the correctness numbers
README.md quotes -- can be rebuilt from a fresh clone instead of living only
in a one-off session.

Run from the repository root:

    python experiments/checkpoint_figures.py

Checkpoint 1 (Phase 1): the mean face must look like a smooth, blurry
averaged face, not noise and not a blank frame.
Checkpoint 2 (Phase 2, CRITICAL): the top eigenfaces must look like ghostly
faces. If they look like noise, stop and work through
.claude/skills/debug-solver/SKILL.md -- the bug is in centering or in the
`u = X~^T v` back-projection, and nothing downstream is worth running.
"""

import numpy as np

import figures
from src import data
from src.model import EigenfaceModel

N_EIGENFACES = 10
K_FULL = 279  # n_train - 1: centering costs one degree of freedom
VARIANCE_REPORT_KS = (1, 10, 40, 160)


def load_train():
    images, labels = data.load_olivetti()
    (train_images, y_train), _ = data.stratified_split(images, labels)
    return data.flatten(train_images), y_train


def render_mean_face(X_train, y_train):
    _, mean = data.center(X_train)
    first = int(np.flatnonzero(y_train == y_train[0])[0])
    fig = figures.image_grid(
        np.stack([X_train[first], mean]),
        titles=[
            f"Training face (identity {y_train[first]})",
            f"Mean face $\\mu$ ({len(X_train)} train images)",
        ],
        ncols=2,
        suptitle="Phase 1 checkpoint: Olivetti faces and the train-set mean",
        panel_size=3.2,
        # Real face images on the dataset's own fixed scale, not autoscaled
        # per panel: the mean face genuinely is low-contrast, and stretching
        # each panel to its own min/max would overstate that contrast and
        # make the two panels incomparable.
        normalize=data.PIXEL_RANGE,
    )
    return figures.save(fig, "mean_face.png")


def render_eigenfaces(model, total_variance, n_train):
    """Per-image grey scaling here is deliberate -- see figures.image_grid."""
    shares = 100.0 * model.eigenvalues_[:N_EIGENFACES] / total_variance
    fig = figures.image_grid(
        model.components_[:N_EIGENFACES],
        titles=[f"#{i + 1}  ({share:.1f}% var)" for i, share in enumerate(shares)],
        ncols=5,
        suptitle=(
            f"Top {N_EIGENFACES} eigenfaces (snapshot trick, k={K_FULL} basis, "
            f"{n_train} training images)"
        ),
        panel_size=3.0,
    )
    return figures.save(fig, "eigenfaces_top10.png")


def cross_check(X_train, model, total_variance):
    """Verify the snapshot components against the matrix they stand in for.

    The whole point of the snapshot trick is that decomposing the 280x280
    Gram matrix gives the eigenvectors of the 4096x4096 pixel covariance
    matrix. That is only worth claiming if it has been checked against the
    real thing, so this builds C explicitly -- the one place in the project
    where paying for the big matrix is justified.
    """
    Xc, _ = data.center(X_train)
    C = (Xc.T @ Xc) / len(X_train)

    residuals = [
        np.max(np.abs(C @ u - lam * u))
        for u, lam in zip(
            model.components_[:N_EIGENFACES], model.eigenvalues_[:N_EIGENFACES]
        )
    ]
    eig_err = float(np.max(residuals))

    U = model.components_
    orth_err = float(np.max(np.abs(U @ U.T - np.eye(len(U)))))

    print(f"components fitted:            k={U.shape[0]} of n_train={len(X_train)}")
    print(f"max |C u - lambda u| (top {N_EIGENFACES}):  {eig_err:.7e}")
    print(f"max |U U^T - I| (k={U.shape[0]}):       {orth_err:.7e}")
    cumulative = np.cumsum(model.eigenvalues_) / total_variance
    shares = ", ".join(
        f"k={k}: {100.0 * cumulative[k - 1]:.1f}%" for k in VARIANCE_REPORT_KS
    )
    print(f"variance explained:           {shares}")
    return eig_err, orth_err


def main():
    X_train, y_train = load_train()
    model = EigenfaceModel().fit(X_train, k=K_FULL)
    total_variance = float(model.eigenvalues_.sum())

    mean_path = render_mean_face(X_train, y_train)
    eigen_path = render_eigenfaces(model, total_variance, len(X_train))
    cross_check(X_train, model, total_variance)

    print(f"wrote {mean_path.relative_to(figures.FIGURES_DIR.parent)}")
    print(f"wrote {eigen_path.relative_to(figures.FIGURES_DIR.parent)}")
    print(
        "\nBoth figures are human-judged checkpoints: open them and confirm a\n"
        "smooth averaged mean face and ten ghostly faces before trusting\n"
        "anything downstream."
    )


if __name__ == "__main__":
    main()
