"""E3 -- Reconstruction vs. k.

Claim tested: how much variance do the top-k eigenfaces capture?
Measures: reconstruction MSE across k in {1,5,10,20,40,80,160,279}, plus a
visual grid of reconstructions at each k for a handful of test faces.
Output: figures/reconstruction_grid.png, results row(s) in results/results.md.

MSE is measured on the TEST set: the model is fit on X_train only (per the
data-leakage guard in src/data.py) and then evaluated on held-out faces the
basis never saw. It must fall monotonically (or plateau) as k grows -- a
rise anywhere in the sweep means a bug (unsorted eigenvalues, or a k
mismatch between transform/reconstruct), not a finding, per
.claude/skills/debug-solver/SKILL.md.
"""

from datetime import date
from pathlib import Path

import numpy as np

import figures
from src import data
from src.model import EigenfaceModel

K_VALUES = (1, 5, 10, 20, 40, 80, 160, 279)
N_SAMPLE_FACES = 4
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.md"


def load_split():
    images, labels = data.load_olivetti()
    (train_images, y_train), (test_images, y_test) = data.stratified_split(
        images, labels
    )
    return data.flatten(train_images), y_train, data.flatten(test_images), y_test


def sample_face_indices(y_test, n=N_SAMPLE_FACES):
    """First test image of each of the first n distinct identities.

    Deterministic (no rng) so the reconstruction grid is reproducible run
    to run without needing a seed argument of its own.
    """
    identities = np.unique(y_test)[:n]
    return np.array([int(np.flatnonzero(y_test == ident)[0]) for ident in identities])


def reconstruction_mse(model, X_test):
    """Mean squared error over all pixels and all test images."""
    X_hat = model.reconstruct(model.transform(X_test))
    return float(np.mean((X_test - X_hat) ** 2))


def build_grid(X_test, y_test, sample_idx, models):
    """original | recon@k1 | recon@k2 | ... , one row per sampled test face."""
    samples = X_test[sample_idx]
    k_list = list(models.keys())
    n_cols = len(k_list) + 1
    recon_by_k = {
        k: model.reconstruct(model.transform(samples)) for k, model in models.items()
    }

    images = []
    titles = [None] * (n_cols * len(sample_idx))
    for row in range(len(sample_idx)):
        images.append(samples[row])
        if row == 0:
            titles[0] = "original"
        for col, k in enumerate(k_list, start=1):
            images.append(recon_by_k[k][row])
            if row == 0:
                titles[col] = f"k={k}"

    row_labels = [f"id {y_test[idx]}" for idx in sample_idx]
    fig = figures.image_grid(
        images,
        titles=titles,
        ncols=n_cols,
        suptitle="Reconstruction quality vs. k (held-out test faces)",
        row_labels=row_labels,
        # "shared": one grey scale across every panel, so the return of
        # detail as k grows is visually comparable rather than each panel
        # auto-stretching its own (possibly still-blurry) contrast.
        normalize="shared",
        panel_size=1.5,
    )
    return fig


def check_monotonic(mse_rows):
    mses = [mse for _, mse in mse_rows]
    increases = [
        (k_prev, k_next, prev, nxt)
        for (k_prev, prev), (k_next, nxt) in zip(mse_rows, mse_rows[1:])
        if nxt - prev > 1e-9
    ]
    if increases:
        details = "; ".join(
            f"k={k_prev}->k={k_next}: {prev:.6f} -> {nxt:.6f}"
            for k_prev, k_next, prev, nxt in increases
        )
        raise RuntimeError(
            f"reconstruction MSE increased in the k sweep ({details}); this is "
            "a bug (unsorted eigenvalues, or a k mismatch between transform "
            "and reconstruct), not a finding -- see "
            ".claude/skills/debug-solver/SKILL.md before trusting this run"
        )


def append_results(mse_rows):
    """Append one row per k to E3's table in results.md, without overwriting."""
    today = date.today().isoformat()
    new_lines = [f"| {today} | {k} | {mse:.6f} | |" for k, mse in mse_rows]

    text = RESULTS_PATH.read_text()
    section_idx = text.index("## E3 -- Reconstruction vs. k")
    sep_idx = text.index("|---|---|---|---|", section_idx)
    insert_at = text.index("\n", sep_idx) + 1
    new_text = text[:insert_at] + "\n".join(new_lines) + "\n" + text[insert_at:]
    RESULTS_PATH.write_text(new_text)


def main():
    X_train, y_train, X_test, y_test = load_split()

    models = {}
    mse_rows = []
    for k in K_VALUES:
        model = EigenfaceModel().fit(X_train, k=k)
        models[k] = model
        mse = reconstruction_mse(model, X_test)
        mse_rows.append((k, mse))
        print(f"k={k:>3}  test MSE={mse:.6f}")

    check_monotonic(mse_rows)

    sample_idx = sample_face_indices(y_test)
    fig = build_grid(X_test, y_test, sample_idx, models)
    fig_path = figures.save(fig, "reconstruction_grid.png")
    print(f"wrote {fig_path.relative_to(figures.FIGURES_DIR.parent)}")

    append_results(mse_rows)
    print(f"appended {len(mse_rows)} rows to results/results.md")


if __name__ == "__main__":
    main()
