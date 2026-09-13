# Eigenfaces

A from-scratch implementation of PCA-based face recognition (Turk & Pentland,
1991), packaged as a reproducible experiment repo with an optional demo layer.

**Claim:** PCA-based recognition matches a raw-pixel baseline while using
~100x fewer dimensions, and its failure modes under geometric and
photometric perturbation can be measured precisely along each axis
independently.

## Status

Phases 1-6 -- the mandatory path -- are complete and independently
reproducible: load -> split -> center -> snapshot-trick PCA -> project ->
reconstruct -> 1-NN classify, validated against the real 4096x4096
covariance matrix the snapshot trick stands in for
(`max |C u - lambda u| = 1.25e-15` over the top 10 eigenfaces).

The project's central claim -- PCA-based recognition matches a raw-pixel
baseline at far fewer dimensions -- is measured, not asserted:

- **E1 (raw-pixel baseline):** 1-NN on 4096-d pixels scores 0.9250.
- **E2 (accuracy vs. k):** eigenspace 1-NN reaches E1's 0.9250 at k=20 --
  4096/20 ~= 205x fewer dimensions -- and peaks at 0.9333 (k=80-160) before
  plateauing through k=279.
- **E3 (reconstruction vs. k):** test MSE falls monotonically from 0.014872
  (k=1) to 0.001707 (k=279).

Full per-k tables are in `results/results.md`; the figures below are in
`figures/`. A short remediation pass (**Phase 6.5**, tracked in the local
build docs, not shipped in this file) is closing a handful of guard-rail
gaps found in Phases 4-6 -- most notably that the "unknown face" rejection
threshold is miscalibrated exactly at the `k` E2 just identified as best --
before work continues into the robustness, ablation, and demo phases.

Implemented:

- `src/data.py` -- Olivetti loading, stratified 7/3-per-identity split,
  flattening, and centering on the training mean.
- `src/eigensolver.py` -- `solve_eigh` and the `solve` dispatcher
  (`solve_power` is still a stub, planned for a later phase).
- `src/model.py` -- `EigenfaceModel.fit/transform/reconstruct/residual`,
  using the snapshot trick.
- `src/classify.py` -- Euclidean 1-NN over arbitrary vectors (raw pixels or
  eigenspace weights, so the baseline and the PCA result stay comparable),
  and a rejection threshold calibrated on the training residual
  distribution (recalibration in progress -- see Phase 6.5 above).
- `experiments/exp_baseline.py`, `exp_accuracy_vs_k.py`,
  `exp_reconstruction.py` -- E1/E2/E3, producing the numbers above.
- Figures: `figures/mean_face.png`, `figures/eigenfaces_top10.png`,
  `figures/accuracy_vs_k.png`, `figures/reconstruction_grid.png`.

Not yet implemented: `corrupt.py`, `align.py`, `gallery.py`,
`exp_robustness.py`, `exp_ablations.py`, `exp_enrollment.py`,
`exp_solver_validation.py` (power-iteration solver), the notebook, and the
Streamlit demo.

## Project layout

```
src/            PCA, eigensolver, classifier, corruption, demo-only alignment/gallery
experiments/    One script per experiment (E1-E8), each writes to figures/ and results/
figures/        Generated figures, committed as they're produced
results/        results.md, one section per experiment
notebook.ipynb  Clean narrative, imports from src/
app.py          Streamlit demo, thin wrapper only
```

Dataset: Olivetti faces (400 images, 40 people, 64x64), loaded via
scikit-learn. PCA itself is hand-written using numpy; scikit-learn is used
only for dataset loading and the train/test split. OpenCV, if used, is
confined to the demo path (`align.py`, `gallery.py`, `app.py`) and never
appears in the core `src/` pipeline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
```

## Tests

```bash
pytest
```

Each `src/` module has a matching `tests/test_<module>.py`, written
alongside it rather than retrofitted.

## Running experiments

```bash
python experiments/exp_baseline.py
python experiments/exp_accuracy_vs_k.py
python experiments/exp_reconstruction.py
# ...
```

Each experiment script writes its figure(s) to `figures/` and appends a row
to `results/results.md`. E1/E2/E3 above are implemented and reproducible;
the remaining `exp_*.py` scripts (robustness, ablations, solver validation,
enrollment) are still stubs -- see Status above.

## Demo

```bash
streamlit run app.py
```
