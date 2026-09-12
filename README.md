# Eigenfaces

A from-scratch implementation of PCA-based face recognition (Turk & Pentland,
1991), packaged as a reproducible experiment repo with an optional demo layer.

**Claim:** PCA-based recognition matches a raw-pixel baseline while using
~100x fewer dimensions, and its failure modes under geometric and
photometric perturbation can be measured precisely along each axis
independently.

## Status

The core pipeline is implemented and verified; the experiments are not yet
written, so no recognition accuracy has been measured.

Implemented:

- `src/data.py` -- Olivetti loading, stratified 7/3-per-identity split,
  flattening, and centering on the training mean.
- `src/eigensolver.py` -- `solve_eigh` and the `solve` dispatcher
  (`solve_power` is still a stub).
- `src/model.py` -- `EigenfaceModel.fit/transform/reconstruct/residual`,
  using the snapshot trick. The recovered components are checked against
  the 4096x4096 covariance matrix they stand in for: `max |C u - lambda u|`
  is 1.25e-15 over the top 10.
- `src/classify.py` -- Euclidean 1-NN over arbitrary vectors (raw pixels or
  eigenspace weights, so the baseline and the PCA result stay comparable),
  and a rejection threshold calibrated on the training residual
  distribution.
- Figures: `figures/mean_face.png`, `figures/eigenfaces_top10.png`.

Not yet implemented: `corrupt.py`, `align.py`, `gallery.py`, every
`experiments/exp_*.py` script, the notebook, and the Streamlit demo.
`results/results.md` currently holds empty per-experiment templates.

## Project layout

```
src/            PCA, eigensolver, classifier, corruption, demo-only alignment/gallery
experiments/    One script per experiment (E1-E8), each writes to figures/ and results/
figures/        Generated figures, committed as they're produced
results/        results.csv / results.md
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
# ...
```

Each experiment script writes its figure(s) to `figures/` and appends a row
to `results/results.md`. These scripts are currently stubs and raise
`NotImplementedError` -- see Status above.

## Demo

```bash
streamlit run app.py
```
