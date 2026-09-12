# Eigenfaces

A from-scratch implementation of PCA-based face recognition (Turk & Pentland,
1991), packaged as a reproducible experiment repo with an optional demo layer.

**Claim:** PCA-based recognition matches a raw-pixel baseline while using
~100x fewer dimensions, and its failure modes under geometric and
photometric perturbation can be measured precisely along each axis
independently.

## Status

Implementation not started. See `results/results.md` once experiments have
been run.

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

## Running experiments

```bash
python experiments/exp_baseline.py
python experiments/exp_accuracy_vs_k.py
# ...
```

Each experiment script writes its figure(s) to `figures/` and appends a row
to `results/results.md`.

## Demo

```bash
streamlit run app.py
```
