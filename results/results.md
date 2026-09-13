# Results

One section per experiment. Append rows as experiments are run; do not
overwrite prior results without noting what changed.

## How to append

Each experiment owns the section below matching its number. When a script
runs, it appends a row to its own table and stamps the run date. A re-run
that supersedes an earlier row does **not** delete it -- append the new row
and add a note saying what changed (code fix, different seed, different
`k`), so a wrong number and its correction are both visible.

Every section records the split it was measured on. Unless a row says
otherwise, that is: Olivetti, stratified 7 train / 3 test per identity,
`seed=0`, n_train=280, n_test=120, mean and eigenfaces fit on train only.

Tables below are empty templates. Do not fill them with placeholder or
estimated values -- a row appears only when a script actually produced it.

---

## E1 -- Raw-pixel baseline

1-NN on raw flattened 4096-d pixel vectors, no PCA and no centering. This is
the number the PCA claim has to match at ~100x fewer dimensions.

| Date | Accuracy | Dimensions | Query time (ms/query) | Notes |
|---|---|---|---|---|
| 2026-09-13 | 0.9250 | 4096 | 2.172 | no PCA, no centering; mean of 5 repeated sweeps of 120 single-query calls each, stdev 0.685 ms across sweeps (wall-clock timing, machine-load sensitive -- see decisions.md R13) |
| 2026-09-13 | 0.9250 | 4096 | 3.270 | no PCA, no centering; mean of 120 single-query calls |

---

## E2 -- Accuracy vs. k

1-NN in eigenspace. One row per `k`; a run fills the whole sweep at once.

| Date | k | Accuracy | Notes |
|---|---|---|---|
| 2026-09-13 | 1 | 0.1250 | |
| 2026-09-13 | 5 | 0.7333 | |
| 2026-09-13 | 10 | 0.8667 | |
| 2026-09-13 | 20 | 0.9250 | |
| 2026-09-13 | 40 | 0.9167 | |
| 2026-09-13 | 80 | 0.9333 | |
| 2026-09-13 | 160 | 0.9333 | |
| 2026-09-13 | 279 | 0.9250 | |

**Saturation point:** k=20 at or above E1's 0.9250 (rows appended 2026-09-13)
-- 4096/20 ~= 205x fewer dimensions than the raw-pixel baseline. This is the
project's central claim. Accuracy plateaus from k=20 onward (111-112 of 120
test faces); the one-face drops at k=40 and k=279 are held-out quantization,
not a decomposition bug -- the checklist was verified clean: eigenvalues
descending, `max |C u - lambda u| = 3.1e-15` in pixel space, and E3's
reconstruction MSE (same bases) monotonic.

---

## E3 -- Reconstruction vs. k

Reconstruction MSE on the test set. MSE must fall monotonically (or
plateau) as `k` grows; a non-monotonic curve is a bug, not a finding.

| Date | k | Test MSE | Notes |
|---|---|---|---|
| 2026-09-13 | 1 | 0.014872 | |
| 2026-09-13 | 5 | 0.008965 | |
| 2026-09-13 | 10 | 0.007108 | |
| 2026-09-13 | 20 | 0.005233 | |
| 2026-09-13 | 40 | 0.003763 | |
| 2026-09-13 | 80 | 0.002807 | |
| 2026-09-13 | 160 | 0.002118 | |
| 2026-09-13 | 279 | 0.001707 | |

---

## E4 -- Robustness suite

Corruption applied to the **test set only**; training stays clean. Accuracy
measured at the `k` that E2 identified as best.

| Date | Axis | Level | Accuracy | Notes |
|---|---|---|---|---|

**Linking observation:** _(record the shift magnitude at which accuracy
collapses -- this sets the px precision `align.py`'s eye-clicking needs, per
`.agents/experiments.md`. Do not let this number get lost; the README
depends on it.)_

---

## E5 -- Drop the first 3 eigenfaces

Hypothesis: the leading components encode illumination rather than identity,
so dropping them should not hurt and may help.

| Date | Variant | k | Accuracy | Notes |
|---|---|---|---|---|

---

## E6 -- Whitening

Weight vectors scaled by `1/sqrt(eigenvalue)` before the 1-NN distance.

| Date | Whitening | k | Accuracy | Notes |
|---|---|---|---|---|

---

## E7 -- Solver validation (power iteration vs. eigh)

Validated at k=10 before any higher `k`. Eigenvectors are compared as
`|u . u_ref|` (never raw -- sign ambiguity). Orthogonality drift is a
reported metric, not a bug to eliminate.

| Date | k | Max eigenvalue abs err | Min \|u . u_ref\| | Max orthogonality err | Runtime eigh (s) | Runtime power (s) | Notes |
|---|---|---|---|---|---|---|---|

---

## E8 -- Enrollment of unseen identities

Olivetti-trained basis, identities never seen during fit, enrolled through
the `align.py` + `gallery.py` demo pipeline.

| Date | Identities | Photos enrolled | Photos queried | Accuracy | Rejection threshold | Notes |
|---|---|---|---|---|---|---|
