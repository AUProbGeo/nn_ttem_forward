# Accelerating tTEM forward modelling with a generalizable neural network

Companion code for:

> Mehl, T., Madsen, R. B., & Hansen, T. M. (2024). *Accelerating tTEM forward modelling with a generalizable neural network to enable interactive probabilistic inversion – demonstrated in Daugaard, Denmark.* GEUS Bulletin.

## Overview

This repository demonstrates how a neural network (the *General NN*) trained on a broad, geologically unconstrained prior generalises — without retraining — to site-specific informed priors. Applied to tTEM (towed transient electromagnetic) data from the Daugaard valley, Denmark, the General NN replaces the GA-AEM forward solver, reducing 2 million forward evaluations from ~11 hours to 1–3 seconds (~1900× speedup in batch mode).

## Quick start

**Start here:** [`nn_ttem_inversion.py`](nn_ttem_inversion.py) — the primary companion script to the paper. It runs cell-by-cell in VS Code or Spyder and follows the paper's two-stage workflow:

- **Stage A** — Construct the general prior, compute GA-AEM forward responses, and train the General NN.
- **Stage B** — Load the pre-trained General NN, replace GA-AEM responses in the informed Daugaard prior with NN predictions, and run probabilistic inversion using the extended rejection sampler.

At the top of the script, set the problem size:

```python
N_prior  = 2_000_000  # Set to 2_000 for a quick test run
N_use    = 2_000_000
N_inv    = 2_000_000
N_reject = 2_000_000
```

## Installation

```bash
pip install -r requirements.txt
```

## Scripts

| Script | Description |
|--------|-------------|
| [`nn_ttem_inversion.py`](nn_ttem_inversion.py) | Primary companion script — trains the General NN and runs inversion (paper workflow) |
| [`Train_NN_and_invert_full_GEUS_paper_github.py`](Train_NN_and_invert_full_GEUS_paper_github.py) | Full pipeline with N = 2 000 000 |
| [`Geus relative plots_github.py`](<Geus relative plots_github.py>) | Training stability experiments (Fig. 1a in paper) |
| [`Single_forward_comparition_test_github.py`](Single_forward_comparition_test_github.py) | Sequential NN vs GA-AEM speed benchmark |

## Pre-trained model

A pre-trained General NN is provided in `trained models/model_big_prior_DG_HL_3_HU_300_CN_0.5_PV_200.h5`. To use it, proceed directly to Section C of `nn_ttem_inversion.py`.

## Data

- `DAUGAARD_AVG.h5` — observed tTEM data from the Daugaard survey
- `PRIOR_UNIFORM_NL_1-9_log-uniform_N2000000_*.h5` — pre-computed general prior (2M realizations)
- `daugaard_valley_prior_N2000000.h5` — informed Daugaard prior (used for generalisation evaluation and inversion)
- `inversion_data/` — saved posterior files from previous inversions

## Directory structure

```
trained models/     Pre-trained General NN weights
inversion_data/     Saved posterior HDF5 files
plots/              Output figures
stability_of_training/  Saved weights from repeated training runs
```
