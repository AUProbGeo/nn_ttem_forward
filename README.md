# Accelerating tTEM forward modelling with a generalizable neural network

Companion code for:

> Mehl, T., Madsen, R. B., & Hansen, T. M. (2024). *Accelerating tTEM forward modelling with a generalizable neural network to enable interactive probabilistic inversion – demonstrated in Daugaard, Denmark.* GEUS Bulletin.

## Overview

This repository demonstrates how a neural network (the *General NN*) trained on a broad, geologically unconstrained prior generalises — without retraining — to site-specific informed priors. Applied to tTEM (towed transient electromagnetic) data from the Daugaard valley, Denmark, the General NN replaces the GA-AEM forward solver, reducing 2 million forward evaluations from ~11 hours to 1–3 seconds (~1900× speedup in batch mode).

## Quick start

**Start here:** [`nn_ttem.py`](nn_ttem.py) — the primary companion script to the paper. It runs cell-by-cell in VS Code or Spyder and follows the paper's two-stage workflow:

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

### Using [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv venv --python 3.12            # creates .venv/
uv pip install -r requirements.txt
```

Activate the environment before running scripts:

```bash
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### GPU / CUDA support

`requirements.txt` installs CPU-only TensorFlow. Both scripts also force CPU mode via
`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` at the top. To use a GPU, remove that line
and install TensorFlow with the CUDA extras instead:

```bash
pip install "tensorflow[and-cuda]"
```

## Pre-trained model

A pre-trained General NN is provided in [`trained_models/model_big_prior_DG_HL_3_HU_300_CN_0.5_PV_200.h5`](trained_models/model_big_prior_DG_HL_3_HU_300_CN_0.5_PV_200.h5). Set `use_pretrained_model = True` at the top of `nn_ttem.py` to skip training and proceed directly to Section C.

## Check NN speed for single predictions

Use the script [`Single_forward_comparison.py`](Single_forward_comparison.py).
Test how fast the trained NN is compared to the GA-AEM function for single predictions.
You should be able to run the whole script without user input.

## Directory structure

```text
nn_ttem.py                          Primary companion script
Single_forward_comparison.py        Script to test single forward predictions

lib/                                Helper modules (NN training, error analysis, plotting)
trained_models/                     Pre-trained General NN weights
requirements.txt                    Python dependencies
```
