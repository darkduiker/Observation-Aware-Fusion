# Observation-Aware Fusion

Code for the paper: Observation-Aware Fusion for Intrapartum Delivery Intervention Prediction from Sparse Longitudinal Assessments.

## Contents

- `common_decomposition.py`: matched RF conditions A–E
- `fusion.py`: OAF gate and shrinkage
- `grud.py`: GRU-D historical expert
- `run_baselines.py`: six-model benchmark
- `run_nested.py`: fully nested validation with source-only calibration
- `publication_figures.py`: Figures 2–5
- `evaluate.py`: AUROC, AUPRC, Brier, bootstrap intervals
- ...

## Environment

Python 3.x, see `requirements.txt`.

## How to reproduce

1. Primary benchmark: `python run_baselines.py`
2. Fully nested validation: `python run_nested.py`
3. Figures 2–5: `python publication_figures.py`

## Data

Clinical data are not publicly available due to privacy restrictions.
