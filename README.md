# AXIOM-X v1.0  
**Symbolic Physics Discovery Engine**

Brandon M. Vasquez (2025)

---

## Overview

AXIOM-X is a high-performance symbolic discovery engine for identifying governing equations of dynamical systems directly from time-series data using large-scale evolutionary search.

It combines GPU-accelerated evaluation, deterministic reproducibility, and population-scale optimization into a single research-grade system.

---

## Core Features

- Population-scale evolution (up to 131,072 candidates)
- GPU-accelerated fitness evaluation (PyTorch)
- Tournament selection with elite preservation
- Adaptive mutation scheduling
- Full checkpointing with exact reproducibility
- CPU and CUDA deterministic seeding
- Automatic resume on interruption
- Training and validation trajectory system
- Real-time best-equation reporting

---

## Discovery Objective

AXIOM-X optimizes equations of the form:

```
dx = a·mean(x) − b·x + c·tanh(d·mean(x))
```

by evolving parameters directly against time-series trajectories using mean-squared error as the fitness objective.

---

## Architecture

- Batched tensor evolution pipeline
- GPU MSE evaluation
- Tournament genetic reproduction
- Elitism with continuous population refresh
- Signal-safe autosave recovery
- Full RNG state preservation (Python, NumPy, Torch, CUDA)

---

## Requirements

- Python 3.9+
- torch
- numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python axiom-x.py
```

Optional CI smoke test:

```bash
python axiom-x.py --smoke-test
```

Checkpoints and logs are written to:

```
axiom_x_run/
```

---

## Reproducibility

All runs are fully deterministic across CPU and CUDA.  
Checkpoint state includes population tensors, fitness state, mutation schedule, and full RNG state for exact resume.

---

## Applications

- Symbolic regression
- Dynamical system identification
- Equation discovery from data
- Evolutionary physics modeling

---

## Status

v1.0 — Public research release.

---

## Author

Brandon M. Vasquez  
2025

---

## License

Apache License 2.0

---

## Citation

See `CITATION.cff`

---

## Contributing

See `CONTRIBUTING.md`

---

## Security

See `SECURITY.md`
