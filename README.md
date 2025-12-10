# AXIOM-X v1  
**Symbolic Physics Discovery Engine**  
*“We do not search for truth. We extract it.”*

AXIOM-X is a fully reproducible, GPU-accelerated evolutionary discovery engine designed to evolve symbolic dynamical equations directly from data. It uses large-scale tournament-based evolution to recover governing equations of physical systems without assuming a model class in advance.

This repository contains the first public release of AXIOM-X.

---

## 🚀 What AXIOM-X Does

AXIOM-X evolves equations of the form:

dx = a·mean(x) − b·x + c·tanh(d·mean(x))

It automatically:
- Generates synthetic physical trajectories
- Evolves symbolic parameters using large-population evolution
- Performs tournament selection and crossover
- Applies adaptive mutation
- Tracks best discoveries on training + validation data
- Resumes seamlessly from checkpoints
- Runs on CPU or CUDA GPU
- Is bit-reproducible across runs

This is not a neural network.  
This is direct equation discovery by evolution.

---

## 🧠 Core Features

- Large-scale evolutionary search (131,072 agents)
- Tournament selection + crossover
- Adaptive mutation scheduling
- GPU acceleration via PyTorch
- Bit-reproducible CPU + CUDA execution
- Automatic checkpointing + resume
- SIGINT-safe shutdown (Ctrl+C)
- Smoke-test CI mode for fast validation

---

## 📦 Installation

Requires:

- Python 3.9+
- PyTorch
- NumPy

Install dependencies:

pip install torch numpy

---

## ▶️ Running AXIOM-X

Standard run:

python axiom_x.py

Custom experiment directory:

python axiom_x.py --exp-dir my_experiment

Fast smoke test:

python axiom_x.py --smoke-test

---

## 💾 Checkpointing & Resume

AXIOM-X automatically saves:

- Population
- Best parameters
- RNG states (Python, NumPy, Torch, CUDA)
- Mutation state

To resume, just run again in the same directory:

python axiom_x.py --exp-dir my_experiment

---

## 🧪 What This Is For

AXIOM-X is designed for:

- Symbolic regression
- Dynamical system discovery
- Physics law discovery
- Chaos system recovery
- Equation search research
- Evolutionary computation research

It is not:
- A neural network
- A language model
- A game engine

This is a scientific discovery system.

---

## ⚠️ Disclaimer

This project is experimental research software.  
No claims are made about correctness of discovered physics laws.  
Use for research, education, and experimentation only.

---

## 📜 License

This project is licensed under the Apache License, Version 2.0 (Apache-2.0).

You may:
- Use it
- Modify it
- Distribute it
- Use it commercially

You must:
- Include the license
- Provide attribution

This software is provided “AS IS”, without warranty.

---

## 👤 Author

Brandon M. Vasquez  
2025  
Independent Researcher & Systems Developer

