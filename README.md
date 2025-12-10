AXIOM-X v1.0 is the first public release of a high-performance symbolic physics discovery engine built for evolutionary equation search and dynamical system identification.

This release introduces a full GPU-accelerated population-based evolutionary framework for discovering governing equations directly from time-series data.

━━━━━━━━━━━━━━━━━━━━
CORE FEATURES
━━━━━━━━━━━━━━━━━━━━
• Large-scale population evolution (up to 131,072 candidates)
• GPU-accelerated evaluation via PyTorch
• Tournament selection + elite preservation
• Adaptive mutation scheduling
• Full checkpointing with deterministic reproducibility
• CPU & CUDA deterministic seeding
• Automatic resume on interruption
• Training + validation trajectory system
• Real-time best-equation reporting

━━━━━━━━━━━━━━━━━━━━
DISCOVERY OBJECTIVE
━━━━━━━━━━━━━━━━━━━━
AXIOM-X searches for governing equations of the form:

dx = a·mean(x) − b·x + c·tanh(d·mean(x))

by evolving parameters directly against generated physical trajectories using mean-squared trajectory error as the fitness objective.

━━━━━━━━━━━━━━━━━━━━
SYSTEM DESIGN
━━━━━━━━━━━━━━━━━━━━
• Torch tensor evolution pipeline
• Batched MSE evaluation
• Tournament genetic reproduction
• Elitism with population refresh
• Continuous generation loop with autosave recovery
• Signal-safe checkpointing (Ctrl+C safe)

━━━━━━━━━━━━━━━━━━━━
REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━
Python 3.9+
torch
numpy

Install with:
pip install -r requirements.txt

Run with:
python axiom-x.py

━━━━━━━━━━━━━━━━━━━━
STATUS
━━━━━━━━━━━━━━━━━━━━
This is a v1.0 research release intended for:
• Symbolic regression experiments
• Dynamical system identification
• Evolutionary physics modeling
• AI-driven equation discovery

Future versions will expand symbolic operator libraries, equation structure evolution, and multi-system discovery.

━━━━━━━━━━━━━━━━━━━━
AUTHOR
━━━━━━━━━━━━━━━━━━━━
Brandon M. Vasquez
2025
