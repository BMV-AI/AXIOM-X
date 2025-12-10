# =====================================================
# AXIOM-X v1 — THE FINAL SYMBOLIC PHYSICS DISCOVERY ENGINE
# "We do not search for truth. We extract it."
# Brandon M. Vasquez (2025)
# =====================================================

import torch
import numpy as np
import time
import json
import signal
import random
import argparse
import sys                     # ← FIXED: now imported
from pathlib import Path

# -------------------- FULL REPRODUCIBILITY (CPU + CUDA) --------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- CLI & EXPERIMENT DIR --------------------
parser = argparse.ArgumentParser(description="AXIOM-X v1")
parser.add_argument("--exp-dir", type=str, default="axiom_x_run", help="Experiment directory")
parser.add_argument("--smoke-test", action="store_true", help="Fast CI mode")
args = parser.parse_args()

exp_dir = Path(args.exp_dir)
exp_dir.mkdir(exist_ok=True)
CHECKPOINT_PATH = exp_dir / "state.pth"
LOG_PATH = exp_dir / "log.jsonl"
ELITE_LOG_PATH = exp_dir / "elites.jsonl"
BACKUP_DIR = exp_dir / "backups"
BACKUP_DIR.mkdir(exist_ok=True)

# -------------------- CONFIG --------------------
if args.smoke_test:
    POP = 8192
    STEPS = 50
    N = 1000
else:
    POP = 131072
    STEPS = 500
    N = 8000

DT = 0.02
BATCH_SIZE = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"AXIOM-X v1 — Running on {DEVICE} | POP={POP}")

# -------------------- REAL DATA --------------------
@torch.no_grad()
def generate_trajectory(seed: int = 42, noise: float = 0.01):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x = torch.randn(N, device=DEVICE) * 2
    traj = [x.clone()]
    for _ in range(STEPS):
        mean_x = x.mean()
        dx = 0.7 * mean_x - 1.2 * x + 0.9 * torch.tanh(1.8 * mean_x)
        x = x + dx * DT + torch.randn_like(x) * noise
        traj.append(x.clone())
    return torch.stack(traj)

train_data = generate_trajectory(42)
val_data = generate_trajectory(999)

# -------------------- EVALUATION --------------------
@torch.no_grad()
def evaluate_batch(params: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    B = params.shape[0]
    a = params[:, 0:1]
    b = params[:, 1:2]
    c = params[:, 2:3]
    d = params[:, 3:4]

    x = data[0:1].expand(B, -1)
    mse_total = torch.zeros(B, device=DEVICE)

    for t in range(1, STEPS + 1):
        mean_x = x.mean(dim=1, keepdim=True)
        dx = a * mean_x - b * x + c * torch.tanh(d * mean_x)
        x = x + dx * DT
        target = data[t:t+1].expand(B, -1)
        mse_total += ((x - target)**2).mean(dim=1)

    return mse_total / STEPS

# -------------------- TOURNAMENT SELECT --------------------
def tournament_select(scores: torch.Tensor, k: int = 5) -> int:
    idx = torch.randint(0, scores.shape[0], (k,), device=DEVICE)
    winner_pos = torch.argmin(scores[idx])
    return int(idx[winner_pos].item())

# -------------------- POPULATION --------------------
pop = torch.randn(POP, 4, device=DEVICE) * 0.5
pop[:, 0] += 0.7
pop = pop.clamp(-3, 3)

best_score = float('inf')
best_params = pop[0].clone()
mutation_sigma = 0.4
gen = 0
start_time = time.time()

# -------------------- CHECKPOINTING --------------------
def save_checkpoint():
    state = {
        'gen': gen,
        'pop': pop.cpu(),
        'best_params': best_params.cpu(),
        'best_score': best_score,
        'mutation_sigma': mutation_sigma,
        'rng_state': torch.get_rng_state(),
        'np_rng_state': np.random.get_state(),
        'py_rng_state': random.getstate(),
        'cuda_rng_states': torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None,
        'timestamp': time.time()
    }
    tmp = CHECKPOINT_PATH.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(CHECKPOINT_PATH)
    print(f"[checkpoint] gen {gen} saved")

def load_state():
    global pop, gen, best_score, best_params, mutation_sigma
    if CHECKPOINT_PATH.exists():
        state = torch.load(CHECKPOINT_PATH, map_location='cpu')
        pop[:] = state['pop'].to(DEVICE)
        gen = state['gen']
        best_score = state['best_score']
        best_params = state['best_params'].to(DEVICE)
        mutation_sigma = state['mutation_sigma']
        torch.set_rng_state(state['rng_state'])
        np.random.set_state(state['np_rng_state'])
        random.setstate(state['py_rng_state'])
        if DEVICE.type == "cuda" and state.get('cuda_rng_states') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_states'])
        print(f"Resumed from gen {gen}")

load_state()

# -------------------- SIGNAL HANDLING (NOW WORKS) --------------------
def signal_handler(signum, frame):
    print(f"\nSignal {signum} — saving and exiting...")
    save_checkpoint()
    sys.exit(0)  # ← sys is now imported

signal.signal(signal.SIGINT, signal_handler)

# -------------------- MAIN LOOP --------------------
print("AXIOM-X v1 — Running")
while True:
    gen += 1
    scores = torch.zeros(POP, device=DEVICE)

    for i in range(0, POP, BATCH_SIZE):
        batch = pop[i:i+BATCH_SIZE]
        scores[i:i+BATCH_SIZE] = evaluate_batch(batch, train_data)

    best_idx = scores.argmin()
    cur_best = scores[best_idx].item()

    if cur_best < best_score:
        best_score = cur_best
        best_params = pop[best_idx].clone()
        val_score = evaluate_batch(best_params.unsqueeze(0), val_data).item()
        a,b,c,d = best_params.tolist()
        print(f"\nGEN {gen:,} | TRAIN {best_score:.3e} | VAL {val_score:.3e}")
        print(f"dx = {a:+.6f}mean - {b:+.6f}x + {c:+.6f}tanh({d:+.6f}mean)")
        mutation_sigma = max(0.05, mutation_sigma * 0.95)
    else:
        mutation_sigma = min(0.7, mutation_sigma * 1.05)

    # Tournament reproduction
    elite_idx = scores.topk(8192, largest=False).indices
    elites = pop[elite_idx]
    num_children = POP - elites.shape[0]
    children = torch.empty((num_children, 4), device=DEVICE)
    for i in range(num_children):
        p1 = pop[tournament_select(scores)]
        p2 = pop[tournament_select(scores)]
        mask = torch.rand(4, device=DEVICE) < 0.5
        child = torch.where(mask, p1, p2)
        child = (child + torch.randn(4, device=DEVICE) * mutation_sigma).clamp(-3, 3)
        children[i] = child

    pop = torch.cat([elites, children])

    if gen % 100 == 0:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        print(f"gen/s {gen/elapsed:.1f} | σ={mutation_sigma:.3f}")

    if gen % 1000 == 0:
        save_checkpoint()
