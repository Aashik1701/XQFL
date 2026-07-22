"""Phase 6: Explanation logging across federated rounds.

Runs the same 4-client IID FedAvg setup as Phase 5, but after every round:
  - each client computes a local Q-SHAP explanation on its local data
    (with locally-trained model weights, before aggregation)
  - the server computes a global Q-SHAP explanation on the shared validation
    set (with the newly aggregated global model)
  - both are written to experiments/phase6_log.csv

Produces:
  experiments/phase6_log.csv  — one row per (round, client_id)

Definition of done (spec section 6):
  - CSV contains per-round rows for all 4 clients + "global"
  - Kendall's tau between consecutive global explanations computes without error

Run from project root:
    python experiments/phase6_explanation_logging.py
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from experiments.config import ExperimentConfig
from experiments.logging_utils import ExperimentLogger
from explain.metrics import kendalls_tau
from federation.run_federation import run_federation

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
cfg = ExperimentConfig(
    dataset="wisconsin",
    encoding="angle",
    aggregator="fedavg",
    distribution="iid",
    n_clients=4,
    n_rounds=15,
    n_qubits=4,
    n_layers=2,
    local_epochs=3,
    seed=SEED,
)

LOG_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase6_log.csv")

# ── data ─────────────────────────────────────────────────────────────────────
print("=== Phase 6: Explanation Logging Across Federated Rounds ===")
print(f"  Config: {cfg.encoding} | {cfg.aggregator} | {cfg.distribution} "
      f"| {cfg.n_clients} clients | {cfg.n_rounds} rounds")
print(f"  Log -> {LOG_PATH}\n")

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

# Use first 10 test samples as the shared explanation validation set
X_val = X_test[:10]

# ── logger ───────────────────────────────────────────────────────────────────
logger = ExperimentLogger(LOG_PATH, cfg)

# ── run federation with explanation logging ───────────────────────────────────
print("--- Data partitions ---")
history, round_log = run_federation(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    encoding_fn=angle_encoding,
    n_clients=cfg.n_clients,
    n_rounds=cfg.n_rounds,
    local_epochs=cfg.local_epochs,
    n_qubits=cfg.n_qubits,
    n_layers=cfg.n_layers,
    split=cfg.distribution,
    strategy_name=cfg.aggregator,
    seed=cfg.seed,
    project_root=PROJECT_ROOT,
    # Phase 6 logging
    logger=logger,
    X_val=X_val,
    n_global_explain=10,
    n_global_bg=20,
    n_explain=5,
    n_bg=10,
    n_permutations=8,
)

# ── verify log ────────────────────────────────────────────────────────────────
print("\n=== Log verification ===")
rows = logger.load()
print(f"  Total rows in log: {len(rows)}")

global_rows  = [r for r in rows if r["client_id"] == "global"]
client_rows  = [r for r in rows if r["client_id"] != "global"]
rounds_logged = sorted({r["round"] for r in rows})

print(f"  Rounds with data  : {rounds_logged}")
print(f"  Global rows       : {len(global_rows)}")
print(f"  Client rows       : {len(client_rows)}")

# Spot-check: every round 1-N should have 4 client rows + 1 global row
expected_rounds = list(range(1, cfg.n_rounds + 1))
missing_global = [r for r in expected_rounds
                  if not any(g["round"] == r for g in global_rows)]
if missing_global:
    print(f"  [WARN] Missing global rows for rounds: {missing_global}")
else:
    print("  [PASS] Global explanation present for every round")

# Print a sample explanation vector
if global_rows:
    sample = global_rows[-1]   # last round's global explanation
    print(f"\n  Round {sample['round']} global Q-SHAP: "
          f"{[round(v, 4) for v in sample['explanation_vector']]}")
    print(f"  Round {sample['round']} global accuracy: {sample['accuracy']:.4f}")

# ── definition-of-done: Kendall's tau between consecutive global rounds ───────
print("\n=== Kendall's tau between consecutive global rounds ===")
global_by_round = {r["round"]: np.array(r["explanation_vector"]) for r in global_rows}
sorted_rounds   = sorted(global_by_round)

tau_values = []
for i in range(1, len(sorted_rounds)):
    rnd_a = sorted_rounds[i - 1]
    rnd_b = sorted_rounds[i]
    tau   = kendalls_tau(global_by_round[rnd_a], global_by_round[rnd_b])
    tau_values.append(tau)
    print(f"  Round {rnd_a:2d} -> {rnd_b:2d}  tau = {tau:+.4f}")

if tau_values:
    print(f"\n  Mean tau across consecutive rounds: {np.mean(tau_values):+.4f}")
    print(f"  (tau -> 1.0 = stable; tau -> -1.0 = fully reversed ranking)")
    assert all(not np.isnan(t) for t in tau_values), "Kendall's tau returned NaN"
    print("\n  [PASS] Kendall's tau computes without errors for all consecutive rounds")

print("\n[Phase 6 complete]")
print(f"  Log written to: {LOG_PATH}")
