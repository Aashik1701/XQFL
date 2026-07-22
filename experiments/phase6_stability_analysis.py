"""Post-hoc magnitude-based stability analysis on the Phase 6 log.

Reads experiments/phase6_log.csv without rerunning any training.

Motivation
----------
Phase 6 reported Kendall's tau = 1.000 for 13 of 15 consecutive round pairs
while the accuracy curve was still improving from 77% (round 3) to 86% (round
12).  Tau with n_features=4 has only 3 distinct values (+1, +1/3, -1/3, -1)
for a 4-element ranking, so rank saturation is expected as soon as one feature
dominates.  This script checks whether raw attribution magnitudes are still
moving — if they are, tau is too coarse and we must report cosine similarity
and L2 distance alongside it going forward.

Outputs a table of four stability metrics per consecutive round pair:
  tau          — Kendall's tau (rank-based, known to saturate here)
  rho          — Spearman correlation (also rank-based, same saturation risk)
  cos          — cosine similarity  (direction-sensitive, magnitude-agnostic)
  l2           — L2 distance        (sensitive to both direction and magnitude)

and a separate table of per-round explanation L2 norms, which shows whether
the absolute attribution magnitude is growing as the model trains.

Run from project root:
    python experiments/phase6_stability_analysis.py
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from experiments.logging_utils import ExperimentLogger
from experiments.config import ExperimentConfig
from explain.metrics import kendalls_tau, spearman_corr, cosine_similarity, l2_distance

LOG_PATH = os.path.join(PROJECT_ROOT, "experiments", "phase6_log.csv")

# ── load log ─────────────────────────────────────────────────────────────────
print("=== Phase 6 Stability Analysis (magnitude-based metrics) ===")
print(f"  Source: {LOG_PATH}\n")

cfg    = ExperimentConfig()   # config only needed for logger.load()
logger = ExperimentLogger.__new__(ExperimentLogger)
logger.csv_path = LOG_PATH
logger.config   = cfg
rows   = logger.load()

global_rows     = sorted(
    [r for r in rows if r["client_id"] == "global"],
    key=lambda r: r["round"],
)
by_round        = {r["round"]: np.array(r["explanation_vector"]) for r in global_rows}
sorted_rounds   = sorted(by_round)

# ── per-round L2 norm (is the vector itself growing?) ────────────────────────
print("--- Per-round global explanation L2 norm and accuracy ---")
print(f"  {'Round':>5}  {'||φ||':>9}  {'Accuracy':>9}  {'φ (PC0…PC3)':}")
for r in global_rows:
    vec  = np.array(r["explanation_vector"])
    norm = np.linalg.norm(vec)
    acc  = r["accuracy"]
    print(f"  {r['round']:>5}  {norm:>9.5f}  {acc:>9.4f}  "
          f"{[round(float(v), 5) for v in vec]}")

# ── consecutive-round stability across all four metrics ──────────────────────
print("\n--- Consecutive-round stability metrics ---")
print(f"  {'Rounds':>10}  {'tau':>7}  {'rho':>7}  {'cos':>7}  {'L2':>9}  "
      f"{'Δ||φ||':>9}  {'note':}")

tau_vals  = []
cos_vals  = []
l2_vals   = []
delta_norms = []

for i in range(1, len(sorted_rounds)):
    ra  = sorted_rounds[i - 1]
    rb  = sorted_rounds[i]
    va  = by_round[ra]
    vb  = by_round[rb]

    tau = kendalls_tau(va, vb)
    rho = spearman_corr(va, vb)
    cos = cosine_similarity(va, vb)
    l2  = l2_distance(va, vb)
    dn  = np.linalg.norm(vb) - np.linalg.norm(va)

    tau_vals.append(tau)
    cos_vals.append(cos)
    l2_vals.append(l2)
    delta_norms.append(dn)

    note = ""
    if tau == 1.0 and l2 > 1e-4:
        note = "<- tau saturated; magnitude still moving"
    elif tau == 1.0 and l2 < 1e-4:
        note = "<- converged (magnitude frozen too)"

    print(f"  {ra:>3} -> {rb:<3}  {tau:>+7.4f}  {rho:>+7.4f}  {cos:>+7.4f}  "
          f"{l2:>9.5f}  {dn:>+9.5f}  {note}")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n--- Summary ---")
print(f"  Mean Kendall's tau      : {np.mean(tau_vals):+.4f}  "
      f"(saturates at n_features=4: only 3 distinct non-trivial values)")
print(f"  Mean cosine similarity  : {np.mean(cos_vals):+.4f}  "
      f"(direction agreement)")
print(f"  Mean L2 distance        : {np.mean(l2_vals):.5f}  "
      f"(magnitude shift per round)")
print(f"  Mean Δ||φ|| per round   : {np.mean(delta_norms):+.5f}  "
      f"({'growing' if np.mean(delta_norms) > 0 else 'shrinking'})")

# ── verdict ───────────────────────────────────────────────────────────────────
any_l2_nonzero = any(v > 1e-4 for v in l2_vals)
print("\n--- Verdict ---")
if any_l2_nonzero and np.mean(tau_vals) >= 0.9:
    print("  CONFIRMED: tau=1.0 is rank saturation, NOT explanation convergence.")
    print("  The PC ranking locks in early (PC0 >> PC1,2,3 from round 1),")
    print("  but raw attribution magnitudes keep growing as the model trains.")
    print()
    print("  Implication: for n_features <= 4, rank metrics alone are misleading.")
    print("  Going forward, report ALL of:")
    print("    - Kendall's tau / Spearman rho  (rank stability)")
    print("    - Cosine similarity              (direction stability)")
    print("    - L2 distance                    (absolute magnitude shift)")
    print("    - ||φ|| per round                (attribution growth curve)")
elif not any_l2_nonzero:
    print("  Attribution magnitudes are also frozen — genuine explanation convergence.")
    print("  This would warrant investigation (gradient vanishing on VQC weights?).")
else:
    print("  Mixed signal — check individual round pairs above.")
