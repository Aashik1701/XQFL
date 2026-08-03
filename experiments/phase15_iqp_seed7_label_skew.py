"""Phase 15: IQP/seed=7 label-skew replication (n=4 for α=0.5 and α=0.1).

Background (MN-09, methods_notes.md):

Phase 11 produced 3 draws each for IQP/seed=7 under α=0.5 and α=0.1.
Both showed 2/3 anomalous and 0/3 in the normal ‖φ‖ range (0.28–0.41).
The 'zero normal draws' claim rests on n=3 per condition.

With an underlying ~50% anomaly rate (as IID/quantity-skew suggest), landing
0/3 normal by chance has p=0.125 per condition — not strong enough to assert
a qualitatively different 'hard degenerate' regime under label skew.

This script adds one more draw each (n=4 for both), matching IID/quantity count:

  If both remain 0/4 normal → the 'label skew produces qualitatively harder
      degenerate attractor' claim in MN-09 is well-supported.
  If either produces a normal-range draw (‖φ‖ ≥ 0.28) → soften to
      'reduced but nonzero escape probability under label skew'.

Protocol: matches Phase 11 exactly.
  - Data: random_state=42 (Phase 10/11 fixed split)
  - IQP encoding, seed=7
  - 15 rounds FedAvg, 4 clients, 3 local epochs
  - Dirichlet split: alpha=0.5 for first run, alpha=0.1 for second
  - n_global_explain=10, n_global_bg=20, n_permutations=8
  - Geometry check (G1/G2) at startup for both partitions

Run from project root:
    python experiments/phase15_iqp_seed7_label_skew.py
"""

import os, sys, csv, json, signal, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import load_breast_cancer_data
from data.partition import dirichlet_split
from models.encodings import iqp_encoding
from federation.run_federation import run_federation

# ── constants ─────────────────────────────────────────────────────────────────
SEED         = 7
N_QUBITS     = 4
N_LAYERS     = 2
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20
N_CLIENT_EXPLAIN = 5
N_CLIENT_BG      = 10
RUN_TIMEOUT  = 600
COLLAPSE_MARGIN = 0.02

NORM_ANOMALY_THRESH     = 0.15
JSD_CENT_ANOMALY_THRESH = 0.40
NORM_NORMAL_THRESH      = 0.28   # below this = below-normal (even if not strictly anomalous)

ALPHAS = [0.5, 0.1]   # one fresh draw each

# ── data loading ──────────────────────────────────────────────────────────────
print("Loading Wisconsin (random_state=42) …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Collapse threshold: {collapse_threshold:.6f}\n")

# ── JSD helper ────────────────────────────────────────────────────────────────
def jsd(a, b, eps=1e-10):
    a = np.abs(np.array(a, dtype=float)) + eps
    b = np.abs(np.array(b, dtype=float)) + eps
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2))

# ── Load Phase 10 centralized baseline ───────────────────────────────────────
rows10 = list(csv.DictReader(open(os.path.join(PROJECT_ROOT, 'results/phase10_log.csv'))))
cent_vec = np.array(next(
    json.loads(r['explanation_vector']) for r in rows10
    if r['run_type'] == 'centralized' and r['encoding'] == 'iqp'
    and r['seed'] == '7' and r['round'] == '15'
))
print(f"Phase 10 centralized IQP/seed=7: ‖φ‖={np.linalg.norm(cent_vec):.4f}\n")

# ── Existing Phase 11 draws (context) ─────────────────────────────────────────
rows11 = list(csv.DictReader(open(os.path.join(PROJECT_ROOT, 'results/phase11_log.csv'))))

for alpha in ALPHAS:
    dist_key = f"alpha_{alpha}"
    draws = [
        r for r in rows11
        if r['encoding'] == 'iqp' and r['distribution'] == dist_key
        and r['seed'] == '7' and r['round'] == '15' and r['client_id'] == 'global'
    ]
    print(f"Existing Phase 11 draws for IQP/α={alpha}/seed=7 (n={len(draws)}):")
    for r in sorted(draws, key=lambda x: x['timestamp']):
        vec = np.array(json.loads(r['explanation_vector']))
        n = float(np.linalg.norm(vec))
        anom = "ANOMALOUS" if n < NORM_ANOMALY_THRESH else (
               "below-normal" if n < NORM_NORMAL_THRESH else "NORMAL")
        print(f"  {r['run_type']:12}  acc={float(r['accuracy']):.4f}  "
              f"‖φ‖={n:.4f}  JSD_cent={jsd(vec, cent_vec):.4f}  "
              f"track={r['track_class']}  → {anom}")
print()

# ── Geometry check ────────────────────────────────────────────────────────────
def geometry_check(X, y, n_clients, alpha):
    part = dirichlet_split(X, y, n_clients, alpha=alpha, seed=SEED)
    n_total   = sum(len(yc) for _, yc in part)
    total_c1  = int((y == 1).sum())
    total_c0  = int((y == 0).sum())
    flags = []
    for i, (_, yc) in enumerate(part):
        n_c = len(yc)
        c1 = int((yc == 1).sum()); c0 = n_c - c1
        weight  = n_c / n_total
        purity  = max(c1, c0) / n_c if n_c > 0 else 0
        c1_share = c1 / total_c1 if total_c1 > 0 else 0
        c0_share = c0 / total_c0 if total_c0 > 0 else 0
        print(f"    C{i}: n={n_c:3d}  c1={c1:3d}  weight={weight:.1%}  purity={purity:.1%}")
        if weight >= 0.60 and purity >= 0.90:
            flags.append(f"G1: C{i} weight={weight:.1%} purity={purity:.1%}")
        if c1_share >= 0.90:
            flags.append(f"G2: C{i} holds {c1_share:.1%} of class-1 samples")
        if c0_share >= 0.90:
            flags.append(f"G2: C{i} holds {c0_share:.1%} of class-0 samples")
    return flags

for alpha in ALPHAS:
    print(f"Partition geometry — α={alpha}/seed=7:")
    flags = geometry_check(X_train, y_train, N_CLIENTS, alpha)
    print(f"  G1/G2 flags: {flags if flags else 'NONE'}\n")

# ── Logger ────────────────────────────────────────────────────────────────────
class CapturingLogger:
    def __init__(self):
        self.final_global_expl = None
        self.final_acc         = None

    def log(self, round_num, client_id, explanation_vector, accuracy):
        if client_id == 'global' and round_num == N_ROUNDS:
            self.final_global_expl = np.asarray(explanation_vector, dtype=float)
            self.final_acc         = float(accuracy)

def _timeout_handler(signum, frame):
    raise TimeoutError("Training timed out")

def cleanup_ray():
    try:
        import ray
        if ray.is_initialized(): ray.shutdown()
    except Exception: pass

# ── Single run ────────────────────────────────────────────────────────────────
def run_one(alpha, label):
    print(f"  [α={alpha} draw 4]  ", end='', flush=True)
    t0 = time.time()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = CapturingLogger()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=iqp_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split='dirichlet',
            dirichlet_alpha=alpha,
            strategy_name='fedavg',
            seed=SEED,
            project_root=PROJECT_ROOT,
            logger=logger,
            n_global_explain=N_GLOBAL_EXPLAIN,
            n_global_bg=N_GLOBAL_BG,
            n_explain=N_CLIENT_EXPLAIN,
            n_bg=N_CLIENT_BG,
            n_permutations=N_PERM,
        )
        signal.alarm(0)
    except TimeoutError:
        signal.alarm(0); cleanup_ray()
        print("TIMEOUT"); return None
    except Exception as e:
        signal.alarm(0); cleanup_ray()
        print(f"ERROR: {e}"); return None

    elapsed = time.time() - t0
    acc = round_log[-1][1] if round_log else float('nan')

    if logger.final_global_expl is None:
        print(f"no expl logged  t={elapsed:.0f}s"); return None

    expl     = logger.final_global_expl
    norm_phi = float(np.linalg.norm(expl))
    jsd_cent = jsd(expl, cent_vec)
    track    = 'A' if acc > collapse_threshold else 'B'

    status = ("NORMAL" if norm_phi >= NORM_NORMAL_THRESH else
              "below-normal" if norm_phi >= NORM_ANOMALY_THRESH else
              "ANOMALOUS")
    print(f"acc={acc:.4f}  track={track}  ‖φ‖={norm_phi:.4f}  "
          f"JSD_cent={jsd_cent:.4f}  → {status}  t={elapsed:.0f}s")

    return {
        'alpha': alpha, 'acc': acc, 'track': track,
        'norm_phi': norm_phi, 'jsd_cent': jsd_cent,
        'status': status, 'elapsed': elapsed,
    }


def main():
    print("=== Phase 15: IQP/seed=7 label-skew replication (draw 4 each) ===")
    print(f"  IQP encoding, seed=7, 1 fresh draw × α=[0.5, 0.1]")
    print(f"  Anomaly: ‖φ‖ < {NORM_ANOMALY_THRESH}  |  Below-normal: ‖φ‖ < {NORM_NORMAL_THRESH}")
    print(f"  Normal IQP range (other seeds): ‖φ‖ 0.28–0.41\n")

    results = {}

    print("--- Fresh draw 4: α=0.5 ---")
    results[0.5] = run_one(0.5, "alpha_0.5")
    print()

    print("--- Fresh draw 4: α=0.1 ---")
    results[0.1] = run_one(0.1, "alpha_0.1")
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=== Summary (n=4 per condition, including Phase 11) ===\n")

    for alpha in ALPHAS:
        dist_key = f"alpha_{alpha}"
        draws11 = [
            r for r in rows11
            if r['encoding'] == 'iqp' and r['distribution'] == dist_key
            and r['seed'] == '7' and r['round'] == '15' and r['client_id'] == 'global'
        ]
        p11_norms = []
        n_normal_p11 = 0
        for r in draws11:
            vec = np.array(json.loads(r['explanation_vector']))
            n = float(np.linalg.norm(vec))
            p11_norms.append(n)
            if n >= NORM_NORMAL_THRESH: n_normal_p11 += 1

        new_r = results.get(alpha)
        if new_r:
            all_norms = p11_norms + [new_r['norm_phi']]
            n_normal_all = n_normal_p11 + (1 if new_r['norm_phi'] >= NORM_NORMAL_THRESH else 0)
            n_anom_all   = sum(1 for n in all_norms if n < NORM_ANOMALY_THRESH)
            n_below_all  = sum(1 for n in all_norms
                               if NORM_ANOMALY_THRESH <= n < NORM_NORMAL_THRESH)

            print(f"  α={alpha} — all {len(all_norms)} draws:")
            print(f"    norms: {[f'{n:.3f}' for n in all_norms]}")
            print(f"    NORMAL (‖φ‖ ≥ 0.28):         {n_normal_all}/{len(all_norms)}")
            print(f"    below-normal (0.15–0.28):      {n_below_all}/{len(all_norms)}")
            print(f"    ANOMALOUS (‖φ‖ < 0.15):        {n_anom_all}/{len(all_norms)}")
            print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("=== VERDICT ===")
    for alpha in ALPHAS:
        r = results.get(alpha)
        if r is None:
            print(f"  α={alpha}: RUN FAILED — verdict requires rerun")
            continue
        if r['norm_phi'] >= NORM_NORMAL_THRESH:
            print(f"  α={alpha}: draw 4 → NORMAL (‖φ‖={r['norm_phi']:.4f} ≥ 0.28)")
            print(f"    → Soften MN-09 taxonomy: label skew has REDUCED but nonzero escape probability.")
            print(f"       Replace 'hard zero' with 'substantially lower ~1/4 escape rate vs ~1/2 under balanced'.")
        else:
            n_label = "below-normal" if r['norm_phi'] >= NORM_ANOMALY_THRESH else "ANOMALOUS"
            print(f"  α={alpha}: draw 4 → {n_label} (‖φ‖={r['norm_phi']:.4f} < 0.28)")
            print(f"    → MN-09 'hard zero under label skew' claim holds at n=4.")

    print("\n[Phase 15 complete]")


if __name__ == '__main__':
    main()
