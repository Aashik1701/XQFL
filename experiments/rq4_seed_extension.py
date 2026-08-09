"""RQ4 seed extension: direct MIA sweep, seeds 11–47.

Extends Phase 13's cross-encoding MIA record from 5 to 15 seeds to test whether
AUC rankings become statistically resolvable, specifically the amplitude–IQP gap
(currently 0.9pp, well below the n=5 threshold of 3.1pp).

Existing Phase 13 seeds:   [1, 7, 42, 123, 999]
New seeds (this script):   [11, 13, 17, 19, 23, 29, 31, 37, 43, 47]
Combined n per encoding:   15

Reliability threshold: 7pp / sqrt(n)
  n=5:  3.13pp — amplitude vs IQP = 0.9pp  UNRESOLVED
  n=15: 1.81pp — amplitude vs IQP still UNRESOLVED if gap holds; check all pairs

Shadow MIA omitted — confirmed to fail across all encodings at n=5. Only
direct MIA (cross-validated LR on target model explanations) is extended.

Protocol per (encoding, seed):
  1. Retrain IID FedAvg (Wisconsin, 4 clients, 15 rounds)
  2. Compute Q-SHAP for 50 members + 50 non-members
  3. Direct MIA (cv=5 LR), log AUC

Outputs:
  results/rq4_extension_log.csv
  results/rq4_extension_checkpoint.csv
"""

import os, sys, csv, json, signal, time, statistics
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import (angle_encoding, amplitude_encoding,
                               reuploading_encoding, iqp_encoding)
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack, _qshap_for_samples

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS       = 4
N_LAYERS       = 2
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_MIA_SAMPLES  = 50
N_BG           = 20
N_PERMUTATIONS = 8
RUN_TIMEOUT    = 600

EXISTING_SEEDS = [1, 7, 42, 123, 999]
NEW_SEEDS      = [11, 13, 17, 19, 23, 29, 31, 37, 43, 47]

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq4_extension_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq4_extension_checkpoint.csv")
PHASE13_CSV    = os.path.join(PROJECT_ROOT, "results", "phase13_mia_log.csv")

RESULTS_COLS = [
    "encoding", "seed", "target_acc", "direct_mia_auc",
    "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(enc, seed):
    return f"{enc}|{seed}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("done",):
                done.add(row["run_key"])
    return done


def mark_done(rk, status="done"):
    first = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "status", "completed_at"])
        if first:
            w.writeheader()
        w.writerow({
            "run_key":      rk,
            "status":       status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [checkpoint] {status}: {rk}")


def append_result(enc, seed, target_acc, direct_auc, n_mem, n_nonmem, wall_time):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if first:
            w.writeheader()
        w.writerow({
            "encoding":      enc,
            "seed":          seed,
            "target_acc":    f"{target_acc:.6f}",
            "direct_mia_auc": f"{direct_auc:.6f}",
            "n_members":     n_mem,
            "n_nonmembers":  n_nonmem,
            "wall_time_s":   f"{wall_time:.1f}",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        })


# ── timeout / Ray cleanup ─────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("Federated training timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


# ── run one (encoding, seed) ──────────────────────────────────────────────────

def run_one(enc_name, seed):
    rk   = make_run_key(enc_name, seed)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  RQ4-EXT  {enc_name} / IID / seed={seed}")
    print(f"{'='*70}")

    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)
    print(f"  Train={len(X_train)}  Test={len(y_test)}")

    # 1. Retrain federated model, capture weights
    model_out = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=ENCODINGS[enc_name],
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            split="iid",
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            n_permutations=0,
            model_out=model_out,
        )
        signal.alarm(0)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  [TIMEOUT] {e}")
        cleanup_ray()
        mark_done(rk, status="timeout")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"  [ERROR] {e}")
        cleanup_ray()
        raise

    final_acc = round_log[-1][1] if round_log else float("nan")
    print(f"  Federated R15 acc: {final_acc:.4f}  ({time.time()-t0:.0f}s so far)")

    # 2. Reconstruct target model
    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
    target_model = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODINGS[enc_name]
    )
    target_model.pca      = shared_pca
    target_model._scaler  = shared_scaler
    target_model.set_weights(model_out[0])
    target_acc = float((target_model.predict(X_test) == y_test).mean())
    print(f"  Reconstructed acc: {target_acc:.4f}")

    # 3. Q-SHAP for members and non-members
    print(f"  Q-SHAP members ({N_MIA_SAMPLES} samples) …")
    exps_in  = _qshap_for_samples(target_model, X_train,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)
    print(f"  Q-SHAP non-members ({N_MIA_SAMPLES} samples) …")
    exps_out = _qshap_for_samples(target_model, X_test,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)
    print(f"  exps_in={exps_in.shape}  exps_out={exps_out.shape}")

    # 4. Direct MIA (cross-validated LR)
    direct_auc = run_mia_attack(exps_in, exps_out, seed=seed, cv=5)
    print(f"  Direct MIA AUC: {direct_auc:.4f}")

    wall_time = time.time() - t0
    append_result(enc_name, seed, target_acc, direct_auc,
                  len(exps_in), len(exps_out), wall_time)
    mark_done(rk)
    print(f"  [DONE]  {wall_time:.0f}s  ({wall_time/60:.1f} min)")


# ── analysis: combine Phase 13 + extension → n=15 AUC ranking ────────────────

def print_analysis():
    print("\n" + "=" * 72)
    print("  RQ4 LEAKAGE RANKING ANALYSIS — n=15 combined")
    print("=" * 72)

    # Load Phase 13 (existing 5 seeds)
    old_aucs = {enc: [] for enc in ENCODINGS}
    if os.path.exists(PHASE13_CSV):
        with open(PHASE13_CSV) as fh:
            for row in csv.DictReader(fh):
                enc = row["encoding"]
                if enc in old_aucs:
                    old_aucs[enc].append((int(row["seed"]), float(row["direct_mia_auc"])))

    # Load new seeds
    new_aucs = {enc: [] for enc in ENCODINGS}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as fh:
            for row in csv.DictReader(fh):
                enc = row["encoding"]
                if enc in new_aucs:
                    new_aucs[enc].append((int(row["seed"]), float(row["direct_mia_auc"])))

    # Merge
    threshold_5  = 0.07 / (5  ** 0.5)
    threshold_15 = 0.07 / (15 ** 0.5)
    print(f"\nReliability threshold:  n=5→{threshold_5*100:.2f}pp   n=15→{threshold_15*100:.2f}pp")

    enc_stats = {}
    print(f"\n{'Encoding':<12}  {'n':>3}  {'Mean AUC':>10}  {'Std AUC':>9}")
    for enc in ENCODINGS:
        combined = sorted(old_aucs[enc] + new_aucs[enc], key=lambda x: x[0])
        aucs     = [a for _, a in combined]
        if not aucs:
            continue
        mean = statistics.mean(aucs)
        std  = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        enc_stats[enc] = {"mean": mean, "std": std, "n": len(aucs),
                           "aucs": aucs}
        old_seeds = sorted(s for s, _ in old_aucs[enc])
        new_seeds = sorted(s for s, _ in new_aucs[enc])
        print(f"  {enc:<12}  {len(aucs):>3}  {mean:>10.4f}  {std:>9.4f}  "
              f"(old seeds={old_seeds}, new seeds={new_seeds})")

    print()
    enc_list = list(ENCODINGS.keys())
    print(f"{'Pair':<30}  {'Δ (pp)':>10}  {'n=5 status':>14}  {'n=15 status':>16}")
    print("-" * 74)
    for i, e1 in enumerate(enc_list):
        for e2 in enc_list[i + 1:]:
            if e1 not in enc_stats or e2 not in enc_stats:
                continue
            diff = abs(enc_stats[e1]["mean"] - enc_stats[e2]["mean"])
            s5  = "RESOLVED" if diff > threshold_5  else "unresolved"
            s15 = "RESOLVED" if diff > threshold_15 else "UNRESOLVED (tie)"
            print(f"  |{e1} - {e2}|{'':<18}  {diff*100:>10.2f}  {s5:>14}  {s15:>16}")

    print()
    ranked = sorted(enc_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    print("Ranking by mean direct MIA AUC (highest = most leakage):")
    for rank, (enc, st) in enumerate(ranked, 1):
        flag = " ← reuploading-highest supported?" if st["mean"] == max(s["mean"] for s in enc_stats.values()) else ""
        print(f"  {rank}. {enc:<12}  AUC={st['mean']:.4f} ± {st['std']:.4f}  "
              f"(n={st['n']}){flag}")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  RQ4 SEED EXTENSION SWEEP — Direct MIA, Wisconsin IID")
print(f"  Existing seeds: {EXISTING_SEEDS}")
print(f"  New seeds:      {NEW_SEEDS}")
print(f"  New runs:       {len(NEW_SEEDS) * len(ENCODINGS)}")
print(f"  Shadow MIA: OMITTED (confirmed to fail at n=5)")
print("=" * 72 + "\n")

for enc_name in ENCODINGS:
    for seed in NEW_SEEDS:
        run_one(enc_name, seed)

print_analysis()
