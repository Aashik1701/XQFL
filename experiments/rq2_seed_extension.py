"""RQ2 seed extension: Wisconsin IID explanation sweep, seeds 11–47.

Extends Phase 10's IID stability record from 5 to 15 seeds per encoding to test
whether the encoding-stability ranking (Kendall's tau, R1→R15, global) becomes
statistically resolvable.

Existing Phase 10 seeds:   [1, 7, 42, 123, 999]
New seeds (this script):   [11, 13, 17, 19, 23, 29, 31, 37, 43, 47]
Combined n per encoding:   15

Reliability threshold: 7pp / sqrt(n)
  n=5:  3.13pp — |angle-amplitude| = 1.9pp UNRESOLVED
                 |amplitude-reuploading| = 2.9pp UNRESOLVED
  n=15: 1.81pp — both pairs cross if the direction holds in new seeds

Protocol per (encoding, seed):
  - Wisconsin IID federated (4 clients, 15 rounds, 3 local epochs)
  - Q-SHAP logged every round for global model
  - Same schema as phase10_log.csv

Outputs:
  results/rq2_extension_log.csv
  results/rq2_extension_checkpoint.csv
"""

import os, sys, csv, json, signal, time
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import (angle_encoding, amplitude_encoding,
                               reuploading_encoding, iqp_encoding)
from federation.run_federation import run_federation

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS         = 4
N_LAYERS         = 2
N_CLIENTS        = 4
N_ROUNDS         = 15
LOCAL_EPOCHS     = 3
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20
N_CLIENT_EXPLAIN = 5
N_CLIENT_BG      = 10
COLLAPSE_MARGIN  = 0.02
RUN_TIMEOUT      = 600

EXISTING_SEEDS = [1, 7, 42, 123, 999]
NEW_SEEDS      = [11, 13, 17, 19, 23, 29, 31, 37, 43, 47]

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq2_extension_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq2_extension_checkpoint.csv")
PHASE10_CSV    = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data ──────────────────────────────────────────────────────────────────────

print("Loading Wisconsin dataset …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train={len(X_train)}  Test={len(y_test)}")
print(f"  majority={majority_baseline:.4f}  collapse_threshold={collapse_threshold:.4f}")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(encoding, seed):
    return f"federated|{encoding}|iid|{seed}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "done":
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


# ── logger (duck-typed for LoggingFedAvg) ─────────────────────────────────────

class ExtLogger:
    def __init__(self, encoding, seed):
        self.encoding = encoding
        self.seed     = seed

    def log(self, round_num, client_id, explanation_vector, accuracy):
        margin   = float(accuracy) - collapse_threshold
        track    = "A" if float(accuracy) > collapse_threshold else "B"
        norm_phi = float(np.linalg.norm(explanation_vector))
        vec      = [float(v) for v in np.asarray(explanation_vector).flatten()]
        row = {
            "run_type":           "federated",
            "encoding":           self.encoding,
            "distribution":       "iid",
            "dirichlet_alpha":    "",
            "seed":               self.seed,
            "round":              round_num,
            "client_id":          client_id,
            "explanation_vector": json.dumps(vec),
            "accuracy":           f"{float(accuracy):.6f}",
            "track_class":        track,
            "margin":             f"{margin:.6f}",
            "norm_phi":           f"{norm_phi:.6f}",
            "timestamp":          datetime.now(timezone.utc).isoformat(),
        }
        first = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
            if first:
                w.writeheader()
            w.writerow(row)


# ── timeout / Ray cleanup ─────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("Run timed out")


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
    print(f"  RQ2-EXT  {enc_name} / IID / seed={seed}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = ExtLogger(enc_name, seed)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        run_federation(
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
            logger=logger,
            n_global_explain=N_GLOBAL_EXPLAIN,
            n_global_bg=N_GLOBAL_BG,
            n_explain=N_CLIENT_EXPLAIN,
            n_bg=N_CLIENT_BG,
            n_permutations=N_PERM,
        )
        signal.alarm(0)
        mark_done(rk)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  [TIMEOUT] {e}")
        cleanup_ray()
        mark_done(rk, status="timeout")
    except Exception as e:
        signal.alarm(0)
        print(f"  [ERROR] {e}")
        cleanup_ray()
        raise


# ── analysis: combine Phase 10 + extension → n=15 tau ranking ────────────────

def _load_iid_global_phis(csv_path):
    """Return {enc: {seed: {round: phi}}} for IID global rows."""
    from collections import defaultdict
    data = defaultdict(lambda: defaultdict(dict))
    if not os.path.exists(csv_path):
        return data
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if (row.get("run_type") in ("federated",)
                    and row.get("distribution") == "iid"
                    and row.get("client_id") == "global"):
                enc  = row["encoding"]
                seed = int(row["seed"])
                rnd  = int(row["round"])
                phi  = json.loads(row["explanation_vector"])
                data[enc][seed][rnd] = phi
    return data


def kendalls_tau(a, b):
    n = len(a)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total if total > 0 else 0.0


def mean_tau_per_seed(rounds_dict):
    """Compute mean round-to-round Kendall's tau across rounds 1-15."""
    phis = [rounds_dict[r] for r in range(1, N_ROUNDS + 1) if r in rounds_dict]
    if len(phis) < 2:
        return None
    taus = [kendalls_tau(phis[i], phis[i + 1]) for i in range(len(phis) - 1)]
    return sum(taus) / len(taus)


def print_analysis():
    import statistics

    print("\n" + "=" * 72)
    print("  RQ2 STABILITY RANKING ANALYSIS — n=15 combined")
    print("=" * 72)

    old = _load_iid_global_phis(PHASE10_CSV)
    new = _load_iid_global_phis(RESULTS_CSV)

    # Merge
    merged = {}
    for enc in ENCODINGS:
        merged[enc] = {}
        for seed, rounds in old[enc].items():
            merged[enc][seed] = rounds
        for seed, rounds in new[enc].items():
            merged[enc][seed] = rounds

    threshold_5  = 0.07 / (5  ** 0.5)
    threshold_15 = 0.07 / (15 ** 0.5)
    print(f"\nReliability threshold:  n=5→{threshold_5*100:.2f}pp   n=15→{threshold_15*100:.2f}pp")

    enc_stats = {}
    print(f"\n{'Encoding':<12}  {'n':>3}  {'Mean τ':>8}  {'Std τ':>7}  {'Seeds(τ)'}")
    for enc in ENCODINGS:
        taus = []
        for seed, rounds in sorted(merged[enc].items()):
            t = mean_tau_per_seed(rounds)
            if t is not None:
                taus.append((seed, t))
        if not taus:
            continue
        vals = [t for _, t in taus]
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        enc_stats[enc] = {"mean": mean, "std": std, "n": len(vals)}
        seed_str = "  ".join(f"s{s}={t:.3f}" for s, t in sorted(taus))
        print(f"  {enc:<12}  {len(vals):>3}  {mean:>8.4f}  {std:>7.4f}  {seed_str}")

    print()
    enc_list = list(ENCODINGS.keys())
    print(f"{'Pair':<30}  {'Δτ (pp)':>10}  {'n=5 status':>14}  {'n=15 status':>14}")
    print("-" * 72)
    for i, e1 in enumerate(enc_list):
        for e2 in enc_list[i + 1:]:
            if e1 not in enc_stats or e2 not in enc_stats:
                continue
            diff = abs(enc_stats[e1]["mean"] - enc_stats[e2]["mean"])
            s5  = "RESOLVED" if diff > threshold_5  else "unresolved"
            s15 = "RESOLVED" if diff > threshold_15 else "UNRESOLVED (tie)"
            print(f"  |{e1} - {e2}|{'':<18}  {diff*100:>10.2f}  {s5:>14}  {s15:>14}")

    print()
    ranked = sorted(enc_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    print("Ranking by mean τ (highest = most stable):")
    for rank, (enc, st) in enumerate(ranked, 1):
        print(f"  {rank}. {enc:<12}  τ={st['mean']:.4f} ± {st['std']:.4f}  (n={st['n']})")


# ── main ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  RQ2 SEED EXTENSION SWEEP — Wisconsin IID")
print(f"  Existing seeds: {EXISTING_SEEDS}")
print(f"  New seeds:      {NEW_SEEDS}")
print(f"  New runs:       {len(NEW_SEEDS) * len(ENCODINGS)}")
print("=" * 72 + "\n")

for enc_name in ENCODINGS:
    for seed in NEW_SEEDS:
        run_one(enc_name, seed)

print_analysis()
