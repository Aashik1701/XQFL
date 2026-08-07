"""Angle encoding G1-partition falsification test.

Tests the claim that angle encoding ALWAYS collapses (Track B) under G1/G2-
flagged partitions (MN-08).  The stated falsification criterion is a single
Track-A draw for angle under any G1/G2-flagged condition.

Previous evidence (Phase 10/11/16): 12 Wisconsin draws + 41 Pima non-IID draws,
all Track B — but all sourced from the same 5 RNG seeds (1/7/42/123/999) whose
Dirichlet-sampled partitions happened to flag G1/G2.

This experiment uses MANUALLY CONSTRUCTED G1 partitions that are independent of
those five seeds, providing genuinely new partition geometries.

Wisconsin — 3 G1 geometries (class-1/benign dominant, 455 train samples):
  G1-W1: client0  61% weight, 100% purity  (277 cls1, 0 cls0)
  G1-W2: client1  65% weight,  93% purity  (274 cls1, 21 cls0)
  G1-W3: client2  68% weight,  91% purity  (281 cls1, 28 cls0)

Pima — 2 G1 geometries (class-0/non-diabetic dominant, 614 train samples):
  G1-P1: client0  62% weight, 100% purity  (380 cls0, 0 cls1)
  G1-P2: client0  66% weight,  91% purity  (368 cls0, 37 cls1)

Each partition × 3 draws (actor-RNG varies per draw; Python seed=200 fixed).
Track A threshold = majority-class baseline + 2pp (per MN-08 protocol).

Outputs:
  results/angle_falsification_results.csv   — one row per draw
  results/angle_falsification_checkpoint.csv
"""

import os, sys, csv, time
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data, load_pima_data
from models.encodings import angle_encoding
from federation.run_federation import run_federation

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS      = 4
N_LAYERS      = 2
N_CLIENTS     = 4
N_ROUNDS      = 15
LOCAL_EPOCHS  = 3
N_DRAWS       = 3
INIT_SEED     = 200       # not in original [1, 7, 42, 123, 999] pool
COLLAPSE_MARGIN = 0.02

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "angle_falsification_results.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "angle_falsification_checkpoint.csv")

# ── data ──────────────────────────────────────────────────────────────────────

print("Loading Wisconsin …")
Xw_tr, Xw_te, yw_tr, yw_te = load_breast_cancer_data(random_state=42)
w_majority  = float(np.bincount(yw_te.astype(int)).max()) / len(yw_te)
w_threshold = w_majority + COLLAPSE_MARGIN
print(f"  train={len(Xw_tr)} (cls0={int((yw_tr==0).sum())} cls1={int((yw_tr==1).sum())})")
print(f"  majority={w_majority:.4f}  Track-A threshold={w_threshold:.4f}")

print("Loading Pima …")
Xp_tr, Xp_te, yp_tr, yp_te = load_pima_data(random_state=42)
p_majority  = float(np.bincount(yp_te.astype(int)).max()) / len(yp_te)
p_threshold = p_majority + COLLAPSE_MARGIN
print(f"  train={len(Xp_tr)} (cls0={int((yp_tr==0).sum())} cls1={int((yp_tr==1).sum())})")
print(f"  majority={p_majority:.4f}  Track-A threshold={p_threshold:.4f}")

# ── partition construction ─────────────────────────────────────────────────────

def make_g1_partition(X_train, y_train, dominant_client, dom_class,
                       dom_n_maj, dom_n_min, n_clients=4, dist_seed=0):
    """Build a G1-flagged partition with a manually specified dominant client.

    dominant_client : which client index gets the concentrated slice
    dom_class       : label of the dominant class (0 or 1)
    dom_n_maj       : number of dominant-class samples for the dominant client
    dom_n_min       : number of minority-class samples for the dominant client
    dist_seed       : numpy RNG seed for shuffling before assignment
    """
    rng = np.random.default_rng(dist_seed)

    idx_maj = np.where(y_train == dom_class)[0].copy()
    idx_min = np.where(y_train != dom_class)[0].copy()
    rng.shuffle(idx_maj)
    rng.shuffle(idx_min)

    dom_idx  = np.concatenate([idx_maj[:dom_n_maj], idx_min[:dom_n_min]])
    rest_idx = np.concatenate([idx_maj[dom_n_maj:], idx_min[dom_n_min:]])
    rng.shuffle(dom_idx)
    rng.shuffle(rest_idx)

    other_clients = [i for i in range(n_clients) if i != dominant_client]
    rest_splits   = np.array_split(rest_idx, len(other_clients))

    partitions = [None] * n_clients
    partitions[dominant_client] = (X_train[dom_idx], y_train[dom_idx])
    for oc, sl in zip(other_clients, rest_splits):
        partitions[oc] = (X_train[sl], y_train[sl])

    return partitions


# Wisconsin G1 geometries — class-1 (benign) is majority (285/455)
WISCONSIN_G1 = {
    "G1-W1": dict(
        X_tr=Xw_tr, y_tr=yw_tr, X_te=Xw_te, y_te=yw_te,
        threshold=w_threshold, dataset="wisconsin",
        dominant_client=0, dom_class=1, dom_n_maj=277, dom_n_min=0,
        # weight=60.9%, purity=100%
    ),
    "G1-W2": dict(
        X_tr=Xw_tr, y_tr=yw_tr, X_te=Xw_te, y_te=yw_te,
        threshold=w_threshold, dataset="wisconsin",
        dominant_client=1, dom_class=1, dom_n_maj=274, dom_n_min=21,
        # weight=64.8%, purity=92.9%
    ),
    "G1-W3": dict(
        X_tr=Xw_tr, y_tr=yw_tr, X_te=Xw_te, y_te=yw_te,
        threshold=w_threshold, dataset="wisconsin",
        dominant_client=2, dom_class=1, dom_n_maj=281, dom_n_min=28,
        # weight=67.9%, purity=90.9%
    ),
}

# Pima G1 geometries — class-0 (non-diabetic) is majority (400/614)
PIMA_G1 = {
    "G1-P1": dict(
        X_tr=Xp_tr, y_tr=yp_tr, X_te=Xp_te, y_te=yp_te,
        threshold=p_threshold, dataset="pima",
        dominant_client=0, dom_class=0, dom_n_maj=380, dom_n_min=0,
        # weight=61.9%, purity=100%
    ),
    "G1-P2": dict(
        X_tr=Xp_tr, y_tr=yp_tr, X_te=Xp_te, y_te=yp_te,
        threshold=p_threshold, dataset="pima",
        dominant_client=0, dom_class=0, dom_n_maj=368, dom_n_min=37,
        # weight=66.0%, purity=90.9%
    ),
}

ALL_PARTITIONS = {**WISCONSIN_G1, **PIMA_G1}

# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "done":
                done.add(row["run_key"])
    return done


def mark_done(rk, acc, track):
    first = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "accuracy", "track", "status", "completed_at"])
        if first:
            w.writeheader()
        w.writerow({
            "run_key":      rk,
            "accuracy":     f"{acc:.4f}",
            "track":        track,
            "status":       "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [checkpoint] done: {rk}  acc={acc:.4f}  {track}")


def append_result(partition_name, dataset, draw, dom_client, dom_class,
                   dom_n_maj, dom_n_min, weight_pct, purity_pct,
                   acc, track, threshold):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "partition", "dataset", "draw", "dominant_client", "dominant_class",
            "dom_n_maj", "dom_n_min", "weight_pct", "purity_pct",
            "seed", "accuracy", "track", "threshold",
        ])
        if first:
            w.writeheader()
        w.writerow({
            "partition":        partition_name,
            "dataset":          dataset,
            "draw":             draw,
            "dominant_client":  dom_client,
            "dominant_class":   dom_class,
            "dom_n_maj":        dom_n_maj,
            "dom_n_min":        dom_n_min,
            "weight_pct":       f"{weight_pct:.1f}",
            "purity_pct":       f"{purity_pct:.1f}",
            "seed":             INIT_SEED,
            "accuracy":         f"{acc:.4f}",
            "track":            track,
            "threshold":        f"{threshold:.4f}",
        })

# ── run logic ─────────────────────────────────────────────────────────────────

def describe_partition(pname, cfg):
    n_total = cfg["dom_n_maj"] + cfg["dom_n_min"]
    n_train = len(cfg["X_tr"])
    weight  = 100.0 * n_total / n_train
    purity  = 100.0 * cfg["dom_n_maj"] / n_total if n_total > 0 else 0.0
    return weight, purity


def run_partition(pname, cfg, draw):
    rk   = f"angle|{pname}|seed{INIT_SEED}|draw{draw}"
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return None

    weight, purity = describe_partition(pname, cfg)
    n_train = len(cfg["X_tr"])
    n_total_dom = cfg["dom_n_maj"] + cfg["dom_n_min"]

    print(f"\n{'='*72}")
    print(f"  angle / {pname} / draw {draw}  "
          f"(client{cfg['dominant_client']} dominant: "
          f"cls{cfg['dom_class']} weight={weight:.1f}%, purity={purity:.1f}%)")
    print(f"  dataset={cfg['dataset']}  seed={INIT_SEED}  "
          f"threshold={cfg['threshold']:.4f}")
    print(f"{'='*72}")

    # Build partition (dist_seed=draw ensures consistent data-level shuffles
    # across draws while still being deterministic per draw)
    partitions = make_g1_partition(
        cfg["X_tr"], cfg["y_tr"],
        dominant_client=cfg["dominant_client"],
        dom_class=cfg["dom_class"],
        dom_n_maj=cfg["dom_n_maj"],
        dom_n_min=cfg["dom_n_min"],
        n_clients=N_CLIENTS,
        dist_seed=draw,
    )

    for i, (Xi, yi) in enumerate(partitions):
        dom_marker = " [G1 DOMINANT]" if i == cfg["dominant_client"] else ""
        print(f"  Client {i}: {len(Xi):4d} samples  "
              f"(cls1 ratio={yi.mean():.2f}){dom_marker}")

    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)

    _, round_log = run_federation(
        X_train=cfg["X_tr"], y_train=cfg["y_tr"],
        X_test=cfg["X_te"],  y_test=cfg["y_te"],
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        seed=INIT_SEED,
        project_root=PROJECT_ROOT,
        n_permutations=0,          # no Q-SHAP — accuracy only
        precomputed_partitions=partitions,
    )

    final_acc  = round_log[-1][1] if round_log else 0.0
    track      = "Track A" if final_acc > cfg["threshold"] else "Track B"
    above_flag = " *** TRACK A — CLAIM FALSIFIED ***" if track == "Track A" else ""
    print(f"\n  Final R15 accuracy: {final_acc:.4f}  → {track}{above_flag}")

    append_result(
        pname, cfg["dataset"], draw,
        cfg["dominant_client"], cfg["dom_class"],
        cfg["dom_n_maj"], cfg["dom_n_min"],
        weight, purity,
        final_acc, track, cfg["threshold"],
    )
    mark_done(rk, final_acc, track)
    return (pname, draw, final_acc, track)

# ── main ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

print("\n" + "="*72)
print("  ANGLE G1-PARTITION FALSIFICATION TEST")
print(f"  Partitions: {list(ALL_PARTITIONS.keys())}")
print(f"  Draws per partition: {N_DRAWS}  |  Init seed: {INIT_SEED}")
print(f"  Falsification criterion: ANY single Track-A draw")
print("="*72 + "\n")

results = []
for pname, cfg in ALL_PARTITIONS.items():
    for draw in range(1, N_DRAWS + 1):
        out = run_partition(pname, cfg, draw)
        if out is not None:
            results.append(out)

# ── summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("  FALSIFICATION TEST — SUMMARY")
print("="*72)

# Read all results (including resumed ones from checkpoint)
all_results = []
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            all_results.append(row)

track_a_rows = [r for r in all_results if r["track"] == "Track A"]
track_b_rows = [r for r in all_results if r["track"] == "Track B"]

print(f"\nTotal draws completed: {len(all_results)}")
print(f"  Track A: {len(track_a_rows)}")
print(f"  Track B: {len(track_b_rows)}")
print()

for pname in ALL_PARTITIONS:
    rows = [r for r in all_results if r["partition"] == pname]
    accs = [float(r["accuracy"]) for r in rows]
    tracks = [r["track"] for r in rows]
    print(f"  {pname:8s}  draws={len(rows):2d}  "
          f"Track A={tracks.count('Track A')}/{len(tracks)}  "
          f"acc=[{min(accs):.4f}–{max(accs):.4f}]")

print()
if track_a_rows:
    print("*** FALSIFICATION: claim does NOT hold — angle reached Track A on:")
    for r in track_a_rows:
        print(f"    {r['partition']} draw {r['draw']}: acc={r['accuracy']} (threshold={r['threshold']})")
    print("\nConclusion: soften to 'angle shows elevated G1/G2 vulnerability, not absolute failure.'")
else:
    n_total = len(all_results)
    print(f"CLAIM HOLDS: angle is 0/{n_total} Track A across all new G1 partitions.")
    print("This extends the previous 12-draw Wisconsin evidence to genuinely new")
    print("partition geometries and adds 2 Pima G1 geometries — strengthening the")
    print("'angle uniquely all-B under G1/G2' finding from flagged to well-tested.")
print()
