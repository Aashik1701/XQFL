"""Phase 12: Amplitude/Wisconsin/α=0.1 collapse replication.

Resolves the MN-08c amplitude comparison gap from Phase 11.  Runs 3 independent
draws for amplitude/Wisconsin/α=0.1/seeds 42 and 999 (6 total), logging per-draw
accuracy and full explanation vectors to a permanent CSV.

These seeds showed COLLAPSED category in Phase 10 RQ3 replication, but per-draw
accuracy values were never logged to any CSV (scratch-script-only run).  This
script closes that gap and determines whether MN-08c (within-collapse bifurcation)
is amplitude-specific or a general phenomenon.

Context from Phase 11 replication data:
  angle/α=0.1/seed=42  : draws [63.16%, 46.49%, 46.49%]  range=16.7pp  all B
  angle/α=0.1/seed=999 : draws [36.84%, 63.16%, 63.16%]  range=26.3pp  all B
  reuploading/α=0.1/seed=999: draws [64.91%, 63.16%, 44.74%]  range=20.2pp  all B
  IQP/α=0.1/seed=42    : draws [63.16%, 39.47%, 57.89%]  range=23.7pp  all B
  IQP/α=0.1/seed=123   : draws [35.96%, 85.09%(A), 64.91%]  range=49.1pp  crosses A/B
  IQP/α=0.1/seed=999   : draws [63.16%, 53.51%, 75.44%(A)]  range=21.9pp  crosses A/B

  → within-B variance is NOT angle-specific; amplitude comparison is the remaining gap.

Infrastructure rule (established Phase 12): every replication run logs per-draw
accuracy and explanation vectors to a permanent results CSV — no exceptions.

Outputs
-------
  results/phase12_log.csv        — per-draw rows (phase11 schema + draw_num column)
  results/phase12_checkpoint.csv — completed run keys for restart safety

Runtime estimate (MN-07 cost model):
  6 draws × ~165s = ~16.5 min (amplitude is slowest due to StatePrep)
"""

import os, sys, csv, json, math, signal, subprocess, time
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import amplitude_encoding
from federation.run_federation import run_federation

# ── constants ──────────────────────────────────────────────────────────────────

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
N_DRAWS          = 3
RUN_TIMEOUT      = 720    # 12 min — amplitude (StatePrep) runs ~10% slower than angle

DIRICHLET_ALPHA = 0.1
DISTRIBUTION    = "alpha_0.1"
ENCODING        = "amplitude"
TARGET_SEEDS    = [42, 999]

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "phase12_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "phase12_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed", "draw_num",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# Phase 11 angle reference draws for MN-08c comparison (reported 2026-08-02)
ANGLE_DRAWS = {
    42:  [0.6316, 0.4649, 0.4649],   # range=16.7pp all B
    999: [0.3684, 0.6316, 0.6316],   # range=26.3pp all B
}

# ── data + thresholds ──────────────────────────────────────────────────────────

print("Loading Wisconsin (breast cancer) dataset …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + 0.02
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Majority-class baseline : {majority_baseline:.4f}  ({majority_baseline*100:.2f}%)")
print(f"  Collapse threshold (A/B): {collapse_threshold:.4f}  ({collapse_threshold*100:.2f}%)")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ─────────────────────────────────────────────────────────

def make_run_key(seed, draw_num):
    return f"amplitude|alpha_0.1|{seed}|draw{draw_num}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("done", "timeout"):
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


def cleanup_ray():
    subprocess.run(["pkill", "-f", r"ray.*worker"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray/core.*raylet"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray.*log_monitor"], capture_output=True)
    time.sleep(3)

# ── results CSV helpers ────────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(seed, draw_num, round_num, client_id, explanation_vector, accuracy):
    margin   = float(accuracy) - collapse_threshold
    track    = "A" if float(accuracy) > collapse_threshold else "B"
    norm_phi = float(np.linalg.norm(explanation_vector))
    vec      = [float(v) for v in np.asarray(explanation_vector).flatten()]
    with open(RESULTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=RESULTS_COLS).writerow({
            "run_type":           "replication",
            "encoding":           ENCODING,
            "distribution":       DISTRIBUTION,
            "dirichlet_alpha":    DIRICHLET_ALPHA,
            "seed":               seed,
            "draw_num":           draw_num,
            "round":              round_num,
            "client_id":          client_id,
            "explanation_vector": json.dumps(vec),
            "accuracy":           f"{float(accuracy):.6f}",
            "track_class":        track,
            "margin":             f"{margin:.6f}",
            "norm_phi":           f"{norm_phi:.6f}",
            "timestamp":          datetime.now(timezone.utc).isoformat(),
        })


class Phase12Logger:
    def __init__(self, seed, draw_num):
        self.seed     = seed
        self.draw_num = draw_num

    def log(self, round_num, client_id, explanation_vector, accuracy):
        append_row(self.seed, self.draw_num, round_num, client_id,
                   explanation_vector, float(accuracy))


# ── federated draw ─────────────────────────────────────────────────────────────

def run_draw(seed, draw_num):
    rk   = make_run_key(seed, draw_num)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  DRAW {draw_num}/{N_DRAWS}  amplitude / alpha_0.1 / seed={seed}")
    print(f"  Run key: {rk}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase12Logger(seed, draw_num)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded {RUN_TIMEOUT}s — Ray actor deadlock suspected")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=amplitude_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,   n_layers=N_LAYERS,
            split="dirichlet",
            dirichlet_alpha=DIRICHLET_ALPHA,
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
        print(f"\n  [TIMEOUT] {rk}: {e}")
        mark_done(rk, status="timeout")
        cleanup_ray()
    except Exception as e:
        signal.alarm(0)
        print(f"\n  [ERROR] {rk}: {e}")
        raise


# ── analysis ───────────────────────────────────────────────────────────────────

def load_results():
    done = load_checkpoint()
    rows = []
    if not os.path.exists(RESULTS_CSV):
        return rows
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rk = make_run_key(int(row["seed"]), int(row["draw_num"]))
            if rk not in done:
                continue
            row["round"]    = int(row["round"])
            row["accuracy"] = float(row["accuracy"])
            row["seed"]     = int(row["seed"])
            row["draw_num"] = int(row["draw_num"])
            try:
                row["client_id"] = int(row["client_id"])
            except (ValueError, TypeError):
                pass
            rows.append(row)
    return rows


def _r15_acc(rows, seed, draw_num):
    """Return R15 global accuracy for one draw, or None."""
    r15 = [
        r for r in rows
        if r["seed"] == seed
        and r["draw_num"] == draw_num
        and r["client_id"] == "global"
        and r["round"] == N_ROUNDS
    ]
    return r15[-1]["accuracy"] if r15 else None


def print_phase12_analysis(rows):
    print("\n" + "#"*70)
    print("PHASE 12  AMPLITUDE / WISCONSIN / α=0.1  COLLAPSE REPLICATION")
    print(f"  3 draws × seeds [42, 999]  |  MN-08c amplitude comparison")
    print(f"  Collapse threshold: {collapse_threshold:.4f}  "
          f"(majority {majority_baseline:.4f} + 2pp)")
    print(f"  Track A = functional (acc > threshold)  |  Track B = collapsed")
    print("#"*70)

    for seed in TARGET_SEEDS:
        print(f"\n  --- seed={seed} ---")
        draw_accs   = {}
        draw_tracks = {}
        for draw_num in range(1, N_DRAWS + 1):
            acc = _r15_acc(rows, seed, draw_num)
            if acc is not None:
                trk = "A" if acc > collapse_threshold else "B"
                draw_accs[draw_num]   = acc
                draw_tracks[draw_num] = trk
                print(f"    draw{draw_num}: {acc*100:.2f}%  Track {trk}")
            else:
                print(f"    draw{draw_num}: (no data)")

        if len(draw_accs) >= 2:
            accs = list(draw_accs.values())
            rng  = (max(accs) - min(accs)) * 100
            tracks = list(draw_tracks.values())
            n_a = tracks.count("A")
            n_b = tracks.count("B")
            print(f"\n    draws complete : {len(accs)}/{N_DRAWS}")
            print(f"    range          : {rng:.1f}pp")
            print(f"    Track A/B      : {n_a}A / {n_b}B")

            # MN-08c comparison
            if seed in ANGLE_DRAWS:
                ang_accs = ANGLE_DRAWS[seed]
                ang_rng  = (max(ang_accs) - min(ang_accs)) * 100
                print(f"\n    Phase 11 angle/α=0.1/seed={seed}: range={ang_rng:.1f}pp  all B")
                if rng > 10.0:
                    if n_a == 0:
                        verdict = f"within-B variance CONFIRMED — range={rng:.1f}pp"
                    else:
                        verdict = f"full bifurcation — range={rng:.1f}pp  {n_a}A/{n_b}B"
                    print(f"    verdict: {verdict}")
                    print(f"    MN-08c is GENERAL — confirmed for angle, reuploading, IQP, and amplitude")
                elif rng > 5.0:
                    print(f"    verdict: ELEVATED ({rng:.1f}pp) but below 10pp threshold — ambiguous")
                else:
                    print(f"    verdict: within noise floor ({rng:.1f}pp) — amplitude may cluster at single attractor")
                    print(f"    If confirmed with all 3 draws: MN-08c may be angle/reuploading/IQP-specific")

    # Summary table
    print(f"\n  {'Seed':>5}  {'Draw1':>10}  {'Draw2':>10}  {'Draw3':>10}  {'Range':>8}  {'A/B':>5}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*5}")
    for seed in TARGET_SEEDS:
        vals   = []
        tracks = []
        for draw_num in range(1, N_DRAWS + 1):
            acc = _r15_acc(rows, seed, draw_num)
            if acc is not None:
                trk = "A" if acc > collapse_threshold else "B"
                vals.append(f"{acc*100:.2f}%({trk})")
                tracks.append((acc, trk))
            else:
                vals.append("—")
        while len(vals) < 3:
            vals.append("—")
        if len(tracks) >= 2:
            acc_nums = [a for a, _ in tracks]
            rng_str  = f"{(max(acc_nums) - min(acc_nums))*100:.1f}pp"
            ab_str   = f"{sum(1 for _,t in tracks if t=='A')}A/{sum(1 for _,t in tracks if t=='B')}B"
        else:
            rng_str = "—"
            ab_str  = "—"
        print(f"  {seed:>5}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}  {rng_str:>8}  {ab_str:>5}")

    # Reference: other encodings at α=0.1 for comparison
    print(f"\n  Reference — other encodings at α=0.1 (from Phase 11, collapsed seeds):")
    print(f"  angle/seed=42   : [63.16%, 46.49%, 46.49%]  range=16.7pp  all B")
    print(f"  angle/seed=999  : [36.84%, 63.16%, 63.16%]  range=26.3pp  all B")
    print(f"  reup/seed=999   : [64.91%, 63.16%, 44.74%]  range=20.2pp  all B")
    print(f"  IQP/seed=42     : [63.16%, 39.47%, 57.89%]  range=23.7pp  all B")
    print(f"  IQP/seed=123    : [35.96%, 85.09%(A), 64.91%]  range=49.1pp  crosses A/B")
    print(f"  IQP/seed=999    : [63.16%, 53.51%, 75.44%(A)]  range=21.9pp  crosses A/B")


# ── main ───────────────────────────────────────────────────────────────────────

init_results_csv()

print(f"\n{'#'*70}")
print(f"  Phase 12: Amplitude Collapse Replication  |  Wisconsin  |  MN-08c")
print(f"  Encoding: amplitude  |  Distribution: alpha_0.1  |  Seeds: {TARGET_SEEDS}")
print(f"  Draws per seed: {N_DRAWS}  |  Total runs: {len(TARGET_SEEDS) * N_DRAWS}")
print(f"  Timeout: {RUN_TIMEOUT}s per run  |  Checkpoint: per draw")
est_min = len(TARGET_SEEDS) * N_DRAWS * 165 // 60
print(f"  Runtime estimate: {len(TARGET_SEEDS) * N_DRAWS} × ~165s = ~{est_min} min")
print(f"  Results    → {RESULTS_CSV}")
print(f"  Checkpoint → {CHECKPOINT_CSV}")
already_done = load_checkpoint()
print(f"  Already completed: {len(already_done)} checkpoint entries")
print(f"{'#'*70}")

for seed in TARGET_SEEDS:
    for draw_num in range(1, N_DRAWS + 1):
        run_draw(seed, draw_num)

rows = load_results()
print(f"\nTotal rows loaded from phase12_log.csv: {len(rows)}")
print_phase12_analysis(rows)

print(f"\n{'='*70}")
print("[Phase 12 complete]")
print(f"  Results    : {RESULTS_CSV}")
print(f"  Checkpoint : {CHECKPOINT_CSV}")
print(f"{'='*70}")
