"""Phase 11: Cross-encoding non-IID explanation sweep (RQ3 extension).

Sweep design
------------
Extends Phase 10 Phase B (amplitude × non-IID × 5 seeds) to the three remaining
encodings — amplitude is excluded (already logged in phase10_log.csv):

  angle      × {α=0.5, α=0.1, quantity-skew} × [1, 7, 42, 123, 999]  = 15 runs
  reuploading × {α=0.5, α=0.1, quantity-skew} × [1, 7, 42, 123, 999] = 15 runs
  IQP        × {α=0.5, α=0.1, quantity-skew} × [1, 7, 42, 123, 999]  = 15 runs
  Primary total: 45 federated runs
  Replication buffer: up to +20 runs (MN-08 proactive protocol)

No new centralized or IID runs needed — all reference vectors come from
phase10_log.csv (Phase 10 Phase A).

MN-08 proactive flagging — two independent triggers (MN-08, 2026-07-30):

  Trigger G (geometry, pre-computed at startup, deterministic):
    G1: any client FedAvg weight >= 60% AND class purity >= 90%
        (matches Wisconsin/alpha_0.1/seeds 42,999 risk signature)
    G2: any single client holds >= 90% of one class's total training samples

  Trigger S (statistical, applied after all 5 primary seeds run per condition):
    |seed_accuracy - condition_mean| > 2 * encoding_IID_std
    IID std (from Phase 10 Phase A, MN-03a): angle=3.06pp, reup=0.88pp, iqp=3.94pp

When either trigger fires for a seed x condition, 2 additional replication runs
are executed inline before that condition's results are reported.  Replication
run keys are tagged 'replication|...' and stored in the same checkpoint.

A 'replcheck|{enc}|{dist}' checkpoint key marks that a condition's flag
evaluation and all consequent replication runs have completed.  On resume, if
replcheck is set the condition's replication block is skipped entirely.

JSD analysis outputs (two reference columns, both from phase10_log.csv):
  local_global_jsd  — JS-divergence between per-client and global explanation
                      vectors within the same non-IID round (Track A only).
                      Measures within-round explanation heterogeneity under non-IID.
  jsd_vs_iid_fed    — JSD between this condition's R15 global explanation and the
                      IID-federated R15 global (same encoding x seed, Phase 10).
                      Measures non-IID explanation drift from the clean baseline.
  jsd_vs_central    — JSD between this condition's R15 global explanation and the
                      centralized baseline (same encoding x seed, Phase 10).
                      Connects back to RQ1's fidelity framing.

Outputs
-------
  results/phase11_log.csv        — all explanation rows (same schema as phase10_log.csv)
  results/phase11_checkpoint.csv — completed run keys for restart safety

Runtime estimate (MN-07 cost model):
  45 primary x ~110s = ~83 min baseline; +20 replication x ~110s = ~37 min
  Total worst-case: ~120 min (~2 hours), 10-13 background passes
"""

import os, sys, csv, json, math, signal, subprocess, time
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from data.partition import iid_split, dirichlet_split, quantity_skew_split
from models.encodings import angle_encoding, reuploading_encoding, iqp_encoding
from federation.run_federation import run_federation
from explain.metrics import (
    kendalls_tau, cosine_similarity, l2_distance, js_divergence,
)

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
N_REPLICATION_REPS = 2         # additional reps per flagged seed
COLLAPSE_MARGIN    = 0.02      # threshold = majority_baseline + 2pp
GHOST_THRESHOLD    = 20        # samples below which a client is a 'ghost'
RUN_TIMEOUT        = 600       # seconds before a hung Ray actor is killed (10 min)

SEEDS = [1, 7, 42, 123, 999]

ENCODINGS = {
    "angle":       angle_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

# IID within-seed std per encoding (from Phase 10 Phase A, MN-03a, 5-seed Wisconsin)
ENCODING_IID_STD = {
    "angle":       0.0306,   # 3.06pp
    "reuploading": 0.0088,   # 0.88pp
    "iqp":         0.0394,   # 3.94pp
}

DISTRIBUTIONS = {
    "alpha_0.5": {"split": "dirichlet", "alpha": 0.5},
    "alpha_0.1": {"split": "dirichlet", "alpha": 0.1},
    "quantity":  {"split": "quantity",  "alpha": None},
}

PHASE10_LOG     = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")
RESULTS_CSV     = os.path.join(PROJECT_ROOT, "results", "phase11_log.csv")
CHECKPOINT_CSV  = os.path.join(PROJECT_ROOT, "results", "phase11_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data + thresholds ─────────────────────────────────────────────────────────

print("Loading Wisconsin (breast cancer) dataset …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Majority-class baseline : {majority_baseline:.4f}")
print(f"  Collapse threshold (A/B): {collapse_threshold:.4f}")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(run_type, encoding, distribution, seed, rep=None):
    """Canonical checkpoint key.  rep is None for primary runs."""
    if rep is None:
        return f"{run_type}|{encoding}|{distribution}|{seed}"
    return f"{run_type}|{encoding}|{distribution}|{seed}|rep{rep}"


def make_replcheck_key(encoding, distribution):
    """Marks that replication evaluation for enc x dist is complete."""
    return f"replcheck|{encoding}|{distribution}"


def load_checkpoint():
    """Return set of run keys that should be skipped (done or timed-out)."""
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("done", "timeout"):
                done.add(row["run_key"])
    return done


def timed_out_seeds(encoding_name, distribution_name):
    """Return set of seeds whose primary run timed out (to skip replication)."""
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    timed = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "timeout":
                rk = row["run_key"]
                parts = rk.split("|")
                if (len(parts) >= 4
                        and parts[0] == "federated"
                        and parts[1] == encoding_name
                        and parts[2] == distribution_name):
                    try:
                        timed.add(int(parts[3]))
                    except ValueError:
                        pass
    return timed


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
    """Kill any lingering Ray worker processes from a previous hung run."""
    subprocess.run(["pkill", "-f", r"ray.*worker"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray/core.*raylet"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray.*log_monitor"], capture_output=True)
    time.sleep(3)  # give OS time to reap workers before next Ray init

# ── results CSV helpers ───────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(run_type, encoding, distribution, dirichlet_alpha, seed,
               round_num, client_id, explanation_vector, accuracy):
    margin   = float(accuracy) - collapse_threshold
    track    = "A" if float(accuracy) > collapse_threshold else "B"
    norm_phi = float(np.linalg.norm(explanation_vector))
    vec      = [float(v) for v in np.asarray(explanation_vector).flatten()]
    row = {
        "run_type":           run_type,
        "encoding":           encoding,
        "distribution":       distribution,
        "dirichlet_alpha":    "" if dirichlet_alpha is None else dirichlet_alpha,
        "seed":               seed,
        "round":              round_num,
        "client_id":          client_id,
        "explanation_vector": json.dumps(vec),
        "accuracy":           f"{float(accuracy):.6f}",
        "track_class":        track,
        "margin":             f"{margin:.6f}",
        "norm_phi":           f"{norm_phi:.6f}",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=RESULTS_COLS).writerow(row)


# ── logger class ──────────────────────────────────────────────────────────────

class Phase11Logger:
    """Duck-typed for run_federation's logger parameter."""
    def __init__(self, run_type, encoding, distribution, dirichlet_alpha, seed):
        self.run_type        = run_type
        self.encoding        = encoding
        self.distribution    = distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.seed            = seed

    def log(self, round_num, client_id, explanation_vector, accuracy):
        append_row(
            self.run_type, self.encoding, self.distribution, self.dirichlet_alpha,
            self.seed, round_num, client_id, explanation_vector, float(accuracy),
        )


# ── partition geometry analysis (MN-08 Trigger G) ────────────────────────────

def get_partition(distribution_name, seed):
    """Replicate run_federation's _partition_data call exactly."""
    dist = DISTRIBUTIONS[distribution_name]
    if dist["split"] == "dirichlet":
        return dirichlet_split(X_train, y_train, N_CLIENTS,
                               alpha=dist["alpha"], seed=seed)
    elif dist["split"] == "quantity":
        return quantity_skew_split(X_train, y_train, N_CLIENTS,
                                   skew_factor=0.5, seed=seed)
    else:
        return iid_split(X_train, y_train, N_CLIENTS, seed=seed)


def partition_stats(partition):
    """Per-client stats needed for geometry flag."""
    n_total   = sum(len(yc) for _, yc in partition)
    total_c1  = int((y_train == 1).sum())
    total_c0  = int((y_train == 0).sum())
    stats = []
    for i, (_, yc) in enumerate(partition):
        n_c    = len(yc)
        c1     = int((yc == 1).sum())
        c0     = int((yc == 0).sum())
        weight = n_c / n_total if n_total > 0 else 0.0
        c1_rat = c1 / n_c if n_c > 0 else float("nan")
        purity = max(c1_rat, 1 - c1_rat) if n_c > 0 else 0.0
        stats.append({
            "client":   i,
            "n":        n_c,
            "c1":       c1,
            "c0":       c0,
            "weight":   weight,
            "c1_ratio": c1_rat,
            "purity":   purity,
            "c1_share": c1 / total_c1 if total_c1 > 0 else 0.0,
            "c0_share": c0 / total_c0 if total_c0 > 0 else 0.0,
            "ghost":    n_c < GHOST_THRESHOLD,
        })
    return stats


def geometry_flag(partition):
    """Return (is_flagged: bool, reasons: list[str]).

    MN-08 Trigger G:
      G1: any client with FedAvg weight >= 60% AND class purity >= 90%
      G2: any client holds >= 90% of one class's total training samples
    """
    stats   = partition_stats(partition)
    reasons = []
    for s in stats:
        if s["ghost"]:
            continue
        # G1: weight x purity
        if s["weight"] >= 0.60 and s["purity"] >= 0.90:
            reasons.append(
                f"G1: C{s['client']} weight={s['weight']:.1%} purity={s['purity']:.1%}"
                f" (c1_ratio={s['c1_ratio']:.2f})"
            )
        # G2: class dominance
        if s["c1_share"] >= 0.90:
            reasons.append(
                f"G2: C{s['client']} holds {s['c1_share']:.1%} of class-1 samples"
                f" (n={s['n']})"
            )
        if s["c0_share"] >= 0.90:
            reasons.append(
                f"G2: C{s['client']} holds {s['c0_share']:.1%} of class-0 samples"
                f" (n={s['n']})"
            )
    return bool(reasons), reasons


# Pre-compute geometry flags for all 15 partition×seed combos at startup.
# (Partitions are independent of encoding — same X_train/y_train/seed/split.)
print("\nPre-computing partition geometry for MN-08 Trigger G …")
GEOMETRY_FLAGS = {}   # {dist_name: {seed: (bool, [reasons])}}
for dist_name in DISTRIBUTIONS:
    GEOMETRY_FLAGS[dist_name] = {}
    for seed in SEEDS:
        part = get_partition(dist_name, seed)
        flagged, reasons = geometry_flag(part)
        GEOMETRY_FLAGS[dist_name][seed] = (flagged, reasons)

print(f"\n  {'Dist':>10}  {'Seed':>5}  {'GeoFlag':>8}  Reasons")
print(f"  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*40}")
for dist_name in DISTRIBUTIONS:
    for seed in SEEDS:
        f, rs = GEOMETRY_FLAGS[dist_name][seed]
        tag   = "FLAG" if f else "ok"
        r_str = rs[0] if rs else "—"
        print(f"  {dist_name:>10}  {seed:>5}  {tag:>8}  {r_str}")
        for r in rs[1:]:
            print(f"  {' ':>10}  {' ':>5}  {' ':>8}  {r}")
print()


# ── federated run ─────────────────────────────────────────────────────────────

def run_federated(encoding_name, distribution_name, seed, run_type="federated", rep=None):
    """Execute one federated run.  run_type='replication' for MN-08 reps."""
    rk   = make_run_key(run_type, encoding_name, distribution_name, seed, rep)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    dist = DISTRIBUTIONS[distribution_name]
    rep_tag = f" (rep {rep})" if rep is not None else ""
    print(f"\n{'='*70}")
    print(f"  {run_type.upper()}  {encoding_name} / {distribution_name} / seed={seed}{rep_tag}")
    if GEOMETRY_FLAGS[distribution_name][seed][0]:
        print(f"  [MN-08 GeoFlag] {GEOMETRY_FLAGS[distribution_name][seed][1]}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase11Logger(run_type, encoding_name, distribution_name, dist["alpha"], seed)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded {RUN_TIMEOUT}s — Ray actor deadlock suspected")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=ENCODINGS[encoding_name],
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,   n_layers=N_LAYERS,
            split=dist["split"],
            dirichlet_alpha=dist["alpha"] if dist["alpha"] is not None else 0.5,
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
        signal.alarm(0)  # cancel alarm on clean exit
        mark_done(rk)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"\n  [TIMEOUT] {rk}: {e}")
        print(f"  Marking as timeout and cleaning up Ray workers …")
        mark_done(rk, status="timeout")
        cleanup_ray()
    except Exception as e:
        signal.alarm(0)
        print(f"\n  [ERROR] {rk}: {e}")
        raise


# ── MN-08 statistical flag + replication ────────────────────────────────────

def get_primary_final_accs(encoding_name, distribution_name, rows):
    """R15 global accuracy per seed for primary federated runs."""
    accs = {}
    for seed in SEEDS:
        seed_rows = [r for r in rows
                     if r["run_type"] == "federated"
                     and r["encoding"] == encoding_name
                     and r["distribution"] == distribution_name
                     and r["seed"] == seed
                     and r["client_id"] == "global"
                     and r["round"] == N_ROUNDS]
        if seed_rows:
            accs[seed] = seed_rows[-1]["accuracy"]
    return accs


def statistical_flags(encoding_name, distribution_name, accs):
    """Return set of seeds flagged by Trigger S.

    Flag if |seed_acc - condition_mean| > 2 * encoding_IID_std.
    """
    if len(accs) < 2:
        return set()
    iid_std   = ENCODING_IID_STD[encoding_name]
    threshold = 2.0 * iid_std
    mean_acc  = float(np.mean(list(accs.values())))
    flagged   = set()
    for seed, acc in accs.items():
        if abs(acc - mean_acc) > threshold:
            print(f"  [MN-08 StatFlag] {encoding_name}/{distribution_name}/seed={seed}: "
                  f"acc={acc:.4f}  mean={mean_acc:.4f}  "
                  f"dev={abs(acc-mean_acc)*100:.2f}pp > 2*std={threshold*100:.2f}pp")
            flagged.add(seed)
    return flagged


def check_and_replicate(encoding_name, distribution_name, rows):
    """Check MN-08 flags and run replication reps for any triggered seeds.

    Skips entire block if replcheck key already in checkpoint (resume safety).
    Sets replcheck key after all replication runs complete.
    """
    replcheck_key = make_replcheck_key(encoding_name, distribution_name)
    done = load_checkpoint()
    if replcheck_key in done:
        print(f"  [skip replcheck] {replcheck_key}")
        return

    # Determine which seeds to replicate
    accs          = get_primary_final_accs(encoding_name, distribution_name, rows)
    geo_seeds     = {s for s in SEEDS if GEOMETRY_FLAGS[distribution_name][s][0]}
    stat_seeds    = statistical_flags(encoding_name, distribution_name, accs)
    to_replicate  = geo_seeds | stat_seeds

    # Never replicate seeds whose primary run timed out — they will deadlock again
    timeout_seeds = timed_out_seeds(encoding_name, distribution_name)
    if timeout_seeds:
        print(f"  [skip replication] timed-out seeds (deadlock risk): {sorted(timeout_seeds)}")
        to_replicate -= timeout_seeds

    if not to_replicate:
        print(f"  [no MN-08 flags] {encoding_name}/{distribution_name} — skipping replication")
        mark_done(replcheck_key)
        return

    print(f"\n  [MN-08 replication] {encoding_name}/{distribution_name}")
    print(f"    Geometry-flagged  : {sorted(geo_seeds)}")
    print(f"    Statistical-flagged: {sorted(stat_seeds)}")
    if timeout_seeds:
        print(f"    Timed-out (skipped): {sorted(timeout_seeds)}")
    print(f"    Combined ({len(to_replicate)} seeds): {sorted(to_replicate)}")
    print(f"    Running {N_REPLICATION_REPS} additional reps per flagged seed …")

    for seed in sorted(to_replicate):
        for rep in range(1, N_REPLICATION_REPS + 1):
            run_federated(encoding_name, distribution_name, seed,
                          run_type="replication", rep=rep)

    mark_done(replcheck_key)


# ── load Phase 10 reference vectors ──────────────────────────────────────────

def load_phase10_refs():
    """Load IID-federated R15 global and centralized vectors per encoding x seed.

    Returns dict: refs[encoding][seed] = {"iid_fed": vec, "centralized": vec}
    Vectors are numpy arrays of shape (N_QUBITS,).
    """
    refs = {enc: {} for enc in ENCODINGS}

    if not os.path.exists(PHASE10_LOG):
        print(f"  WARNING: {PHASE10_LOG} not found — JSD reference columns will be empty")
        return refs

    with open(PHASE10_LOG, newline="") as fh:
        for row in csv.DictReader(fh):
            enc  = row["encoding"]
            if enc not in ENCODINGS:
                continue
            seed = int(row["seed"])
            rtyp = row["run_type"]
            dist = row["distribution"]
            cid  = row["client_id"]

            if seed not in refs[enc]:
                refs[enc][seed] = {}

            if (rtyp == "federated" and dist == "iid"
                    and cid == "global" and int(row["round"]) == N_ROUNDS):
                refs[enc][seed]["iid_fed"] = np.array(
                    json.loads(row["explanation_vector"]))

            if (rtyp == "centralized" and cid == "global"):
                refs[enc][seed]["centralized"] = np.array(
                    json.loads(row["explanation_vector"]))

    return refs


# ── load Phase 11 results ─────────────────────────────────────────────────────

def load_results():
    """Load phase11_log.csv; only include rows from checkpointed runs."""
    done = load_checkpoint()
    rows = []
    if not os.path.exists(RESULTS_CSV):
        return rows
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rtyp = row["run_type"]
            enc  = row["encoding"]
            dist = row["distribution"]
            seed = int(row["seed"])
            # Determine the run key this row belongs to
            if rtyp == "replication":
                # Check any rep key for this condition (any rep that's done)
                any_done = any(
                    make_run_key("replication", enc, dist, seed, rep=r) in done
                    for r in range(1, N_REPLICATION_REPS + 1)
                )
                if not any_done:
                    continue
            else:
                rk = make_run_key(rtyp, enc, dist, seed)
                if rk not in done:
                    continue
            row["round"]            = int(row["round"])
            row["accuracy"]         = float(row["accuracy"])
            row["margin"]           = float(row["margin"])
            row["norm_phi"]         = float(row["norm_phi"])
            row["seed"]             = seed
            row["dirichlet_alpha"]  = (float(row["dirichlet_alpha"])
                                       if row["dirichlet_alpha"] else None)
            row["explanation_vector"] = json.loads(row["explanation_vector"])
            try:
                row["client_id"] = int(row["client_id"])
            except (ValueError, TypeError):
                pass
            rows.append(row)
    return rows


# ── analysis helpers ──────────────────────────────────────────────────────────

def _seed_stats(rows, encoding_name, distribution_name, seed, run_type="federated"):
    """Per-seed: final acc, Track A/B counts, mean local-global JSD (Track A)."""
    seed_rows = [r for r in rows
                 if r["run_type"] == run_type
                 and r["encoding"] == encoding_name
                 and r["distribution"] == distribution_name
                 and r["seed"] == seed]
    if not seed_rows:
        return None

    global_rows = sorted(
        [r for r in seed_rows if r["client_id"] == "global"],
        key=lambda x: x["round"],
    )
    client_by_round = {}
    for r in seed_rows:
        if r["client_id"] != "global":
            client_by_round.setdefault(r["round"], []).append(r)

    n_a = n_b = 0
    track_a_jsds = []
    final_acc    = float("nan")
    final_phi    = None

    for gr in global_rows:
        rnd = gr["round"]
        if rnd == 0:
            continue
        trk = gr["track_class"]
        if trk == "A":
            n_a += 1
            if rnd in client_by_round:
                jsds = [
                    js_divergence(cr["explanation_vector"], gr["explanation_vector"])
                    for cr in client_by_round[rnd]
                ]
                jsds = [v for v in jsds if not math.isnan(v)]
                if jsds:
                    track_a_jsds.append(float(np.mean(jsds)))
        elif trk == "B":
            n_b += 1
        if rnd == N_ROUNDS:
            final_acc = gr["accuracy"]
            final_phi = gr["explanation_vector"]

    mean_jsd = float(np.mean(track_a_jsds)) if track_a_jsds else float("nan")
    return {
        "final_acc":   final_acc,
        "n_a":         n_a,
        "n_b":         n_b,
        "mean_jsd":    mean_jsd,
        "final_phi":   final_phi,
        "n_jsd_rounds": len(track_a_jsds),
    }


def _rep_final_accs(rows, encoding_name, distribution_name, seed):
    """Collect R15 global accuracy for each completed replication run.

    Each rep writes exactly one R15 global row; return all of them.
    rows is already checkpoint-filtered so partial runs won't contribute R15 rows.
    """
    return [
        r["accuracy"] for r in rows
        if r["run_type"] == "replication"
        and r["encoding"] == encoding_name
        and r["distribution"] == distribution_name
        and r["seed"] == seed
        and r["client_id"] == "global"
        and r["round"] == N_ROUNDS
    ]


# ── main analysis ─────────────────────────────────────────────────────────────

def print_phase11_analysis(rows, refs):
    print("\n" + "#"*70)
    print("PHASE 11  CROSS-ENCODING NON-IID EXPLANATION SWEEP")
    print("  Wisconsin / FedAvg / 3 encodings × 3 distributions × 5 seeds")
    print(f"  Collapse threshold: {collapse_threshold:.4f}  "
          f"(majority {majority_baseline:.4f} + {COLLAPSE_MARGIN*100:.0f}pp)")
    print("  Track A = functional (acc > threshold)  |  Track B = collapsed")
    print("  JSD columns — local_global: within-round client-global JSD (Track A)")
    print("              — vs_iid_fed : R15 global vs IID-fed R15 (same enc×seed)")
    print("              — vs_central : R15 global vs centralized (same enc×seed)")
    print("#"*70)

    for enc in ENCODINGS:
        print(f"\n{'='*70}")
        print(f"  ENCODING: {enc.upper()}")
        print(f"  IID noise floor (MN-02/06): ±{ENCODING_IID_STD[enc]*100:.2f}pp std  "
              f"| stat-flag threshold: >{ENCODING_IID_STD[enc]*200:.2f}pp from cond. mean")
        print(f"{'='*70}")

        for dist_name in DISTRIBUTIONS:
            print(f"\n  --- {dist_name} ---")

            # Collect per-seed stats
            per_seed = {}
            for seed in SEEDS:
                st = _seed_stats(rows, enc, dist_name, seed)
                if st is None:
                    continue

                # JSD vs IID-federated R15
                jsd_iid = float("nan")
                jsd_cen = float("nan")
                if st["final_phi"] is not None:
                    phi = np.array(st["final_phi"])
                    if seed in refs.get(enc, {}):
                        r = refs[enc][seed]
                        if "iid_fed" in r:
                            jsd_iid = js_divergence(phi, r["iid_fed"])
                        if "centralized" in r:
                            jsd_cen = js_divergence(phi, r["centralized"])

                per_seed[seed] = {**st, "jsd_iid": jsd_iid, "jsd_cen": jsd_cen}

            if not per_seed:
                print(f"    (no data)")
                continue

            # Per-seed table
            geo_flag_seeds  = {s for s in SEEDS if GEOMETRY_FLAGS[dist_name][s][0]}
            done_ck         = load_checkpoint()
            stat_accs       = {s: v["final_acc"] for s, v in per_seed.items()
                               if not math.isnan(v["final_acc"])}
            cond_mean        = float(np.mean(list(stat_accs.values()))) if stat_accs else float("nan")
            iid_std          = ENCODING_IID_STD[enc]
            stat_flag_seeds  = {s for s, a in stat_accs.items()
                                if abs(a - cond_mean) > 2 * iid_std}
            all_flagged      = geo_flag_seeds | stat_flag_seeds

            print(f"  {'Seed':>5}  {'FinalAcc':>9}  {'A/15':>5}  {'B/15':>5}  "
                  f"{'LG-JSD':>8}  {'vs_IID':>8}  {'vs_Cen':>8}  {'Flag':>8}")
            print(f"  {'-'*5}  {'-'*9}  {'-'*5}  {'-'*5}  "
                  f"{'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

            for seed in SEEDS:
                if seed not in per_seed:
                    to_status = "TIMEOUT" if seed in timed_out_seeds(enc, dist_name) else "no data"
                    print(f"  {seed:>5}  ({to_status})")
                    continue
                v = per_seed[seed]
                flags = []
                if seed in geo_flag_seeds:
                    flags.append("G")
                if seed in stat_flag_seeds:
                    flags.append("S")
                if seed in timed_out_seeds(enc, dist_name):
                    flags.append("T")
                flag_str = ",".join(flags) if flags else "—"
                lg_jsd   = f"{v['mean_jsd']:.4f}" if not math.isnan(v['mean_jsd']) else "n/a(B)"
                iid_j    = f"{v['jsd_iid']:.4f}" if not math.isnan(v['jsd_iid']) else "—"
                cen_j    = f"{v['jsd_cen']:.4f}" if not math.isnan(v['jsd_cen']) else "—"
                print(f"  {seed:>5}  {v['final_acc']*100:>8.2f}%  "
                      f"{v['n_a']:>5}  {v['n_b']:>5}  "
                      f"{lg_jsd:>8}  {iid_j:>8}  {cen_j:>8}  {flag_str:>8}")

            # 5-seed summary (primary only)
            if stat_accs:
                accs  = list(stat_accs.values())
                ddof  = 1 if len(accs) > 1 else 0
                lg_jsds = [per_seed[s]["mean_jsd"] for s in per_seed
                           if not math.isnan(per_seed[s]["mean_jsd"])]
                iid_jsds = [per_seed[s]["jsd_iid"] for s in per_seed
                            if not math.isnan(per_seed[s]["jsd_iid"])]
                cen_jsds = [per_seed[s]["jsd_cen"] for s in per_seed
                            if not math.isnan(per_seed[s]["jsd_cen"])]
                print(f"\n  5-seed summary (primary runs):")
                print(f"    Accuracy  : {np.mean(accs)*100:.2f}% ± {np.std(accs, ddof=ddof)*100:.2f}pp")
                if lg_jsds:
                    print(f"    LG-JSD    : {np.mean(lg_jsds):.4f} ± {np.std(lg_jsds, ddof=ddof):.4f}")
                if iid_jsds:
                    print(f"    vs_IID    : {np.mean(iid_jsds):.4f} ± {np.std(iid_jsds, ddof=ddof):.4f}")
                if cen_jsds:
                    print(f"    vs_Central: {np.mean(cen_jsds):.4f} ± {np.std(cen_jsds, ddof=ddof):.4f}")

            # Replication results for flagged seeds
            if all_flagged:
                print(f"\n  MN-08 replication results (flagged seeds: {sorted(all_flagged)}):")
                for seed in sorted(all_flagged):
                    rep_accs = _rep_final_accs(rows, enc, dist_name, seed)
                    prim_acc = per_seed.get(seed, {}).get("final_acc", float("nan"))
                    all_accs = [prim_acc] + rep_accs if not math.isnan(prim_acc) else rep_accs
                    if not all_accs:
                        print(f"    seed={seed}: (no replication data yet)")
                        continue
                    rng = (max(all_accs) - min(all_accs)) * 100
                    rep_str = " ".join(f"{a*100:.2f}%" for a in rep_accs)
                    flag_types = []
                    if seed in geo_flag_seeds:
                        flag_types.append("G: " + "; ".join(GEOMETRY_FLAGS[dist_name][seed][1]))
                    if seed in stat_flag_seeds:
                        flag_types.append(f"S: |{prim_acc*100:.2f}-{cond_mean*100:.2f}|pp={abs(prim_acc-cond_mean)*100:.2f}pp")
                    print(f"    seed={seed}: primary={prim_acc*100:.2f}%  reps=[{rep_str}]"
                          f"  range={rng:.1f}pp")
                    for ft in flag_types:
                        print(f"      flag: {ft}")
                    if len(all_accs) >= 2:
                        iid_nf = ENCODING_IID_STD[enc] * 100 * 2
                        if rng > 10:
                            verdict = f"MN-08 CONFIRMED — range {rng:.1f}pp >> IID floor"
                        elif rng > iid_nf:
                            verdict = f"ELEVATED variance ({rng:.1f}pp > {iid_nf:.1f}pp IID 2-sigma)"
                        else:
                            verdict = f"within noise floor ({rng:.1f}pp <= {iid_nf:.1f}pp)"
                        print(f"      verdict: {verdict}")

    # ── cross-encoding comparison ───────────────────────────────────────────
    print("\n" + "="*70)
    print("CROSS-ENCODING COMPARISON  (5-seed mean accuracy by distribution)")
    print(f"  {'':>12}  {'α=0.5':>10}  {'α=0.1':>10}  {'qty-skew':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for enc in ENCODINGS:
        vals = []
        for dist_name in DISTRIBUTIONS:
            accs = [_seed_stats(rows, enc, dist_name, s)["final_acc"]
                    for s in SEEDS
                    if _seed_stats(rows, enc, dist_name, s) is not None
                    and not math.isnan(_seed_stats(rows, enc, dist_name, s)["final_acc"])]
            if accs:
                vals.append(f"{np.mean(accs)*100:.1f}%±{np.std(accs, ddof=1)*100:.1f}pp")
            else:
                vals.append("—")
        print(f"  {enc:<12}  {vals[0]:>10}  {vals[1]:>10}  {vals[2]:>10}")

    # Amplitude reference (from Phase 10)
    print(f"  {'amplitude':>12}  (see phase10_log.csv — amplitude Phase B reference)")

    print(f"\n  MN-03a hypothesis check:")
    print(f"  Under non-IID, angle and IQP were predicted hurt by adversarial partitions")
    print(f"  (z≈-1.7 at IID); amplitude was helped (z=+1.4). Reuploading was neutral.")
    print(f"  See per-encoding tables above for whether this ordering holds under label skew.")


# ── main ──────────────────────────────────────────────────────────────────────

init_results_csv()

_primary_total = len(ENCODINGS) * len(DISTRIBUTIONS) * len(SEEDS)
print(f"\n{'#'*70}")
print(f"  Phase 11 Encoding × Non-IID Sweep  |  Wisconsin / FedAvg")
print(f"  Encodings: {list(ENCODINGS)}")
print(f"  Distributions: {list(DISTRIBUTIONS)}")
print(f"  Seeds: {SEEDS}")
print(f"  Primary runs: {_primary_total}  |  MN-08 buffer: up to +{_primary_total//2}")
print(f"  Results    → {RESULTS_CSV}")
print(f"  Checkpoint → {CHECKPOINT_CSV}")
already_done = load_checkpoint()
print(f"  Already completed: {len(already_done)} checkpoint entries")
print(f"{'#'*70}")

# Main sweep: for each encoding × distribution, run all 5 seeds,
# then immediately check MN-08 flags and run replication.
for enc in ENCODINGS:
    for dist_name in DISTRIBUTIONS:
        print(f"\n[SWEEP]  {enc} / {dist_name}")

        for seed in SEEDS:
            run_federated(enc, dist_name, seed)

        # Load current results for statistical flag computation
        current_rows = load_results()
        check_and_replicate(enc, dist_name, current_rows)

# Final analysis
rows = load_results()
refs = load_phase10_refs()
print(f"\nTotal rows loaded from phase11_log.csv: {len(rows)}")

print_phase11_analysis(rows, refs)

print(f"\n{'='*70}")
print("[Phase 11 complete]")
print(f"  Results    : {RESULTS_CSV}")
print(f"  Checkpoint : {CHECKPOINT_CSV}")
print(f"  Phase 10 refs: {PHASE10_LOG}")
