"""Phase 16 smoke test: validate Pima explanation pipeline before full sweep.

Runs amplitude × IID (Phase A, 2 seeds) then amplitude × α=0.1 (Phase B, 2 seeds)
to verify four things before launching the full ~115-run, ~3.5-hour sweep:

  CHECK (a) — Data pipeline: Pima loads to 614/154 split; 8 features → 4 via PCA;
              collapse threshold = 0.6694 (majority 64.94% + 2pp)
  CHECK (b) — Checkpoint resume: run-key skip logic works (interrupt after Phase A,
              restart — Phase A should be skipped, Phase B starts fresh)
  CHECK (c) — Trigger S uses live Phase A IID_std (computed from Phase A CSV outputs),
              not the Phase 9b bootstrap fallback (0.0096)
  CHECK (d) — G1/G2 geometry flags compute against Pima's X_train/y_train at α=0.1,
              not Wisconsin's partition structure

Outputs (smoke-test-specific, does NOT write to phase16_log.csv):
  results/phase16_smoke_log.csv        — explanation rows (8 runs max)
  results/phase16_smoke_checkpoint.csv — run-key checkpoint for resume testing

Note on CHECK (b): to fully exercise resume, interrupt during Phase A (Ctrl-C) and
restart the script. Phase A runs should be skipped; Phase B should proceed fresh.
"""

import os, sys, csv, json, signal, subprocess, time
import numpy as np
import torch
from datetime import datetime, timezone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_pima_data
from data.partition import dirichlet_split
from models.encodings import amplitude_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation
from explain.q_shap import q_shap_values, make_quantum_predict_fn

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS         = 4
N_LAYERS         = 2
N_CLIENTS        = 4
N_ROUNDS         = 15
LOCAL_EPOCHS     = 3
CENTRAL_EPOCHS   = 45
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20
N_CLIENT_EXPLAIN = 5
N_CLIENT_BG      = 10
N_REPLICATION_REPS = 2
COLLAPSE_MARGIN    = 0.02
GHOST_THRESHOLD    = 20
RUN_TIMEOUT        = 600

SMOKE_SEEDS = [42, 1]

# Phase 9b bootstrap — used as fallback ONLY if Phase A fails to produce 2+ seeds
PHASE9B_AMPLITUDE_IID_STD = 0.0096

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "phase16_smoke_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "phase16_smoke_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data loading ──────────────────────────────────────────────────────────────

print("Loading Pima dataset …")
X_train, X_test, y_train, y_test = load_pima_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN

print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Features: {X_train.shape[1]}  →  {N_QUBITS} via PCA")
print(f"  Test class counts: 0={int((y_test==0).sum())}  1={int((y_test==1).sum())}")
print(f"  Majority-class baseline:  {majority_baseline:.6f}  ({majority_baseline*100:.2f}%)")
print(f"  Collapse threshold (A/B): {collapse_threshold:.6f}  ({collapse_threshold*100:.2f}%)")
print()

# ── CHECK (a): validate expected Pima dimensions ──────────────────────────────

_a_checks = [
    ("train_size",    len(X_train),       614),
    ("test_size",     len(X_test),        154),
    ("n_features",    X_train.shape[1],     8),
    ("threshold_4dp", round(collapse_threshold, 4), 0.6694),
]
_a_failures = []
print("[CHECK (a)] Pima data pipeline:")
for name, got, want in _a_checks:
    ok = (got == want)
    if not ok:
        _a_failures.append(f"{name}: got={got} expected={want}")
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got}  expected={want}")
if _a_failures:
    raise RuntimeError(f"CHECK (a) FAILED — {_a_failures}")
print("  → CHECK (a) PASS\n")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(run_type, encoding, distribution, seed, rep=None):
    if rep is None:
        return f"{run_type}|{encoding}|{distribution}|{seed}"
    return f"{run_type}|{encoding}|{distribution}|{seed}|rep{rep}"


def make_replcheck_key(encoding, distribution):
    return f"replcheck|{encoding}|{distribution}"


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


def timed_out_seeds(distribution_name):
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    timed = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "timeout":
                parts = row["run_key"].split("|")
                if (len(parts) >= 4 and parts[0] == "federated"
                        and parts[2] == distribution_name):
                    try:
                        timed.add(int(parts[3]))
                    except ValueError:
                        pass
    return timed

# ── results CSV helpers ───────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(run_type, encoding, distribution, dirichlet_alpha, seed,
               round_num, client_id, explanation_vector, accuracy):
    margin   = float(accuracy) - collapse_threshold
    if run_type == "centralized":
        track = "centralized"
    else:
        track = "A" if float(accuracy) > collapse_threshold else "B"
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

# ── logger ────────────────────────────────────────────────────────────────────

class Phase16Logger:
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

# ── partition geometry (MN-08 Trigger G) ─────────────────────────────────────

def partition_stats(partition):
    n_total  = sum(len(yc) for _, yc in partition)
    total_c1 = int((y_train == 1).sum())
    total_c0 = int((y_train == 0).sum())
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
    stats   = partition_stats(partition)
    reasons = []
    for s in stats:
        if s["ghost"]:
            continue
        if s["weight"] >= 0.60 and s["purity"] >= 0.90:
            reasons.append(
                f"G1: C{s['client']} weight={s['weight']:.1%} purity={s['purity']:.1%}"
                f" (c1_ratio={s['c1_ratio']:.2f})"
            )
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


# ── CHECK (d): pre-compute G1/G2 flags for Pima α=0.1 × smoke seeds ──────────

print("[CHECK (d)] Pre-computing G1/G2 geometry flags for Pima α=0.1 …")
print(f"  Using Pima X_train/y_train (n={len(X_train)}, classes: "
      f"0={int((y_train==0).sum())} / 1={int((y_train==1).sum())})")
print(f"  Expected: total ~614 samples across 4 clients, class split ~65/35")
print()

GEOMETRY_FLAGS = {}
for seed in SMOKE_SEEDS:
    part = dirichlet_split(X_train, y_train, N_CLIENTS, alpha=0.1, seed=seed)
    flagged, reasons = geometry_flag(part)
    GEOMETRY_FLAGS[seed] = (flagged, reasons)

    stats = partition_stats(part)
    print(f"  α=0.1 / seed={seed}  [{'FLAG' if flagged else 'clean'}]")
    print(f"    {'Cl':>4}  {'n':>5}  {'c0':>5}  {'c1':>5}  {'weight':>8}  {'purity':>8}")
    for s in stats:
        ghost = " [ghost]" if s["ghost"] else ""
        print(f"    C{s['client']:>3}  {s['n']:>5}  {s['c0']:>5}  {s['c1']:>5}  "
              f"{s['weight']:>7.1%}  {s['purity']:>7.1%}{ghost}")
    if reasons:
        for r in reasons:
            print(f"    → {r}")
    else:
        print(f"    → G1/G2: NONE")
    print()

print("  Inspect the per-client counts above: they should sum to ~614 with")
print("  Dirichlet-skewed class distributions at α=0.1 (severe label imbalance).")
print("  If they sum to ~455 (Wisconsin's train size) or show 30/70 class split,")
print("  the geometry check is using the wrong dataset — CHECK (d) FAILS.")
print()

# ── Ray cleanup ────────────────────────────────────────────────────────────────

def cleanup_ray():
    subprocess.run(["pkill", "-f", r"ray.*worker"],      capture_output=True)
    subprocess.run(["pkill", "-f", r"ray/core.*raylet"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray.*log_monitor"], capture_output=True)
    time.sleep(3)

# ── federated run ─────────────────────────────────────────────────────────────

def run_federated(encoding_name, distribution_name, seed, dirichlet_alpha=None,
                  run_type="federated", rep=None):
    rk   = make_run_key(run_type, encoding_name, distribution_name, seed, rep)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}  ← CHECK (b): resume working if this is a restart")
        return None

    rep_tag  = f" (rep {rep})" if rep is not None else ""
    geo_flag = GEOMETRY_FLAGS.get(seed, (False, []))
    print(f"\n{'='*70}")
    print(f"  {run_type.upper()}  {encoding_name} / {distribution_name} / seed={seed}{rep_tag}")
    if run_type in ("federated", "replication") and distribution_name != "iid" and geo_flag[0]:
        print(f"  [MN-08 GeoFlag] {geo_flag[1]}")
    print(f"{'='*70}")

    if distribution_name == "iid":
        split_kw = dict(split="iid", dirichlet_alpha=0.5)
    else:
        split_kw = dict(split="dirichlet", dirichlet_alpha=dirichlet_alpha)

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase16Logger(
        run_type, encoding_name, distribution_name,
        dirichlet_alpha if distribution_name != "iid" else None,
        seed,
    )

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded {RUN_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=amplitude_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,   n_layers=N_LAYERS,
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            logger=logger,
            n_global_explain=N_GLOBAL_EXPLAIN,
            n_global_bg=N_GLOBAL_BG,
            n_explain=N_CLIENT_EXPLAIN,
            n_bg=N_CLIENT_BG,
            n_permutations=N_PERM,
            **split_kw,
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


def run_centralized(encoding_name, seed):
    rk   = make_run_key("centralized", encoding_name, "iid", seed)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}  ← CHECK (b): resume working if this is a restart")
        return

    print(f"\n{'='*70}")
    print(f"  CENTRALIZED  {encoding_name} / seed={seed}  ({CENTRAL_EPOCHS} epochs)")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    pca    = PCA(n_components=N_QUBITS, random_state=seed)
    X_pca  = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_pca)

    model         = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=amplitude_encoding
    )
    model.pca     = pca
    model._scaler = scaler
    model.fit(X_train, y_train, epochs=CENTRAL_EPOCHS, lr=0.05, batch_size=32)

    acc = float((model.predict(X_test) == y_test).mean())
    print(f"  Centralized test accuracy: {acc:.4f}  "
          f"({'Track A' if acc > collapse_threshold else 'Track B'})")

    X_test_q     = model._preprocess(X_test[:N_GLOBAL_EXPLAIN])
    X_bg_q       = model._preprocess(X_train[:N_GLOBAL_BG])
    q_predict    = make_quantum_predict_fn(model)
    qshap        = q_shap_values(q_predict, X_test_q, X_bg_q,
                                 n_permutations=N_PERM, seed=seed)
    global_qshap = qshap.mean(axis=0)
    print(f"  Centralized ‖φ‖: {np.linalg.norm(global_qshap):.4f}")

    append_row("centralized", encoding_name, "iid", None, seed,
               15, "global", global_qshap, acc)
    mark_done(rk)

# ── Phase A: IID baseline ─────────────────────────────────────────────────────

print("=" * 70)
print("PHASE A: amplitude × IID × seeds", SMOKE_SEEDS)
print("=" * 70)
init_results_csv()

for seed in SMOKE_SEEDS:
    run_federated("amplitude", "iid", seed)
    run_centralized("amplitude", seed)

# ── CHECK (c): compute live IID_std from Phase A results ─────────────────────

print(f"\n[CHECK (c)] Computing live amplitude/Pima IID_std from Phase A outputs …")

iid_accs  = {}
_done_ck  = load_checkpoint()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rk = make_run_key("federated", row["encoding"], row["distribution"],
                              int(row["seed"]))
            if (rk in _done_ck
                    and row["run_type"] == "federated"
                    and row["encoding"] == "amplitude"
                    and row["distribution"] == "iid"
                    and row["client_id"] == "global"
                    and int(row["round"]) == N_ROUNDS):
                iid_accs[int(row["seed"])] = float(row["accuracy"])

if len(iid_accs) >= 2:
    _vals      = list(iid_accs.values())
    live_mean  = float(np.mean(_vals))
    live_std   = float(np.std(_vals, ddof=1))
    trig_s_thr = 2.0 * live_std
    print(f"  Phase A amplitude/IID: n={len(iid_accs)}")
    for s, a in sorted(iid_accs.items()):
        t = "A" if a > collapse_threshold else "B"
        print(f"    seed={s}  acc={a:.4f}  track={t}  margin={a-collapse_threshold:+.4f}")
    print(f"  Live mean={live_mean:.4f}  live_std={live_std:.4f}  "
          f"(Phase 9b bootstrap was {PHASE9B_AMPLITUDE_IID_STD:.4f})")
    print(f"  Trigger S threshold = 2 × {live_std:.4f} = {trig_s_thr:.4f} ({trig_s_thr*100:.2f}pp)")
    print(f"  → [CHECK (c)] PASS — Trigger S will use live_std={live_std:.4f}, not bootstrap")
    amplitude_iid_std  = live_std
    check_c_pass       = True
else:
    print(f"  WARNING: only {len(iid_accs)} Phase A seed(s) in checkpoint "
          f"(some runs may have timed out or failed)")
    print(f"  Falling back to Phase 9b bootstrap std={PHASE9B_AMPLITUDE_IID_STD:.4f}")
    print(f"  → [CHECK (c)] FALLBACK — verify Phase A completed before drawing conclusions")
    live_mean          = collapse_threshold  # neutral default for reporting only
    amplitude_iid_std  = PHASE9B_AMPLITUDE_IID_STD
    check_c_pass       = False

# ── Phase B: amplitude × α=0.1 ───────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"PHASE B: amplitude × α=0.1 × seeds {SMOKE_SEEDS}")
print(f"  Trigger S std: {amplitude_iid_std:.4f} ({amplitude_iid_std*100:.2f}pp) "
      f"— {'LIVE Phase A' if check_c_pass else 'Phase 9b bootstrap'}")
print(f"{'='*70}")

for seed in SMOKE_SEEDS:
    run_federated("amplitude", "alpha_0.1", seed, dirichlet_alpha=0.1)

# ── MN-08 replication check ───────────────────────────────────────────────────

def check_and_replicate():
    replcheck_key = make_replcheck_key("amplitude", "alpha_0.1")
    done          = load_checkpoint()
    if replcheck_key in done:
        print(f"  [skip replcheck] {replcheck_key}  "
              f"← CHECK (b): full replication block skipped on resume")
        return

    # Gather Phase B R15 global accuracies from completed runs
    phase_b_accs = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                rk = make_run_key("federated", row["encoding"],
                                  row["distribution"], int(row["seed"]))
                if (rk in done
                        and row["run_type"] == "federated"
                        and row["encoding"] == "amplitude"
                        and row["distribution"] == "alpha_0.1"
                        and row["client_id"] == "global"
                        and int(row["round"]) == N_ROUNDS):
                    phase_b_accs[int(row["seed"])] = float(row["accuracy"])

    print(f"\n  Phase B amplitude/α=0.1 final accuracies:")
    for s, a in sorted(phase_b_accs.items()):
        t = "A" if a > collapse_threshold else "B"
        delta = a - live_mean
        print(f"    seed={s}  acc={a:.4f}  track={t}  Δiid_mean={delta:+.4f}")

    # Trigger G
    geo_seeds = {s for s in SMOKE_SEEDS if GEOMETRY_FLAGS.get(s, (False,))[0]}

    # Trigger S — using live std (or bootstrap if Phase A failed)
    trig_s_thr = 2.0 * amplitude_iid_std
    stat_seeds = set()
    if len(phase_b_accs) >= 2:
        b_mean = float(np.mean(list(phase_b_accs.values())))
        for seed, acc in phase_b_accs.items():
            dev = abs(acc - b_mean)
            if dev > trig_s_thr:
                print(f"  [MN-08 StatFlag] amplitude/α=0.1/seed={seed}: "
                      f"acc={acc:.4f}  b_mean={b_mean:.4f}  "
                      f"dev={dev*100:.2f}pp > thr={trig_s_thr*100:.2f}pp")
                stat_seeds.add(seed)

    to_replicate = (geo_seeds | stat_seeds) - timed_out_seeds("alpha_0.1")

    print(f"\n  MN-08 flag summary:")
    print(f"    Geometry-flagged seeds:    {sorted(geo_seeds) or 'none'}")
    print(f"    Statistical-flagged seeds: {sorted(stat_seeds) or 'none'}")
    print(f"    Seeds to replicate:        {sorted(to_replicate) or 'none'}")

    if to_replicate:
        print(f"    Running {N_REPLICATION_REPS} replication draws per flagged seed …")
        for seed in sorted(to_replicate):
            for rep in range(1, N_REPLICATION_REPS + 1):
                run_federated("amplitude", "alpha_0.1", seed,
                              dirichlet_alpha=0.1, run_type="replication", rep=rep)
    else:
        print(f"    No replication triggered.")

    mark_done(replcheck_key)
    print(f"\n  Replcheck key written. Restarting the script should skip this block.")

check_and_replicate()

# ── Final smoke test summary ──────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SMOKE TEST SUMMARY")
print(f"{'='*70}")

done_final = load_checkpoint()
fed_a_done  = [make_run_key("federated",   "amplitude", "iid",       s) in done_final
               for s in SMOKE_SEEDS]
cent_a_done = [make_run_key("centralized", "amplitude", "iid",       s) in done_final
               for s in SMOKE_SEEDS]
fed_b_done  = [make_run_key("federated",   "amplitude", "alpha_0.1", s) in done_final
               for s in SMOKE_SEEDS]
repl_done   = make_replcheck_key("amplitude", "alpha_0.1") in done_final

total_rows = 0
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as fh:
        total_rows = sum(1 for _ in csv.DictReader(fh))

print(f"\n  Phase A federated   ({len(SMOKE_SEEDS)} runs):  "
      f"{sum(fed_a_done)}/{len(SMOKE_SEEDS)}  done")
print(f"  Phase A centralized ({len(SMOKE_SEEDS)} runs):  "
      f"{sum(cent_a_done)}/{len(SMOKE_SEEDS)}  done")
print(f"  Phase B federated   ({len(SMOKE_SEEDS)} runs):  "
      f"{sum(fed_b_done)}/{len(SMOKE_SEEDS)}  done")
print(f"  Replication block closed:   {'yes' if repl_done else 'no'}")
print(f"  Total CSV rows logged:      {total_rows}")

geo_results = {s: GEOMETRY_FLAGS[s] for s in SMOKE_SEEDS}
print(f"\n  CHECK (a): Pima load         → PASS (see above)")
print(f"  CHECK (b): Checkpoint resume → run restart to confirm skip logic")
print(f"  CHECK (c): Trigger S IID_std → "
      f"{'LIVE ' + str(round(amplitude_iid_std, 4)) if check_c_pass else 'FALLBACK ' + str(PHASE9B_AMPLITUDE_IID_STD)}")
print(f"  CHECK (d): G1/G2 Pima α=0.1:")
for s in SMOKE_SEEDS:
    f, rs = geo_results[s]
    tag = f"FLAGGED — {rs[0]}" if f else "clean (G1/G2 NONE)"
    print(f"    seed={s}: {tag}")

all_complete = (all(fed_a_done) and all(cent_a_done)
                and all(fed_b_done) and repl_done)
print(f"\n  Pipeline verdict: "
      f"{'COMPLETE — proceed to full Phase 16 sweep' if all_complete else 'INCOMPLETE — see above'}")
print()
