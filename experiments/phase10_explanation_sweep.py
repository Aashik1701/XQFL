"""Phase 10: Targeted explanation sweep for RQ1, RQ2, RQ3.

Sweep design
------------
Phase A (RQ1 + RQ2):
  4 encodings × IID × {1, 7, 42, 123, 999} — federated (15 rounds) + centralized
  baseline per seed. Analysis reports mean±std across 5 seeds; individual seed
  results shown for outlier detection. CENTRAL_EPOCHS=45 matches total gradient
  steps (15 rounds × 3 local epochs) but not data-diversity structure (MN-07 §3).

Phase B (RQ3):
  amplitude × {alpha=0.5, alpha=0.1, quantity-skew} × all 5 seeds — federated only.
  IID amplitude (Phase A, each seed) reused as JSD baseline per seed. Analysis reports
  mean±std JSD and Track A/B fraction across seeds; seed=42 per-round table kept for
  mechanistic detail. The α=0.1 degenerate-JSD mechanism is assessed for replication.

Outputs
-------
  results/phase10_log.csv        — all explanation rows (all runs, all rounds)
  results/phase10_checkpoint.csv — completed run keys for restart safety

CSV schema (phase10_log.csv):
  run_type, encoding, distribution, dirichlet_alpha, seed,
  round, client_id, explanation_vector, accuracy,
  track_class, margin, norm_phi, timestamp

Track A/B classification (per round, per row):
  A  = accuracy >  majority_baseline + COLLAPSE_MARGIN  (functional model)
  B  = accuracy <= majority_baseline + COLLAPSE_MARGIN  (collapsed model)
  centralized  = always labelled "centralized" (not subject to A/B logic)
  margin = accuracy − threshold  (logged for every row; negative → Track B)
"""

import os, sys, csv, json, math
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding, amplitude_encoding
from models.encodings import reuploading_encoding, iqp_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation
from explain.q_shap import q_shap_values, make_quantum_predict_fn
from explain.metrics import (
    kendalls_tau, cosine_similarity, l2_distance, js_divergence,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS         = 4
N_LAYERS         = 2
N_CLIENTS        = 4
N_ROUNDS         = 15
LOCAL_EPOCHS     = 3
CENTRAL_EPOCHS   = 45        # 15 rounds × 3 local epochs — gradient-step matched
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20
N_CLIENT_EXPLAIN = 5
N_CLIENT_BG      = 10
SEEDS        = [1, 7, 42, 123, 999]   # all project-standard seeds for RQ1/RQ2
SEED_RQ3     = 42                     # RQ3 Phase B single-seed (already checkpointed)
COLLAPSE_MARGIN  = 0.02      # threshold = majority_baseline + 2pp

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

DISTRIBUTIONS = {
    "iid":       {"split": "iid",       "alpha": None},
    "alpha_0.5": {"split": "dirichlet", "alpha": 0.5},
    "alpha_0.1": {"split": "dirichlet", "alpha": 0.1},
    "quantity":  {"split": "quantity",  "alpha": None},
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "phase10_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data + thresholds ─────────────────────────────────────────────────────────

print("Loading Wisconsin (breast cancer) dataset …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED_RQ3)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Majority-class baseline : {majority_baseline:.4f}")
print(f"  Collapse threshold (A/B): {collapse_threshold:.4f}  "
      f"(baseline + {COLLAPSE_MARGIN*100:.0f}pp)")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(run_type, encoding, distribution, seed):
    return f"{run_type}|{encoding}|{distribution}|{seed}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "done":
                done.add(row["run_key"])
    return done


def mark_done(rk):
    first = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "status", "completed_at"])
        if first:
            w.writeheader()
        w.writerow({
            "run_key":      rk,
            "status":       "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [checkpoint] done: {rk}")

# ── results CSV helpers ───────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(run_type, encoding, distribution, dirichlet_alpha, seed,
               round_num, client_id, explanation_vector, accuracy):
    margin     = float(accuracy) - collapse_threshold
    if run_type == "centralized":
        track = "centralized"
    else:
        track  = "A" if float(accuracy) > collapse_threshold else "B"
    norm_phi   = float(np.linalg.norm(explanation_vector))
    vec        = [float(v) for v in np.asarray(explanation_vector).flatten()]
    row = {
        "run_type":          run_type,
        "encoding":          encoding,
        "distribution":      distribution,
        "dirichlet_alpha":   "" if dirichlet_alpha is None else dirichlet_alpha,
        "seed":              seed,
        "round":             round_num,
        "client_id":         client_id,
        "explanation_vector": json.dumps(vec),
        "accuracy":          f"{float(accuracy):.6f}",
        "track_class":       track,
        "margin":            f"{margin:.6f}",
        "norm_phi":          f"{norm_phi:.6f}",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=RESULTS_COLS).writerow(row)

# ── Phase10Logger — duck-typed for run_federation's logger param ──────────────

class Phase10Logger:
    """Minimal interface compatible with LoggingFedAvg and run_federation.

    run_federation calls logger.log(round, client_id, vector, accuracy).
    No ExperimentConfig needed — we carry encoding/distribution directly.
    """
    def __init__(self, run_type, encoding, distribution, dirichlet_alpha, seed):
        self.run_type       = run_type
        self.encoding       = encoding
        self.distribution   = distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.seed           = seed

    def log(self, round_num, client_id, explanation_vector, accuracy):
        append_row(
            self.run_type, self.encoding, self.distribution, self.dirichlet_alpha,
            self.seed, round_num, client_id, explanation_vector, float(accuracy),
        )

# ── federated run ─────────────────────────────────────────────────────────────

def run_federated(encoding_name, distribution_name, seed):
    rk = make_run_key("federated", encoding_name, distribution_name, seed)
    if rk in load_checkpoint():
        print(f"  [skip] {rk}")
        return

    dist = DISTRIBUTIONS[distribution_name]
    print(f"\n{'='*70}")
    print(f"  FEDERATED  {encoding_name} / {distribution_name} / seed={seed}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase10Logger(
        "federated", encoding_name, distribution_name, dist["alpha"], seed
    )

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

    mark_done(rk)

# ── centralized baseline ──────────────────────────────────────────────────────

def run_centralized(encoding_name, seed):
    rk = make_run_key("centralized", encoding_name, "iid", seed)
    if rk in load_checkpoint():
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  CENTRALIZED  {encoding_name} / seed={seed}")
    print(f"  {CENTRAL_EPOCHS} epochs = {N_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs")
    print(f"  Caveat (MN-07 §3): controls total gradient steps, not data-diversity")
    print(f"  structure. Centralized sees full dataset each epoch; federated clients")
    print(f"  see per-shard data with periodic aggregation. The comparison isolates")
    print(f"  federation's effect on explanations as cleanly as is reasonably")
    print(f"  possible, but is not a perfect ceteris-paribus control.")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fit the same PCA + scaler as run_federation would use
    pca    = PCA(n_components=N_QUBITS, random_state=seed)
    X_pca  = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_pca)

    model         = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODINGS[encoding_name]
    )
    model.pca     = pca
    model._scaler = scaler
    model.fit(X_train, y_train, epochs=CENTRAL_EPOCHS, lr=0.05, batch_size=32)

    acc = float((model.predict(X_test) == y_test).mean())
    print(f"  Centralized test accuracy: {acc:.4f}")

    X_test_q  = model._preprocess(X_test[:N_GLOBAL_EXPLAIN])
    X_bg_q    = model._preprocess(X_train[:N_GLOBAL_BG])
    q_predict = make_quantum_predict_fn(model)
    qshap     = q_shap_values(
        q_predict, X_test_q, X_bg_q, n_permutations=N_PERM, seed=seed
    )
    global_qshap = qshap.mean(axis=0)

    append_row("centralized", encoding_name, "iid", None, seed,
               15, "global", global_qshap, acc)
    mark_done(rk)

# ── analysis ──────────────────────────────────────────────────────────────────

def load_results():
    """Load results CSV; skip rows from runs not yet marked done."""
    done = load_checkpoint()
    rows = []
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rk = make_run_key(
                row["run_type"], row["encoding"], row["distribution"],
                int(row["seed"])
            )
            if rk not in done:
                continue
            row["round"]            = int(row["round"])
            row["accuracy"]         = float(row["accuracy"])
            row["margin"]           = float(row["margin"])
            row["norm_phi"]         = float(row["norm_phi"])
            row["seed"]             = int(row["seed"])
            row["dirichlet_alpha"]  = (float(row["dirichlet_alpha"])
                                       if row["dirichlet_alpha"] else None)
            row["explanation_vector"] = json.loads(row["explanation_vector"])
            try:
                row["client_id"] = int(row["client_id"])
            except (ValueError, TypeError):
                pass
            rows.append(row)
    return rows


def print_rq1(rows, seeds):
    print("\n" + "="*70)
    print("RQ1  FEDERATED vs CENTRALIZED Q-SHAP  (Wisconsin / IID / 5 seeds)")
    print("  τ, cosine, L2 compare fed round-15 global vs centralized Q-SHAP vector")
    print("  Reported as mean ± std across seeds; per-seed detail below summary")
    print("  Caveat: 45 centralized epochs ≈ gradient-step matched; not ceteris-paribus")
    print("="*70)

    for enc in ENCODINGS:
        per_seed = {}
        for seed in seeds:
            fed_r15 = [r for r in rows
                       if r["run_type"] == "federated" and r["encoding"] == enc
                       and r["distribution"] == "iid" and r["round"] == 15
                       and r["client_id"] == "global" and r["seed"] == seed]
            cent    = [r for r in rows
                       if r["run_type"] == "centralized" and r["encoding"] == enc
                       and r["seed"] == seed]
            if not fed_r15 or not cent:
                continue
            f = fed_r15[-1]; c = cent[-1]
            per_seed[seed] = {
                "fed_acc":  f["accuracy"],
                "cent_acc": c["accuracy"],
                "delta":    f["accuracy"] - c["accuracy"],
                "tau":      kendalls_tau(f["explanation_vector"], c["explanation_vector"]),
                "cos":      cosine_similarity(f["explanation_vector"], c["explanation_vector"]),
                "l2":       l2_distance(f["explanation_vector"], c["explanation_vector"]),
            }

        n = len(per_seed)
        if n == 0:
            print(f"\n  {enc.upper()}: (no data)")
            continue

        vals = list(per_seed.values())
        fa  = [v["fed_acc"]  for v in vals]
        ca  = [v["cent_acc"] for v in vals]
        da  = [v["delta"]    for v in vals]
        taus = [v["tau"] for v in vals]
        coss = [v["cos"] for v in vals]
        l2s  = [v["l2"]  for v in vals]
        ddof = 1 if n > 1 else 0

        print(f"\n  {enc.upper()}  (n={n} seeds)")
        print(f"    FedAcc  {np.mean(fa)*100:.2f} ± {np.std(fa, ddof=ddof)*100:.2f}pp  "
              f"| CentAcc {np.mean(ca)*100:.2f} ± {np.std(ca, ddof=ddof)*100:.2f}pp  "
              f"| Δ {np.mean(da)*100:+.2f} ± {np.std(da, ddof=ddof)*100:.2f}pp")
        print(f"    τ       {np.mean(taus):.3f} ± {np.std(taus, ddof=ddof):.3f}  "
              f"| cosine {np.mean(coss):.4f} ± {np.std(coss, ddof=ddof):.4f}  "
              f"| L2 {np.mean(l2s):.4f} ± {np.std(l2s, ddof=ddof):.4f}")
        print(f"    Per-seed τ: "
              + "  ".join(f"s{s}={v['tau']:.3f}" for s, v in per_seed.items()))
        print(f"    Per-seed Δ: "
              + "  ".join(f"s{s}={v['delta']*100:+.1f}pp" for s, v in per_seed.items()))


def print_rq2(rows, seeds):
    print("\n" + "="*70)
    print("RQ2  EXPLANATION STABILITY PER ENCODING  (Wisconsin / IID / 5 seeds)")
    print("  Per-seed mean τ/cosine/L2 across 15 consecutive round pairs")
    print("  Reported as mean ± std across seeds; ranking with uncertainty")
    print("="*70)

    # Collect per-seed means for each encoding
    enc_summaries = {}
    for enc in ENCODINGS:
        seed_means = {}
        for seed in seeds:
            global_rows = sorted(
                [r for r in rows
                 if r["run_type"] == "federated" and r["encoding"] == enc
                 and r["distribution"] == "iid" and r["client_id"] == "global"
                 and r["seed"] == seed],
                key=lambda x: x["round"],
            )
            if len(global_rows) < 2:
                continue
            taus, coss, l2s = [], [], []
            for i in range(1, len(global_rows)):
                a = global_rows[i - 1]; b = global_rows[i]
                taus.append(kendalls_tau(a["explanation_vector"], b["explanation_vector"]))
                coss.append(cosine_similarity(a["explanation_vector"], b["explanation_vector"]))
                l2s.append(l2_distance(a["explanation_vector"], b["explanation_vector"]))
            seed_means[seed] = {
                "tau": np.mean(taus),
                "cos": np.mean(coss),
                "l2":  np.mean(l2s),
                "final_phi":  global_rows[-1]["norm_phi"],
                "r15_acc":    global_rows[-1]["accuracy"],
            }
        enc_summaries[enc] = seed_means

    # Summary table across encodings
    print(f"\n  {'Encoding':<13} {'n':>2}  {'Mean τ':>10}  {'Mean cos':>12}  "
          f"{'Mean L2':>10}  {'Stable?'}")
    print(f"  {'-'*13} {'-'*2}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*20}")

    enc_tau_means = {}
    for enc in ENCODINGS:
        sm = enc_summaries[enc]
        n = len(sm)
        if n == 0:
            print(f"  {enc:<13}  (no data)")
            continue
        ddof = 1 if n > 1 else 0
        mt    = [v["tau"] for v in sm.values()]
        mc    = [v["cos"] for v in sm.values()]
        ml    = [v["l2"]  for v in sm.values()]
        mean_t = np.mean(mt); std_t = np.std(mt, ddof=ddof)
        mean_c = np.mean(mc); std_c = np.std(mc, ddof=ddof)
        mean_l = np.mean(ml); std_l = np.std(ml, ddof=ddof)
        enc_tau_means[enc] = (mean_t, std_t)

        # Flag if std > half the gap to next encoding (placeholder; compared after)
        print(f"  {enc:<13} {n:>2}  "
              f"{mean_t:.3f}±{std_t:.3f}  "
              f"{mean_c:.4f}±{std_c:.4f}  "
              f"{mean_l:.4f}±{std_l:.4f}  "
              f"(per-seed: " + " ".join(f"{v['tau']:.3f}" for v in sm.values()) + ")")

    # Explicit amplitude vs angle comparison
    if "amplitude" in enc_tau_means and "angle" in enc_tau_means:
        amp_t, amp_std = enc_tau_means["amplitude"]
        ang_t, ang_std = enc_tau_means["angle"]
        gap = amp_t - ang_t
        pooled_std = (amp_std**2 + ang_std**2) ** 0.5
        print(f"\n  Amplitude vs angle τ gap: {gap:+.3f}  "
              f"(pooled std: {pooled_std:.3f}  "
              f"→ {'reliable' if abs(gap) > pooled_std else 'within noise'}"
              f" — gap {'>' if abs(gap) > pooled_std else '≤'} pooled std)")

    # Per-encoding detailed per-seed breakdown
    print(f"\n  Per-seed detail:")
    for enc in ENCODINGS:
        sm = enc_summaries[enc]
        if not sm:
            continue
        print(f"  {enc.upper()}")
        for seed, v in sm.items():
            print(f"    seed={seed:>3}: τ={v['tau']:.3f}  cos={v['cos']:.4f}  "
                  f"L2={v['l2']:.4f}  ||φ||={v['final_phi']:.4f}  "
                  f"R15={v['r15_acc']:.4f}")


def _rq3_seed_stats(rows, dist_name, seed):
    """Return (mean_jsd, n_track_a, n_track_b_post_r0) for one seed×distribution."""
    fed_rows = [r for r in rows
                if r["run_type"] == "federated" and r["encoding"] == "amplitude"
                and r["distribution"] == dist_name and r["seed"] == seed]
    if not fed_rows:
        return float("nan"), 0, 0

    global_rows = sorted(
        [r for r in fed_rows if r["client_id"] == "global"],
        key=lambda x: x["round"],
    )
    client_by_round = {}
    for r in fed_rows:
        if r["client_id"] != "global":
            client_by_round.setdefault(r["round"], []).append(r)

    track_a_jsds = []
    n_b_post = 0
    for gr in global_rows:
        if gr["round"] == 0:
            continue  # skip initial-params round
        trk = gr["track_class"]
        if trk == "A" and gr["round"] in client_by_round:
            jsds = [
                js_divergence(cr["explanation_vector"], gr["explanation_vector"])
                for cr in client_by_round[gr["round"]]
            ]
            jsds = [v for v in jsds if not math.isnan(v)]
            if jsds:
                track_a_jsds.append(float(np.mean(jsds)))
        elif trk == "B":
            n_b_post += 1

    mean_jsd = float(np.mean(track_a_jsds)) if track_a_jsds else float("nan")
    n_a = len(track_a_jsds)
    return mean_jsd, n_a, n_b_post


def print_rq3(rows, seeds=None, detail_seed=SEED_RQ3):
    if seeds is None:
        seeds = [SEED_RQ3]

    print("\n" + "="*70)
    print("RQ3  LOCAL-GLOBAL EXPLANATION DIVERGENCE vs NON-IID LEVEL")
    print(f"  Encoding: amplitude  |  seeds={seeds}")
    print(f"  Collapse threshold = {collapse_threshold:.4f}  "
          f"(majority {majority_baseline:.4f} + {COLLAPSE_MARGIN*100:.0f}pp)")
    print("  Track A: functional  |  Track B: collapsed  |  rounds 1-15 only")
    print("  JSD = JS-divergence between mean local and global Q-SHAP (Track A rounds)")
    print("="*70)

    # ── multi-seed summary ──────────────────────────────────────────────────
    print(f"\n  {'Distribution':>14}  {'Seeds w/ data':>13}  "
          f"{'Mean JSD±std':>14}  {'TrackA rounds/seed':>19}  "
          f"{'TrackB≥1 seeds':>14}")
    for dist_name in DISTRIBUTIONS:
        jsds, n_as, n_bs = [], [], []
        for seed in seeds:
            mjsd, na, nb = _rq3_seed_stats(rows, dist_name, seed)
            if not math.isnan(mjsd):
                jsds.append(mjsd)
                n_as.append(na)
                n_bs.append(nb)
        if not jsds:
            print(f"  {dist_name:>14}  (no data)")
            continue
        mean_j = float(np.mean(jsds))
        std_j  = float(np.std(jsds))
        n_b_seeds = sum(1 for nb in n_bs if nb > 0)
        print(f"  {dist_name:>14}  {len(jsds):>13}  "
              f"{mean_j:.4f}±{std_j:.4f}  "
              f"{np.mean(n_as):.1f}±{np.std(n_as):.1f}/15        "
              f"{n_b_seeds:>3}/{len(jsds)}")

    # ── per-seed detail ─────────────────────────────────────────────────────
    print(f"\n  Per-seed JSD (Track A mean) across distributions:")
    print(f"  {'Seed':>6}  {'iid':>8}  {'α=0.5':>8}  {'α=0.1':>8}  {'qty':>8}  "
          f"{'α=0.1 TrackB':>13}")
    for seed in seeds:
        vals = {}
        for dist_name in DISTRIBUTIONS:
            mjsd, na, nb = _rq3_seed_stats(rows, dist_name, seed)
            vals[dist_name] = (mjsd, nb)
        iid_s = f"{vals['iid'][0]:.4f}"   if not math.isnan(vals['iid'][0])       else "n/a"
        a05_s = f"{vals['alpha_0.5'][0]:.4f}" if not math.isnan(vals['alpha_0.5'][0]) else "n/a"
        a01_s = f"{vals['alpha_0.1'][0]:.4f}" if not math.isnan(vals['alpha_0.1'][0]) else "n/a"
        qty_s = f"{vals['quantity'][0]:.4f}"   if not math.isnan(vals['quantity'][0])  else "n/a"
        nb01  = vals['alpha_0.1'][1]
        print(f"  {seed:>6}  {iid_s:>8}  {a05_s:>8}  {a01_s:>8}  {qty_s:>8}  "
              f"  {nb01:>2}/15 B rounds")

    # ── seed=detail_seed per-round breakdown (mechanistic analysis) ─────────
    print(f"\n  --- Per-round detail (seed={detail_seed}, mechanistic analysis) ---")
    for dist_name in DISTRIBUTIONS:
        fed_rows = [r for r in rows
                    if r["run_type"] == "federated" and r["encoding"] == "amplitude"
                    and r["distribution"] == dist_name and r["seed"] == detail_seed]
        if not fed_rows:
            print(f"\n  {dist_name}: (no data for seed={detail_seed})")
            continue

        global_rows = sorted(
            [r for r in fed_rows if r["client_id"] == "global"],
            key=lambda x: x["round"],
        )
        client_by_round = {}
        for r in fed_rows:
            if r["client_id"] != "global":
                client_by_round.setdefault(r["round"], []).append(r)

        print(f"\n  DISTRIBUTION: {dist_name}  (seed={detail_seed})")
        print(f"    {'Rnd':>4} {'Acc':>7} {'Trk':>4} {'Margin':>8} "
              f"{'||φ_g||':>9} {'JSD(loc,glb)':>14}")

        track_a_jsds  = []
        track_b_norms = []

        for gr in global_rows:
            rnd = gr["round"]
            trk = gr["track_class"]
            mar = gr["margin"]

            if trk == "A" and rnd in client_by_round:
                jsds = [
                    js_divergence(cr["explanation_vector"], gr["explanation_vector"])
                    for cr in client_by_round[rnd]
                ]
                jsds = [v for v in jsds if not math.isnan(v)]
                mjsd = np.mean(jsds) if jsds else float("nan")
                if not math.isnan(mjsd) and rnd > 0:
                    track_a_jsds.append(mjsd)
                jsd_str = f"{mjsd:.4f}" if not math.isnan(mjsd) else "nan"
            elif trk == "B":
                if rnd > 0:
                    track_b_norms.append(gr["norm_phi"])
                jsd_str = "n/a (B)"
            else:
                jsd_str = "n/a"

            print(f"    {rnd:>4} {gr['accuracy']:>7.4f} {trk:>4} {mar:>+8.4f} "
                  f"{gr['norm_phi']:>9.5f} {jsd_str:>14}")

        n_a = sum(1 for g in global_rows if g["track_class"] == "A" and g["round"] > 0)
        n_b = sum(1 for g in global_rows if g["track_class"] == "B" and g["round"] > 0)
        print(f"\n    Track A: {n_a}/15 rounds  |  Track B: {n_b}/15 rounds")
        if track_a_jsds:
            print(f"    Mean JSD (Track A, rds 1-15): {np.mean(track_a_jsds):.4f}  "
                  f"(n={len(track_a_jsds)} rounds)")
        if track_b_norms:
            print(f"    Track B ||φ_global|| range: "
                  f"{min(track_b_norms):.5f} – {max(track_b_norms):.5f}")
            print(f"    Interpretation: near-zero norms confirm explanation degeneration"
                  f" under collapse.")

# ── main ──────────────────────────────────────────────────────────────────────

init_results_csv()
_phase_a_total = len(SEEDS) * len(ENCODINGS) * 2  # fed + cent per seed per encoding
_phase_b_total = len(SEEDS) * 3                     # amplitude × 3 non-IID dists × all seeds
print(f"\n{'#'*70}")
print(f"  Phase 10 Explanation Sweep  |  Wisconsin / FedAvg")
print(f"  Phase A seeds: {SEEDS}  (RQ1 + RQ2)")
print(f"  Phase B seeds: {SEEDS}  (RQ3, amplitude × non-IID)")
print(f"  Results → {RESULTS_CSV}")
print(f"  Checkpoint → {CHECKPOINT_CSV}")
already_done = load_checkpoint()
print(f"  Already completed: {len(already_done)} run(s) of "
      f"{_phase_a_total + _phase_b_total} total "
      f"(A: {len(SEEDS)}×{len(ENCODINGS)} fed + {len(SEEDS)}×{len(ENCODINGS)} cent, "
      f"B: {_phase_b_total} fed)")
print(f"{'#'*70}")

# Phase A: RQ1 + RQ2  (5 seeds × 4 encodings × federated+centralized = 40 runs)
print(f"\n[PHASE A]  {len(SEEDS)} seeds × {len(ENCODINGS)} encodings × IID — fed + centralized")
for seed in SEEDS:
    print(f"\n  --- seed={seed} ---")
    for enc in ENCODINGS:
        run_federated(enc, "iid", seed)
        run_centralized(enc, seed)

# Phase B: RQ3  (amplitude × 3 non-IID dists × all seeds = 15 runs)
print(f"\n[PHASE B]  amplitude × non-IID distributions — federated ({len(SEEDS)} seeds)")
for seed in SEEDS:
    print(f"\n  --- seed={seed} ---")
    for dist in ["alpha_0.5", "alpha_0.1", "quantity"]:
        run_federated("amplitude", dist, seed)

# Analysis
rows = load_results()
print(f"\nTotal rows loaded (from completed runs): {len(rows)}")

print_rq1(rows, SEEDS)
print_rq2(rows, SEEDS)
print_rq3(rows, seeds=SEEDS, detail_seed=SEED_RQ3)

print(f"\n{'='*70}")
print("[Phase 10 complete]")
print(f"  Results : {RESULTS_CSV}")
print(f"  Checkpoint: {CHECKPOINT_CSV}")
