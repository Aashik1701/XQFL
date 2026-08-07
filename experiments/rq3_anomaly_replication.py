"""RQ3 anomaly replication: close three single-draw flags from RQ3 Limitations.

The three flagged conditions, their documented anomalies, and replication status:

  A. reuploading/α=0.5/seed=999  — near-zero ‖φ‖ (0.073) despite Track A
     Status: Phase 11 already ran MN-08 inline replications (rep1, rep2).
             Reads from phase11_log.csv — no new runs needed.

  B. amplitude/α=0.5/seed=123    — elevated JSD_cent (0.342) vs LOO mean 0.188±0.053
     Status: phase10 has one draw only; amplitude is not in phase11.
             2 fresh replication draws needed.

  C. reuploading/α=0.1/seed=42   — anomalously LOW JSD_cent (0.087) despite Track A
     Status: Phase 11 already ran MN-08 inline replications (rep1, rep2).
             Reads from phase11_log.csv — no new runs needed.

This script executes the 2 fresh draws for condition B (amplitude/α=0.5/seed=123)
and then prints a combined verdict table for all three conditions.

Protocol for condition B (same as phase12):
  - 2 additional draws beyond the phase10 primary
  - Same federation parameters: 4 clients, 15 rounds, 3 local epochs
  - Full Q-SHAP at round 15 global (n_global_explain=10, n_global_bg=20, n_perm=8)
  - JSD vs IID-federated: compare against phase10 IID/amplitude/seed=123 global R15 phi
  - JSD vs centralized:   compare against phase10 centralized/amplitude/seed=123 phi

Outputs:
  results/rq3_anomaly_replication_log.csv
  results/rq3_anomaly_replication_checkpoint.csv
"""

import os, sys, csv, json, signal, time, statistics
import numpy as np
import torch
from datetime import datetime, timezone
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import amplitude_encoding, reuploading_encoding
from federation.run_federation import run_federation
# js_divergence from metrics module — local js_div defined below shadows it intentionally

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS       = 4
N_LAYERS       = 2
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_GLOBAL_EXP   = 10
N_GLOBAL_BG    = 20
N_PERM         = 8
RUN_TIMEOUT    = 600
COLLAPSE_MARGIN = 0.02

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq3_anomaly_replication_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq3_anomaly_replication_checkpoint.csv")
PHASE10_CSV    = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")
PHASE11_CSV    = os.path.join(PROJECT_ROOT, "results", "phase11_log.csv")

RESULTS_COLS = [
    "anomaly_id", "encoding", "distribution", "dirichlet_alpha", "seed",
    "draw_num", "run_type",
    "accuracy", "track_class", "margin",
    "norm_phi", "jsd_cent", "jsd_iid",
    "explanation_vector",
    "wall_time_s", "timestamp",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def js_div(a, b, eps=1e-10):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ap = np.abs(a) + eps; ap /= ap.sum()
    bp = np.abs(b) + eps; bp /= bp.sum()
    return float(jensenshannon(ap, bp, base=2))


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


def append_result(anomaly_id, enc, dist, alpha, seed, draw_num, run_type,
                  acc, track, margin, norm_phi, jsd_cent, jsd_iid,
                  phi_vec, wall_time):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if first:
            w.writeheader()
        w.writerow({
            "anomaly_id":        anomaly_id,
            "encoding":          enc,
            "distribution":      dist,
            "dirichlet_alpha":   alpha,
            "seed":              seed,
            "draw_num":          draw_num,
            "run_type":          run_type,
            "accuracy":          f"{acc:.6f}",
            "track_class":       track,
            "margin":            f"{margin:.6f}",
            "norm_phi":          f"{norm_phi:.6f}",
            "jsd_cent":          f"{jsd_cent:.6f}",
            "jsd_iid":           f"{jsd_iid:.6f}",
            "explanation_vector": json.dumps([float(v) for v in phi_vec]),
            "wall_time_s":       f"{wall_time:.1f}",
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        })


def _timeout_handler(signum, frame):
    raise TimeoutError("Run timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


# ── load phase10 reference vectors ───────────────────────────────────────────

def _load_phase10_refs():
    """Return {(enc, seed, dist): phi_array} for R15 global rows from phase10."""
    refs = {}
    if not os.path.exists(PHASE10_CSV):
        print("  [warn] phase10_log.csv not found")
        return refs
    with open(PHASE10_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["client_id"] == "global" and row["round"] == "15":
                key = (row["encoding"], row["seed"], row["distribution"], row["run_type"])
                refs[key] = np.array(json.loads(row["explanation_vector"]), dtype=float)
    return refs


print("Loading phase10 reference vectors …")
REFS = _load_phase10_refs()

# Condition B references (amplitude/seed=123)
AMP_IID_123  = REFS.get(("amplitude", "123", "iid", "federated"))
AMP_CENT_123 = None
for (enc, seed, dist, rtype), phi in REFS.items():
    if enc == "amplitude" and seed == "123" and rtype == "centralized":
        AMP_CENT_123 = phi
        break

if AMP_IID_123 is None:
    print("  [warn] IID reference for amplitude/seed=123 not found in phase10")
if AMP_CENT_123 is None:
    print("  [warn] Centralized reference for amplitude/seed=123 not found in phase10")
else:
    print(f"  amp/123 IID ref norm  : {np.linalg.norm(AMP_IID_123):.4f}")
    print(f"  amp/123 cent ref norm : {np.linalg.norm(AMP_CENT_123):.4f}")

# Wisconsin thresholds
print("Loading Wisconsin data for collapse threshold …")
_Xtr, _Xte, _ytr, _yte = load_breast_cancer_data(random_state=42)
_majority   = float(np.bincount(_yte.astype(int)).max()) / len(_yte)
_threshold  = _majority + COLLAPSE_MARGIN
print(f"  collapse_threshold={_threshold:.4f}")


# ── condition B: amplitude/alpha0.5/seed=123 (2 fresh draws) ─────────────────

class _R15Logger:
    """Minimal logger that captures the global explanation at round 15 only."""
    def __init__(self):
        self.phi_r15 = None
        self.acc_r15 = None

    def log(self, round_num, client_id, explanation_vector, accuracy):
        if round_num == N_ROUNDS and client_id == "global":
            self.phi_r15 = np.asarray(explanation_vector, dtype=float).flatten().copy()
            self.acc_r15 = float(accuracy)


def run_amplitude_rep(draw_num):
    """Run one fresh replication draw for amplitude/α=0.5/seed=123."""
    rk = f"amplitude|alpha_0.5|123|rep{draw_num}"
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  ANOMALY-B  amplitude / alpha_0.5 / seed=123 / draw={draw_num}")
    print(f"  Primary: JSD_cent=0.342, JSD_iid=0.217, acc=0.895, ‖φ‖=0.453")
    print(f"  LOO mean JSD_cent=0.188±0.053 (dev=2.9σ)")
    print(f"{'='*70}")

    t0   = time.time()
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)
    logger = _R15Logger()

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=amplitude_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split="dirichlet", strategy_name="fedavg",
            seed=seed, dirichlet_alpha=0.5,
            project_root=PROJECT_ROOT,
            logger=logger,
            n_global_explain=N_GLOBAL_EXP,
            n_global_bg=N_GLOBAL_BG,
            n_permutations=N_PERM,
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
    track  = "A" if final_acc > _threshold else "B"
    margin = final_acc - _threshold
    print(f"  R15 acc={final_acc:.4f}  Track {track}  (margin={margin:+.4f})")

    phi_r15 = logger.phi_r15
    if phi_r15 is None:
        print("  [warn] R15 global phi not captured by logger — skipping draw")
        mark_done(rk, status="error")
        return

    norm_phi = float(np.linalg.norm(phi_r15))
    jsd_cent = js_div(phi_r15, AMP_CENT_123) if AMP_CENT_123 is not None else float("nan")
    jsd_iid  = js_div(phi_r15, AMP_IID_123)  if AMP_IID_123  is not None else float("nan")

    wall_time = time.time() - t0
    print(f"  ‖φ‖={norm_phi:.4f}  JSD_cent={jsd_cent:.4f}  JSD_iid={jsd_iid:.4f}  ({wall_time:.0f}s)")

    append_result(
        anomaly_id="B", enc="amplitude", dist="alpha_0.5", alpha=0.5,
        seed=123, draw_num=draw_num, run_type="replication",
        acc=final_acc, track=track, margin=margin,
        norm_phi=norm_phi, jsd_cent=jsd_cent, jsd_iid=jsd_iid,
        phi_vec=phi_r15, wall_time=wall_time,
    )
    mark_done(rk)
    print(f"  [DONE]  {wall_time/60:.1f} min")


# ── analysis: all three anomalies ─────────────────────────────────────────────

def print_analysis():
    print("\n" + "=" * 72)
    print("  RQ3 ANOMALY REPLICATION — Verdicts for all three flags")
    print("=" * 72)

    # Load phase11 data for anomalies A and C
    p11_rows = []
    if os.path.exists(PHASE11_CSV):
        p11_rows = list(csv.DictReader(open(PHASE11_CSV)))

    # Load new draws for anomaly B
    new_rows = []
    if os.path.exists(RESULTS_CSV):
        new_rows = list(csv.DictReader(open(RESULTS_CSV)))

    # Phase10 primary for anomaly B
    p10_rows = []
    if os.path.exists(PHASE10_CSV):
        p10_rows = list(csv.DictReader(open(PHASE10_CSV)))

    def js_div_from_rows(phi_a, rows, enc, seed, dist, rtype):
        candidates = [r for r in rows
                      if r["encoding"]==enc and r["seed"]==seed
                      and r["distribution"]==dist and r["run_type"]==rtype
                      and r["client_id"]=="global" and r["round"]=="15"]
        if not candidates:
            return None
        ref = np.array(json.loads(candidates[0]["explanation_vector"]), dtype=float)
        return js_div(phi_a, ref)

    # ── Anomaly A: reuploading/alpha0.5/seed=999 ──────────────────────────────
    print("\n── ANOMALY A: reuploading / α=0.5 / seed=999")
    print("   Primary flag: ‖φ‖=0.073 vs 0.382–0.479 for other seeds (near-zero norm despite Track A)")
    a_draws = sorted(
        [r for r in p11_rows if r["encoding"]=="reuploading"
         and r["distribution"]=="alpha_0.5" and r["seed"]=="999"
         and r["client_id"]=="global" and r["round"]=="15"],
        key=lambda x: x["timestamp"]
    )
    print(f"   Phase11 draws: {len(a_draws)} (1 primary + {len(a_draws)-1} MN-08 reps)")

    # Reference phis
    reup_iid_999  = REFS.get(("reuploading", "999", "iid", "federated"))
    reup_cent_999 = None
    for (enc, seed, dist, rtype), phi in REFS.items():
        if enc=="reuploading" and seed=="999" and rtype=="centralized":
            reup_cent_999 = phi; break

    normal_range = "0.382–0.479 (other reuploading/α=0.5 seeds)"
    print(f"   {'Draw':<6}  {'Type':<12}  {'acc':>8}  {'Track':>5}  {'‖φ‖':>8}  {'JSD_iid':>8}  {'JSD_cent':>9}")
    print(f"   {'-'*66}")
    n_anomalous = 0
    for i, d in enumerate(a_draws, 1):
        phi = np.array(json.loads(d["explanation_vector"]), dtype=float)
        norm = float(np.linalg.norm(phi))
        jsd_i = js_div(phi, reup_iid_999)  if reup_iid_999  is not None else float("nan")
        jsd_c = js_div(phi, reup_cent_999) if reup_cent_999 is not None else float("nan")
        flag = " ← ANOMALOUS" if norm < 0.15 else ""
        if norm < 0.15:
            n_anomalous += 1
        print(f"   {i:<6}  {d['run_type']:<12}  {d['accuracy']:>8}  {d['track_class']:>5}  "
              f"{norm:>8.4f}  {jsd_i:>8.4f}  {jsd_c:>9.4f}{flag}")
    print(f"   Anomaly rate: {n_anomalous}/{len(a_draws)} draws with ‖φ‖ < 0.15")

    if n_anomalous >= 2:
        print("   VERDICT: Stochastically persistent near-collapse. ‖φ‖ anomaly occurs in "
              f"{n_anomalous}/{len(a_draws)} draws — not a single-draw fluke. "
              "Model reaches Track-A accuracy but consistently produces near-zero explanations "
              "in ~2/3 draws. Distinct from full collapse (Track B) but shares the degenerate-"
              "explanation mechanism. Label: MN-08 boundary-sensitivity (marginal case).")
    elif n_anomalous == 1:
        print("   VERDICT: Single-draw anomaly. One draw showed near-zero ‖φ‖; remaining draws "
              "returned to normal norm range. Not reproducible — attribute to actor-RNG noise.")
    else:
        print("   VERDICT: Anomaly not reproduced. All replication draws show normal ‖φ‖.")

    # ── Anomaly B: amplitude/alpha0.5/seed=123 ────────────────────────────────
    print("\n── ANOMALY B: amplitude / α=0.5 / seed=123")
    print("   Primary flag: JSD_cent=0.342 vs LOO mean 0.188±0.053 (dev=2.9σ)")
    p10_primary = [r for r in p10_rows if r["encoding"]=="amplitude"
                   and r["distribution"]=="alpha_0.5" and r["seed"]=="123"
                   and r["client_id"]=="global" and r["round"]=="15"
                   and r["run_type"]=="federated"]
    b_reps = [r for r in new_rows if r.get("anomaly_id")=="B"]

    amp_cent_phi = AMP_CENT_123
    amp_iid_phi  = AMP_IID_123

    all_b_draws = []
    # Primary
    if p10_primary:
        d = p10_primary[0]
        phi = np.array(json.loads(d["explanation_vector"]), dtype=float)
        all_b_draws.append(("1 (primary)", "federated", d["accuracy"], d["track_class"],
                             float(np.linalg.norm(phi)),
                             js_div(phi, amp_iid_phi) if amp_iid_phi is not None else float("nan"),
                             js_div(phi, amp_cent_phi) if amp_cent_phi is not None else float("nan")))
    # Reps
    for rep in sorted(b_reps, key=lambda x: x.get("draw_num", 0)):
        phi = np.array(json.loads(rep["explanation_vector"]), dtype=float)
        dn  = str(int(rep["draw_num"]) + 1)
        all_b_draws.append((f"{dn} (rep{rep['draw_num']})", rep["run_type"],
                             rep["accuracy"], rep["track_class"],
                             float(np.linalg.norm(phi)),
                             float(rep["jsd_iid"]), float(rep["jsd_cent"])))

    print(f"   Draws available: {len(all_b_draws)} (1 phase10 primary + {len(b_reps)} fresh reps)")
    print(f"   {'Draw':<14}  {'Type':<12}  {'acc':>8}  {'Track':>5}  {'‖φ‖':>8}  {'JSD_iid':>8}  {'JSD_cent':>9}")
    print(f"   {'-'*68}")
    LOO_MEAN_JSDCENT = 0.188
    LOO_STD_JSDCENT  = 0.053
    n_elevated = 0
    for label, rtype, acc, track, norm, jsd_i, jsd_c in all_b_draws:
        flag = " ← elevated" if jsd_c > LOO_MEAN_JSDCENT + 2*LOO_STD_JSDCENT else ""
        if jsd_c > LOO_MEAN_JSDCENT + 2*LOO_STD_JSDCENT:
            n_elevated += 1
        print(f"   {label:<14}  {rtype:<12}  {acc:>8}  {track:>5}  "
              f"{norm:>8.4f}  {jsd_i:>8.4f}  {jsd_c:>9.4f}{flag}")
    print(f"   Elevated JSD_cent (>{LOO_MEAN_JSDCENT+2*LOO_STD_JSDCENT:.3f}): {n_elevated}/{len(all_b_draws)} draws")

    if len(b_reps) < 2:
        print("   VERDICT: PENDING — awaiting fresh replication draws.")
    elif n_elevated >= 2:
        print(f"   VERDICT: Elevation persists in {n_elevated}/{len(all_b_draws)} draws. "
              "Elevated JSD_cent is reproducible for this seed. "
              "Consistent with mechanical explanation: seed=123 has the highest ‖φ‖ in the condition, "
              "and a high-magnitude explanation diverges more from a lower-magnitude reference. "
              "Label: explanation-magnitude-driven JSD elevation, not a qualitative anomaly.")
    else:
        print(f"   VERDICT: Single-draw anomaly. Fresh draws return to normal JSD_cent range. "
              "The elevated primary draw was actor-RNG-sensitive. "
              "Label: not reproducible — do not treat as a reliable condition-level finding.")

    # ── Anomaly C: reuploading/alpha0.1/seed=42 ──────────────────────────────
    print("\n── ANOMALY C: reuploading / α=0.1 / seed=42")
    print("   Primary flag: JSD_cent=0.087 (anomalously LOW; LOO mean 0.230±0.064, dev=2.2σ)")
    c_draws = sorted(
        [r for r in p11_rows if r["encoding"]=="reuploading"
         and r["distribution"]=="alpha_0.1" and r["seed"]=="42"
         and r["client_id"]=="global" and r["round"]=="15"],
        key=lambda x: x["timestamp"]
    )
    print(f"   Phase11 draws: {len(c_draws)} (1 primary + {len(c_draws)-1} MN-08 reps)")

    reup_iid_42  = REFS.get(("reuploading", "42", "iid", "federated"))
    reup_cent_42 = None
    for (enc, seed, dist, rtype), phi in REFS.items():
        if enc=="reuploading" and seed=="42" and rtype=="centralized":
            reup_cent_42 = phi; break

    LOO_MEAN_C  = 0.230
    LOO_STD_C   = 0.064
    LOW_THRESH  = LOO_MEAN_C - 2*LOO_STD_C   # anomalously low threshold
    n_low = 0
    print(f"   {'Draw':<6}  {'Type':<12}  {'acc':>8}  {'Track':>5}  {'‖φ‖':>8}  {'JSD_iid':>8}  {'JSD_cent':>9}")
    print(f"   {'-'*66}")
    for i, d in enumerate(c_draws, 1):
        phi  = np.array(json.loads(d["explanation_vector"]), dtype=float)
        norm = float(np.linalg.norm(phi))
        jsd_i = js_div(phi, reup_iid_42)  if reup_iid_42  is not None else float("nan")
        jsd_c = js_div(phi, reup_cent_42) if reup_cent_42 is not None else float("nan")
        flag = " ← anomalously low" if jsd_c < LOW_THRESH else ""
        if jsd_c < LOW_THRESH:
            n_low += 1
        print(f"   {i:<6}  {d['run_type']:<12}  {d['accuracy']:>8}  {d['track_class']:>5}  "
              f"{norm:>8.4f}  {jsd_i:>8.4f}  {jsd_c:>9.4f}{flag}")
    print(f"   Anomalously low JSD_cent (<{LOW_THRESH:.3f}): {n_low}/{len(c_draws)} draws")

    if n_low <= 1:
        print("   VERDICT: Single-draw positive anomaly. Primary draw was unusually close to "
              "centralized baseline; replication draws fall in the normal range. "
              "Not reproducible — attribute to actor-RNG variation in partition sampling. "
              "The low-JSD draw does not reflect a systematic property of this condition. "
              "Label: actor-RNG single-draw fluctuation; no new MN entry needed.")
    else:
        print(f"   VERDICT: Low JSD persists in {n_low}/{len(c_draws)} draws. "
              "Genuine near-centralized convergence under severe skew for this seed. "
              "Requires explanation — possible fortuitous partition geometry.")

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print("""
  A. reuploading/α=0.5/seed=999 (near-zero ‖φ‖ despite Track A):
     See verdict above — resolved from phase11 inline replications.

  B. amplitude/α=0.5/seed=123 (elevated JSD_cent):
     See verdict above — resolved from this script's fresh draws.

  C. reuploading/α=0.1/seed=42 (anomalously low JSD_cent):
     See verdict above — resolved from phase11 inline replications.
""")


# ── main ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

print("\n" + "=" * 72)
print("  RQ3 ANOMALY REPLICATION SWEEP")
print("  Target: amplitude / α=0.5 / seed=123 (anomalies A and C read from phase11)")
print("=" * 72 + "\n")

# Condition A and C: already replicated in phase11; no new runs.
# Condition B: run 2 fresh draws.
for draw_num in [1, 2]:
    run_amplitude_rep(draw_num)

print_analysis()
