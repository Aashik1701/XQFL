"""Phase 14: IQP/seed=7 cross-condition investigation.

Background (methods_notes.md, Replication Discipline + Phase 11 RQ3 analysis):

Two questions:

  Q1 (resolved by existing data — verify before running):
     IQP/alpha_0.5/seed=7 shows JSD_iid=0.799 in Phase 11 primary run.
     Phase 11 has 2 replication rows for this cell (federated+2 reps).
     → Check whether 3 existing draws already constitute a replication
       (actor-RNG-sensitive drift, consistent with MN-08).

  Q2 (the priority investigation):
     IQP/quantity/seed=7 is the only Phase 11 draw for this cell.
     JSD_cent=0.517, JSD_iid=0.589, norm=0.123 — anomalous vs other
     IQP/quantity seeds (norm 0.317–0.395, JSD_cent 0.11–0.20).
     This is Track A (not collapse); MN-08 mechanism (adversarial partition
     geometry, G1/G2 flags) does NOT apply to quantity-skew.
     Two independent signals for IQP/seed=7 under quantity-skew:
       (a) accuracy-adjacent anomaly (previously checked, G1/G2 absent)
       (b) now JSD-based anomaly in Phase 11 data (this is (b))
     User hypothesis: IQP+seed=7 initialization interacting with IQP's
     entangling structure specifically, independent of any partition effect.
     Check: does IQP/seed=7 show anomalies under plain IID as well?
     If yes (similar rate under IID) → initialization-specific, not partition.
     If no (anomaly only under quantity-skew) → partition amplifies something.

Protocol:
  - 3 fresh quantity-skew runs for IQP/seed=7
  - 3 fresh IID runs for IQP/seed=7 (comparison)
  - For each: round-15 global explanation via Phase11Logger protocol
  - JSD_cent: vs Phase 10 centralized IQP/seed=7 (deterministic)
  - Data loading: random_state=42 (matches Phase 10/11 train/test split)
  - Parameters: n_global_explain=10, n_global_bg=20, n_permutations=8
  - Partition geometry of quantity/seed=7 verified at startup (G1/G2 expected absent)

Run from project root:
    python experiments/phase14_iqp_seed7_investigation.py
"""

import os, sys, csv, json, signal, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.loaders import load_breast_cancer_data
from data.partition import iid_split, quantity_skew_split
from models.encodings import iqp_encoding
from federation.run_federation import run_federation

# ── constants (matching Phase 11 protocol) ────────────────────────────────────
SEED             = 7
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
N_REPS           = 3
RUN_TIMEOUT      = 600
COLLAPSE_MARGIN  = 0.02

# Anomaly thresholds relative to normal IQP IID seeds (norm 0.32–0.41, JSD_cent 0.11–0.22)
NORM_ANOMALY_THRESH    = 0.15   # norm_phi below this = anomalous
JSD_CENT_ANOMALY_THRESH = 0.40  # JSD_cent above this = anomalous

# ── data loading ──────────────────────────────────────────────────────────────
print("Loading Wisconsin (random_state=42, matching Phase 10/11 protocol) …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Collapse threshold: {collapse_threshold:.6f}")
print()

# ── JSD helper ────────────────────────────────────────────────────────────────
def jsd(a, b, eps=1e-10):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a = np.abs(a) + eps;           b = np.abs(b) + eps
    a /= a.sum();                   b /= b.sum()
    return float(jensenshannon(a, b, base=2))

# ── Load Phase 10 centralized baseline (deterministic) ───────────────────────
def load_phase10_baselines():
    rows = list(csv.DictReader(open(os.path.join(PROJECT_ROOT, 'results/phase10_log.csv'))))
    cent = next(json.loads(r['explanation_vector']) for r in rows
                if r['run_type']=='centralized' and r['encoding']=='iqp'
                and r['seed']=='7' and r['round']=='15')
    iid_anom = next(json.loads(r['explanation_vector']) for r in rows
                    if r['run_type']=='federated' and r['distribution']=='iid'
                    and r['encoding']=='iqp' and r['seed']=='7' and r['round']=='15'
                    and r['client_id']=='global')
    return cent, iid_anom

cent_vec, iid_anom_vec = load_phase10_baselines()
cent_norm = float(np.linalg.norm(cent_vec))
print(f"Phase 10 centralized IQP/seed=7: norm={cent_norm:.4f}")
print(f"Phase 10 IID/seed=7 (anomalous draw): norm={float(np.linalg.norm(iid_anom_vec)):.4f}  "
      f"JSD_cent={jsd(iid_anom_vec, cent_vec):.4f}")
print()

# ── Partition geometry check ──────────────────────────────────────────────────
def geometry_check(partition):
    n_total = sum(len(yc) for _, yc in partition)
    total_c1 = int((y_train == 1).sum()); total_c0 = int((y_train == 0).sum())
    flags = []
    for i, (_, yc) in enumerate(partition):
        n_c = len(yc)
        c1 = int((yc == 1).sum()); c0 = n_c - c1
        weight = n_c / n_total
        purity = max(c1, c0) / n_c if n_c > 0 else 0
        c1_share = c1 / total_c1 if total_c1 > 0 else 0
        c0_share = c0 / total_c0 if total_c0 > 0 else 0
        if weight >= 0.60 and purity >= 0.90:
            flags.append(f"G1: C{i} weight={weight:.1%} purity={purity:.1%}")
        if c1_share >= 0.90:
            flags.append(f"G2: C{i} c1_share={c1_share:.1%}")
        if c0_share >= 0.90:
            flags.append(f"G2: C{i} c0_share={c0_share:.1%}")
        print(f"    C{i}: n={n_c:3d}  c1={c1:3d}  weight={weight:.1%}  purity={purity:.1%}")
    return flags

print("Quantity-skew partition for seed=7 (geometry check, MN-08 Trigger G):")
qty_part = quantity_skew_split(X_train, y_train, N_CLIENTS, skew_factor=0.5, seed=SEED)
qty_flags = geometry_check(qty_part)
print(f"  G1/G2 flags: {qty_flags if qty_flags else 'NONE (as expected for quantity-skew)'}")
print()
print("IID partition for seed=7 (geometry check, for reference):")
iid_part = iid_split(X_train, y_train, N_CLIENTS, seed=SEED)
iid_flags = geometry_check(iid_part)
print(f"  G1/G2 flags: {iid_flags if iid_flags else 'NONE (expected)'}")
print()

# ── Logger: captures round-15 global explanation ───────────────────────────────
class CapturingLogger:
    def __init__(self):
        self.final_global_expl = None
        self.final_acc         = None

    def log(self, round_num, client_id, explanation_vector, accuracy):
        if client_id == 'global' and round_num == N_ROUNDS:
            self.final_global_expl = np.asarray(explanation_vector, dtype=float)
            self.final_acc         = float(accuracy)

# ── Ray / timeout helpers ──────────────────────────────────────────────────────
def _timeout_handler(signum, frame):
    raise TimeoutError("Training timed out")

def cleanup_ray():
    try:
        import ray
        if ray.is_initialized(): ray.shutdown()
    except Exception: pass

# ── Single run ─────────────────────────────────────────────────────────────────
def run_one(split, rep_idx, label):
    print(f"  [{label} rep {rep_idx+1}/{N_REPS}]  ", end='', flush=True)
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
            split=split,
            dirichlet_alpha=0.5,  # ignored for iid/quantity; required kwarg
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
    except TimeoutError as e:
        signal.alarm(0); cleanup_ray()
        print(f"TIMEOUT"); return None
    except Exception as e:
        signal.alarm(0); cleanup_ray()
        print(f"ERROR: {e}"); return None

    elapsed = time.time() - t0
    acc = round_log[-1][1] if round_log else float('nan')

    if logger.final_global_expl is None:
        print(f"no expl logged  t={elapsed:.0f}s"); return None

    expl      = logger.final_global_expl
    norm_phi  = float(np.linalg.norm(expl))
    jsd_cent  = jsd(expl, cent_vec)
    track     = 'A' if acc > collapse_threshold else 'B'
    anomalous = norm_phi < NORM_ANOMALY_THRESH or jsd_cent > JSD_CENT_ANOMALY_THRESH

    print(f"acc={acc:.4f}  track={track}  norm={norm_phi:.4f}  "
          f"JSD_cent={jsd_cent:.4f}  {'ANOMALOUS' if anomalous else 'normal'}  t={elapsed:.0f}s")
    return {
        'split': split, 'rep': rep_idx+1, 'acc': acc, 'track': track,
        'norm_phi': norm_phi, 'jsd_cent': jsd_cent,
        'anomalous': anomalous, 'elapsed': elapsed,
        'expl': expl.tolist(),
    }


def main():
    print("=== Phase 14: IQP/seed=7 Cross-Condition Investigation ===")
    print(f"  IQP encoding, seed=7, {N_REPS} reps × 2 conditions (IID, quantity-skew)")
    print(f"  Anomaly criteria: norm_phi < {NORM_ANOMALY_THRESH} OR JSD_cent > {JSD_CENT_ANOMALY_THRESH}")
    print(f"  Normal IQP IID seeds: norm_phi 0.32–0.41, JSD_cent 0.11–0.22")
    print()

    # Show existing Phase 11 data as context
    print("--- Existing data context ---")
    rows11 = list(csv.DictReader(open(os.path.join(PROJECT_ROOT, 'results/phase11_log.csv'))))

    print("IQP/quantity/seed=7 existing draws:")
    for r in sorted([r for r in rows11 if r['encoding']=='iqp' and r['distribution']=='quantity'
                     and r['seed']=='7' and r['round']=='15' and r['client_id']=='global'],
                    key=lambda x: x['timestamp']):
        vec = json.loads(r['explanation_vector'])
        print(f"  {r['run_type']:12}  acc={r['accuracy']}  norm={r['norm_phi']}  "
              f"JSD_cent={jsd(vec, cent_vec):.4f}  track={r['track_class']}")

    print("IQP/alpha_0.5/seed=7 existing draws (Q1 reference):")
    for r in sorted([r for r in rows11 if r['encoding']=='iqp' and r['distribution']=='alpha_0.5'
                     and r['seed']=='7' and r['round']=='15' and r['client_id']=='global'],
                    key=lambda x: x['timestamp']):
        vec = json.loads(r['explanation_vector'])
        print(f"  {r['run_type']:12}  acc={r['accuracy']}  norm={r['norm_phi']}  "
              f"JSD_cent={jsd(vec, cent_vec):.4f}  track={r['track_class']}")
    print()

    results = []

    print("--- Fresh quantity-skew replications ---")
    for rep in range(N_REPS):
        r = run_one('quantity', rep, 'quantity')
        if r: results.append(r)
    print()

    print("--- Fresh IID replications ---")
    for rep in range(N_REPS):
        r = run_one('iid', rep, 'iid')
        if r: results.append(r)
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=== Summary ===")
    print(f"\n{'Split':<10} {'Rep':>4}  {'Acc':>7}  {'Track':>5}  {'norm_phi':>9}  "
          f"{'JSD_cent':>9}  {'Anomalous?'}")
    print(f"  {'-'*10} {'-'*4}  {'-'*7}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*10}")
    for r in results:
        print(f"  {r['split']:<10} {r['rep']:>4}  {r['acc']:>7.4f}  {r['track']:>5}  "
              f"{r['norm_phi']:>9.4f}  {r['jsd_cent']:>9.4f}  "
              f"{'YES' if r['anomalous'] else 'no'}")

    print()
    qty_reps = [r for r in results if r['split'] == 'quantity']
    iid_reps = [r for r in results if r['split'] == 'iid']

    for split, reps in [('quantity', qty_reps), ('iid', iid_reps)]:
        if not reps: continue
        n_anom = sum(1 for r in reps if r['anomalous'])
        norms  = [r['norm_phi'] for r in reps]
        jsds   = [r['jsd_cent'] for r in reps]
        print(f"  {split:<10}: anomalous {n_anom}/{len(reps)}  "
              f"norm_phi={np.mean(norms):.4f}±{np.std(norms):.4f}  "
              f"JSD_cent={np.mean(jsds):.4f}±{np.std(jsds):.4f}")

    # Cross-condition comparison with existing Phase 11 draw
    print()
    p11_qty = next((r for r in rows11 if r['encoding']=='iqp' and r['distribution']=='quantity'
                    and r['seed']=='7' and r['round']=='15' and r['client_id']=='global'
                    and r['run_type']=='federated'), None)
    if p11_qty:
        p11_norm = float(p11_qty['norm_phi'])
        p11_jsd  = jsd(json.loads(p11_qty['explanation_vector']), cent_vec)
        all_qty_norms = [p11_norm] + [r['norm_phi'] for r in qty_reps]
        all_qty_jsds  = [p11_jsd]  + [r['jsd_cent']  for r in qty_reps]
        n_qty_all_anom = sum(1 for n, j in zip(all_qty_norms, all_qty_jsds)
                             if n < NORM_ANOMALY_THRESH or j > JSD_CENT_ANOMALY_THRESH)
        print(f"  quantity (all draws incl. Phase 11 primary): "
              f"anomalous {n_qty_all_anom}/{len(all_qty_norms)}  "
              f"norm_phi={np.mean(all_qty_norms):.4f}±{np.std(all_qty_norms):.4f}  "
              f"JSD_cent={np.mean(all_qty_jsds):.4f}±{np.std(all_qty_jsds):.4f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    qty_anom = sum(1 for r in qty_reps if r['anomalous'])
    iid_anom = sum(1 for r in iid_reps if r['anomalous'])

    print()
    print("=== INTERPRETATION ===")
    print(f"  Fresh quantity-skew anomalous: {qty_anom}/{len(qty_reps)}")
    print(f"  Fresh IID anomalous:           {iid_anom}/{len(iid_reps)}")
    print()

    if qty_anom >= 2 and iid_anom >= 2:
        print("  FINDING: Both conditions show persistent anomaly at comparable rates.")
        print("  → IQP+seed=7 has an initialization-level issue with explanation geometry,")
        print("    independent of partition type.  Quantity-skew is not amplifying anything")
        print("    specific — the same seed is routinely entering a low-norm attractor.")
        print("  → MN-08's IID retraction ('single noisy draw') was premature for this seed.")
        print("  → Update MN-08: IQP/seed=7 is systematically anomaly-prone, not a one-off.")
        print("  → RQ3 table: IQP/quantity/seed=7 should carry a note that the anomaly is")
        print("    likely encoding+initialization specific rather than partition-induced.")

    elif qty_anom >= 2 and iid_anom <= 1:
        print("  FINDING: Quantity-skew anomalous at higher rate than IID.")
        print("  → Quantity-skew partition amplifies IQP+seed=7 attractor instability.")
        print("  → The effect is partition-dependent, not purely initialization-driven.")
        print("  → For RQ3: flag as a genuine partition×encoding×seed interaction.")
        print("    Requires further investigation (different quantity-skew seeds with IQP).")

    elif qty_anom == 0 and iid_anom == 0:
        print("  FINDING: Neither condition anomalous in fresh runs.")
        print("  → Phase 11 primary draw was a single-draw fluke (same as IID MN-08 instance).")
        print("  → Apply single-draw treatment to quantity/seed=7; no ongoing phenomenon.")
        print("  → RQ3 table: downgrade the quantity/seed=7 flag to 'not confirmed'.")

    else:
        print(f"  INCONCLUSIVE: Mixed result (qty {qty_anom}/{len(qty_reps)}, "
              f"iid {iid_anom}/{len(iid_reps)}).")
        print("  → Consider additional runs to resolve.")

    print()
    print("[Phase 14 complete]")


if __name__ == '__main__':
    main()
