"""Corrected federated recompute — Stage 1 (explanation sweeps).

Runs, in order, the federated explanation sweeps invalidated by the
node_id%n_clients partition-assignment bug (fixed in
federation/run_federation.client_fn + guarded in federation/server.py):

  1. phase10_explanation_sweep      — Wisconsin IID (5 seeds) + amplitude non-IID
  2. rq2_seed_extension             — Wisconsin IID seeds 11-47 (RQ2/RQ4 n=15)
  3. phase11_encoding_noniid_sweep  — Wisconsin angle/reup/iqp x non-IID x 5 seeds
  4. phase16_pima_explanation_sweep — Pima all encodings x all distributions

Centralized baselines are NOT re-run (unaffected); phase10/phase16 logs were
pre-stripped to centralized-only rows + checkpoints so the scripts resume by
re-running every federated key against the fixed pipeline.

Each sub-script is resume-safe via its own results/<name>_checkpoint.csv, so
this orchestrator can be killed and restarted.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

STAGE1 = [
    "experiments/phase10_explanation_sweep.py",
    "experiments/rq2_seed_extension.py",
    "experiments/phase11_encoding_noniid_sweep.py",
    "experiments/phase16_pima_explanation_sweep.py",
]


def run(script):
    log = os.path.join(ROOT, "results", os.path.basename(script).replace(".py", ".rerun.log"))
    print(f"\n{'='*72}\n[{time.strftime('%H:%M:%S')}] START {script}\n  log -> {log}\n{'='*72}", flush=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call([PY, os.path.join(ROOT, script)], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {script}  rc={rc}  ({dt/60:.1f} min)", flush=True)
    if rc != 0:
        print(f"  !! non-zero exit — inspect {log}. Stopping stage.", flush=True)
        sys.exit(rc)


if __name__ == "__main__":
    t0 = time.time()
    for s in STAGE1:
        run(s)
    print(f"\n[STAGE 1 COMPLETE] total {(time.time()-t0)/3600:.2f} h", flush=True)
