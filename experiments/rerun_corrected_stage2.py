"""Corrected federated recompute — finish Stage 1 + run Stage 2.

  0. resume phase16_pima_explanation_sweep  (Stage 1 tail; crashed on disk-full)
  1. phase13_mia_sweep          — Wisconsin IID direct+shadow MIA, 5 seeds
  2. rq4_seed_extension         — Wisconsin IID direct MIA, seeds 11-47
  3. rq4_noniid_mia             — Wisconsin non-IID MIA (alpha_0.1, qty_skew)
  4. rq4_dp_extension           — Wisconsin IID calibrated-Laplace DP sweep
                                  (Δf auto-derived from corrected phase10_log.csv)

Each sub-script is resume-safe via its own checkpoint. Stale Ray session dirs
are purged between steps (nothing else runs concurrently). Aborts if free disk
drops below MIN_FREE_GB before a step.
"""

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MIN_FREE_GB = 5

STEPS = [
    "experiments/phase16_pima_explanation_sweep.py",
    "experiments/phase13_mia_sweep.py",
    "experiments/rq4_seed_extension.py",
    "experiments/rq4_noniid_mia.py",
    "experiments/rq4_dp_extension.py",
]


def free_gb():
    u = shutil.disk_usage(ROOT)
    return u.free / 1e9


def purge_ray():
    base = "/tmp/ray"
    if not os.path.isdir(base):
        return
    for d in os.listdir(base):
        if d.startswith("session_"):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)


def run(script):
    purge_ray()
    fg = free_gb()
    log = os.path.join(ROOT, "results", os.path.basename(script).replace(".py", ".rerun.log"))
    print(f"\n{'='*72}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {script}\n"
          f"  free disk: {fg:.1f} GB   log -> {log}\n{'='*72}", flush=True)
    if fg < MIN_FREE_GB:
        print(f"  !! only {fg:.1f} GB free (< {MIN_FREE_GB}) — aborting before this step.", flush=True)
        sys.exit(2)
    t0 = time.time()
    # append so a resumed step keeps its earlier log
    with open(log, "a") as fh:
        fh.write(f"\n\n===== rerun_corrected_stage2 resume @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.flush()
        rc = subprocess.call([PY, os.path.join(ROOT, script)], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    print(f"[{time.strftime('%H:%M:%S')}] DONE  {script}  rc={rc}  ({dt:.1f} min)  free now {free_gb():.1f} GB", flush=True)
    if rc != 0:
        print(f"  !! non-zero exit — inspect {log}. Stopping.", flush=True)
        sys.exit(rc)


if __name__ == "__main__":
    t0 = time.time()
    for s in STEPS:
        run(s)
    purge_ray()
    print(f"\n[STAGE 2 COMPLETE] total {(time.time()-t0)/3600:.2f} h", flush=True)
