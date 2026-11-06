"""Run protocol v2 stages.

  xqfl-env/bin/python -m v2.run freeze
  xqfl-env/bin/python -m v2.run primary --workers 8
  xqfl-env/bin/python -m v2.run sweep --workers 8
  xqfl-env/bin/python -m v2.run dry --workers 8

Stages resume: a finished run is skipped. Code changes after the freeze stop the
run, so one output directory never mixes code versions without a recorded deviation.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import replace

from v2 import config


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)


def _call(task):
    name, args = task
    try:
        from v2 import estimators, explain, mia, train

        functions = dict(train=train.train_task, explain=explain.explain_task,
                         target=mia.target_task, control=mia.control_task,
                         validate=estimators.validate_task)
        return name, str(args[0]), functions[name](*args), None
    except Exception:
        return name, str(args[0]), "failed", traceback.format_exc()


def run_tasks(label, tasks, workers, log_dir):
    started = time.monotonic()
    counts = {}
    errors = log_dir / "errors.jsonl"
    log_dir.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for i, (name, what, status, error) in enumerate(pool.imap_unordered(_call, tasks), 1):
            counts[status] = counts.get(status, 0) + 1
            if error:
                with errors.open("a") as f:
                    f.write(json.dumps(dict(name=name, what=what, error=error)) + "\n")
            if i % 20 == 0 or i == len(tasks):
                print(f"[{label}] {i}/{len(tasks)} {counts} {time.monotonic() - started:.0f}s", flush=True)
    return dict(tasks=len(tasks), counts=counts, seconds=time.monotonic() - started)


def execute(specs, out, workers, controls=(), validate_extended=()):
    """Train, explain, validate estimators and attack the given specs."""
    report = {}
    report["train"] = run_tasks("train", [("train", (s, out)) for s in specs], workers, out)
    report["explain"] = run_tasks("explain", [("explain", (s, out)) for s in specs], workers, out)
    iid_final = [s for s in specs if s.kind == "fed" and s.regime == "iid"
                 and s.encoding in config.ENCODINGS and s.zero_handling == "missing"
                 and s.group == "primary"]
    extended = set(validate_extended)
    report["estimators"] = run_tasks(
        "estimators", [("validate", (s, out, s.seed in extended)) for s in iid_final], workers, out)
    targets = [s for s in specs if s.kind == "fed" and s.encoding in config.ENCODINGS
               and s.zero_handling == "missing"]
    report["attacks"] = run_tasks("attacks", [("target", (s, out)) for s in targets], workers, out)
    if controls:
        report["controls"] = run_tasks("controls", [("control", (k, s, out)) for k, s in controls], workers, out)
    return report


def check_freeze(out):
    manifest = out / "manifest.json"
    if not manifest.exists():
        raise SystemExit("Protocol is not frozen. Run acceptance, then `freeze`.")
    frozen = json.loads(manifest.read_text())
    if frozen["protocol_sha256"] != config.protocol_hash():
        raise SystemExit("Protocol file changed after the freeze. Record a deviation and re-freeze.")
    if frozen["code_sha256"] != config.code_fingerprint():
        raise SystemExit("Code changed after the freeze. Record a deviation and re-freeze.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "dry", "primary", "sweep", "tables"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root", default=str(config.OUTPUT))
    args = parser.parse_args()
    from pathlib import Path

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage == "freeze":
        acceptance = root / "acceptance.json"
        if not acceptance.exists() or not json.loads(acceptance.read_text())["passed"]:
            raise SystemExit("Acceptance has not passed.")
        (root / "manifest.json").write_text(json.dumps(dict(
            protocol_sha256=config.protocol_hash(), code_sha256=config.code_fingerprint(),
            frozen_at=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
        print("Frozen.")
        return

    if args.stage == "dry":
        out = root / "dry_run"
        specs = config.primary_specs((1,))
        controls = [("mlp_control", 1), ("vqc_control", 1)]
        report = execute(specs, out, args.workers, controls, validate_extended=(1,))
    elif args.stage == "primary":
        check_freeze(root)
        out = root / "primary"
        controls = [(k, s) for s in config.SEEDS for k in ("mlp_control", "vqc_control")]
        # Extended validation uses the first 5 seeds, as the protocol sets out.
        report = execute(config.primary_specs(), out, args.workers, controls,
                         validate_extended=config.SEEDS[:5])
    elif args.stage == "sweep":
        check_freeze(root)
        out = root / "sweep"
        report = execute(config.sweep_specs(), out, args.workers)
    else:
        from v2.analysis import build_all, sweep_table

        print(build_all(root / "primary", root / "tables"))
        if (root / "sweep").exists():
            sweep_table(root / "sweep", root / "tables")
        return
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {"counts": v["counts"], "seconds": round(v["seconds"])} for k, v in report.items()}))


if __name__ == "__main__":
    main()
