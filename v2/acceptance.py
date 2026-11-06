"""Protocol v2 acceptance tests that are too slow for the unit test suite.

Prints only pass or fail and timing. It never prints study outcomes.
"""
import json
import subprocess
import sys
import time

import numpy as np

from v2 import config
from v2.flower_path import run_flower
from v2.train import run_federated

CASES = [("wisconsin", "angle", "alpha_0.5", 7), ("pima", "reuploading", "iid", 42)]
TOLERANCE = 1e-4


def unit_tests():
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                            cwd=config.ROOT, capture_output=True, text=True)
    return result.returncode == 0, result.stderr.strip().splitlines()[-3:]


def main():
    report = {}
    passed, tail = unit_tests()
    report["unit_tests"] = dict(passed=passed, tail=tail)
    for dataset, encoding, regime, seed in CASES:
        spec = config.RunSpec("fed", dataset, encoding, seed, regime)
        name = f"{dataset}/{encoding}/{regime}/s{seed}"
        started = time.monotonic()
        first = run_federated(spec)[0]
        direct_seconds = time.monotonic() - started
        second = run_federated(spec)[0]
        identical = (np.array_equal(first["global_weights"], second["global_weights"])
                     and np.array_equal(first["client_weights"], second["client_weights"], equal_nan=True))
        started = time.monotonic()
        flower = run_flower(spec)
        flower_seconds = time.monotonic() - started
        difference = float(np.abs(flower - first["global_weights"]).max())
        report[name] = dict(passed=identical and difference <= TOLERANCE, repeat_identical=identical,
                            max_difference_flower_vs_direct=difference,
                            direct_seconds=direct_seconds, flower_seconds=flower_seconds)
    report["passed"] = all(v["passed"] for v in report.values())
    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT / "acceptance.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
