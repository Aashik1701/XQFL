"""Centralized-vs-federated learning-curve comparison — addresses the paper
review's P1 item 6 (quantify the centralized/federated update-budget
mismatch, review_report.md).

Section 6.4 already states the *reason* accuracy at 45 centralized epochs
isn't directly comparable to federated accuracy at round 15 (same update
COUNT, different update diversity/correlation structure). This script adds
the missing quantitative side: log Wisconsin/amplitude/IID centralized test
accuracy after EVERY epoch (not just the final one), pair centralized epoch
3r against federated round r (since each federated round = 3 local epochs,
15 rounds x 3 = 45 = the centralized epoch budget), and report the gap
trajectory against the federated per-round accuracy already logged in
results/phase10_log.csv.

Scope: 3 seeds (a supplementary diagnostic check, not a headline RQ result —
matches the precision level of other small-n robustness checks already in
the paper, e.g. the 3-draw RQ3 anomaly replications).

Run from project root:
    python experiments/rq_learning_curve.py
"""

import os
import sys
import csv
import json
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn

from data.loaders import load_breast_cancer_data
from models.encodings import amplitude_encoding
from models.quantum_classifier import HybridQuantumClassifier

SEEDS = [1, 7, 42]
N_QUBITS = 4
N_LAYERS = 2
EPOCHS = 45
LR = 0.05
BATCH_SIZE = 32
ROUNDS = 15
LOCAL_EPOCHS = 3   # federated round <-> centralized epoch = round * LOCAL_EPOCHS

RESULTS_CSV = "results/rq_learning_curve_log.csv"


def train_logging_every_epoch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)

    model = HybridQuantumClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=amplitude_encoding)
    X_proc = model._preprocess(X_train, fit=True)
    X_t = torch.tensor(X_proc, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_proc = model._preprocess(X_test)
    X_test_t = torch.tensor(X_test_proc, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    n = X_t.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    acc_per_epoch = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = (model(X_test_t) >= 0.5).float()
        acc = (test_preds == y_test_t).float().mean().item()
        acc_per_epoch.append(acc)
        model.train()
        print(f"  seed={seed}  epoch={epoch:2d}/{EPOCHS}  test_acc={acc:.4f}", flush=True)

    return acc_per_epoch


def load_federated_round_acc(seed):
    """Pull amplitude/iid/global per-round accuracy for this seed from the
    existing phase10_log.csv (already logged; no retraining needed)."""
    rows = []
    with open("results/phase10_log.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["run_type"] == "federated" and r["encoding"] == "amplitude"
                    and r["distribution"] == "iid" and r["client_id"] == "global"
                    and int(r["seed"]) == seed):
                rows.append((int(r["round"]), float(r["accuracy"])))
    rows.sort()
    return dict(rows)


def main():
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["seed", "round", "centralized_epoch", "federated_acc",
                        "centralized_acc", "gap", "timestamp"])

        all_gaps = []
        for seed in SEEDS:
            print(f"\n=== seed={seed} ===")
            fed_acc_by_round = load_federated_round_acc(seed)
            cent_acc_per_epoch = train_logging_every_epoch(seed)

            print(f"\n  round  fed_acc  cent_epoch  cent_acc   gap")
            for rnd in range(1, ROUNDS + 1):
                cent_epoch = rnd * LOCAL_EPOCHS
                fed_acc = fed_acc_by_round.get(rnd)
                cent_acc = cent_acc_per_epoch[cent_epoch - 1]
                if fed_acc is None:
                    continue
                gap = fed_acc - cent_acc
                all_gaps.append(gap)
                print(f"  {rnd:5d}  {fed_acc:.4f}  {cent_epoch:10d}  {cent_acc:.4f}  {gap:+.4f}")
                w.writerow([seed, rnd, cent_epoch, f"{fed_acc:.6f}", f"{cent_acc:.6f}",
                            f"{gap:.6f}", datetime.now(timezone.utc).isoformat()])
            f.flush()

    print(f"\n=== Summary (n={len(all_gaps)} round-checkpoints across {len(SEEDS)} seeds) ===")
    print(f"  Mean gap (federated - centralized): {np.mean(all_gaps):+.4f}")
    print(f"  Std of gap: {np.std(all_gaps, ddof=1):.4f}")
    print(f"  Mean |gap|: {np.mean(np.abs(all_gaps)):.4f}")
    print("\n[learning-curve comparison complete]")


if __name__ == "__main__":
    main()
