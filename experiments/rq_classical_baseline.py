"""Classical FedAvg baseline — addresses the paper-review circularity concern
(review_report.md, DA-1 / P0 items 1-2).

The Devil's Advocate review argues the RQ1 fidelity result (federated
attributions match centralized attributions) may be circular: both models
share the same 4-PCA input space and a low-capacity architecture, so their
attributions might converge for *any* model at this capacity, not because
federation specifically preserves quantum explanations.

This script tests that directly: train a CLASSICAL model — a small MLP whose
parameter count is matched to the quantum VQC (13 params: 2 layers x 4 qubits
+ 4 qubit-readout + 1 bias, giving a 2-unit hidden layer) — through the exact
same pipeline as the quantum encodings:
  - the same shared, server-fit PCA(4) + MinMax([0, pi]) preprocessing
    (fit once, broadcast unchanged, never refit per client)
  - the same partitioning (IID, K=4 clients)
  - the same federation protocol (FedAvg, 15 rounds, 3 local epochs)
  - the same centralized reference protocol (45 epochs on pooled data)
  - the same attribution estimator: explain.q_shap.q_shap_values with
    N_GLOBAL_BG=20 background samples and N_PERM=8 permutations, averaged
    over N_GLOBAL_EXPLAIN=10 test samples (identical constants to
    experiments/phase10_explanation_sweep.py)
  - the same JSD formula (explain.metrics.js_divergence)

If federated-vs-centralized JSD is comparable between the classical MLP and
the quantum encodings, the low-distortion result is not quantum-specific —
it is a property of FedAvg at this capacity/dataset scale, and the paper's
claim must be reframed accordingly. If the classical JSD differs materially,
that demonstrates the JSD metric IS sensitive to what changes when the
architecture changes, rebutting the circularity concern for the quantum
result specifically.

Run from project root:
    python experiments/rq_classical_baseline.py

Runtime estimate: under a minute total (classical MLP, no circuit simulation).
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
from data.partition import iid_split
from models.classical_baseline import build_matched_mlp
from explain.q_shap import q_shap_values
from explain.metrics import js_divergence

# ── config — mirrors experiments/phase10_explanation_sweep.py exactly ──────
N_QUBITS         = 4
N_LAYERS         = 2          # matches the quantum VQC's 2 BasicEntanglerLayers
N_CLIENTS        = 4
N_ROUNDS         = 15
LOCAL_EPOCHS     = 3
CENTRAL_EPOCHS   = 45
LR               = 0.05
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20

SEEDS = [1, 7, 42, 123, 999]   # Wisconsin primary seed set

RESULTS_CSV = "results/rq_classical_baseline_log.csv"
RESULTS_COLS = [
    "seed", "fed_test_acc", "cent_test_acc",
    "jsd_cent", "fed_attr", "cent_attr", "n_params", "timestamp",
]


def _fit_shared_preprocessor(X_train, n_qubits):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_pca)
    return pca, scaler


def _transform(X, pca, scaler):
    return scaler.transform(pca.transform(X)).astype(np.float32)


def local_train(state_dict, X_q, y, epochs, lr):
    model = build_matched_mlp(N_QUBITS, N_LAYERS)
    model.load_state_dict(state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    Xt = torch.tensor(X_q, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        preds = model(Xt).squeeze(1)
        loss = crit(preds, yt)
        loss.backward()
        opt.step()
    return {k: v.clone() for k, v in model.state_dict().items()}, len(X_q)


def fedavg(states_and_ns):
    total_n = sum(n for _, n in states_and_ns)
    keys = states_and_ns[0][0].keys()
    avg = {}
    for k in keys:
        avg[k] = sum(s[k] * (n / total_n) for s, n in states_and_ns)
    return avg


def train_central(X_q, y, epochs, lr, seed):
    torch.manual_seed(seed)
    model = build_matched_mlp(N_QUBITS, N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    Xt = torch.tensor(X_q, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model.train()
    n = len(X_q)
    batch_size = 32
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            preds = model(Xt[idx]).squeeze(1)
            loss = crit(preds, yt[idx])
            loss.backward()
            opt.step()
    return model


def explain_global(model, X_test_q, X_train_q, seed):
    model.eval()

    def predict_fn(X_q):
        with torch.no_grad():
            out = model(torch.tensor(X_q.astype(np.float32)))
        return out.squeeze(-1).numpy()

    X_test_sub = X_test_q[:N_GLOBAL_EXPLAIN]
    X_bg_sub = X_train_q[:N_GLOBAL_BG]
    qshap = q_shap_values(predict_fn, X_test_sub, X_bg_sub, n_permutations=N_PERM, seed=seed)
    return qshap.mean(axis=0)


def eval_acc(model, X_q, y):
    model.eval()
    with torch.no_grad():
        preds = (model(torch.tensor(X_q, dtype=torch.float32)).squeeze(1).numpy() >= 0.5).astype(np.float32)
    return float((preds == y).mean())


def append_result(seed, fed_acc, cent_acc, jsd, fed_attr, cent_attr, n_params):
    write_header = not os.path.exists(RESULTS_CSV)
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(RESULTS_COLS)
        w.writerow([
            seed, f"{fed_acc:.6f}", f"{cent_acc:.6f}", f"{jsd:.6f}",
            json.dumps(fed_attr.tolist()), json.dumps(cent_attr.tolist()),
            n_params, datetime.now(timezone.utc).isoformat(),
        ])


def main():
    print("=== Classical FedAvg baseline (Wisconsin, IID, n=5 seeds) ===")
    print("Addresses review_report.md DA-1 / P0 items 1-2 (circularity concern)")
    n_params_ref = build_matched_mlp(N_QUBITS, N_LAYERS)
    n_params = sum(p.numel() for p in n_params_ref.parameters())
    print(f"  Matched MLP parameter count: {n_params} "
          f"(quantum VQC reference: {N_LAYERS * N_QUBITS + N_QUBITS + 1})")

    jsds = []
    for seed in SEEDS:
        print(f"\n[seed={seed}]")
        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)
        pca, scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
        X_train_q = _transform(X_train, pca, scaler)
        X_test_q = _transform(X_test, pca, scaler)

        # --- federated (FedAvg, IID, 4 clients, 15 rounds x 3 local epochs) ---
        partitions = iid_split(X_train, y_train, N_CLIENTS, seed=seed)
        torch.manual_seed(seed)
        global_state = {k: v.clone() for k, v in build_matched_mlp(N_QUBITS, N_LAYERS).state_dict().items()}
        for rnd in range(1, N_ROUNDS + 1):
            local_states = []
            for Xk, yk in partitions:
                Xk_q = _transform(Xk, pca, scaler)
                state, nk = local_train(global_state, Xk_q, yk, LOCAL_EPOCHS, LR)
                local_states.append((state, nk))
            global_state = fedavg(local_states)
        fed_model = build_matched_mlp(N_QUBITS, N_LAYERS)
        fed_model.load_state_dict(global_state)
        fed_acc = eval_acc(fed_model, X_test_q, y_test)
        fed_attr = explain_global(fed_model, X_test_q, X_train_q, seed)
        print(f"  federated  test_acc={fed_acc:.4f}  attr={fed_attr}")

        # --- centralized (45 epochs, pooled data, same shared PCA) ---
        cent_model = train_central(X_train_q, y_train, CENTRAL_EPOCHS, LR, seed)
        cent_acc = eval_acc(cent_model, X_test_q, y_test)
        cent_attr = explain_global(cent_model, X_test_q, X_train_q, seed)
        print(f"  centralized test_acc={cent_acc:.4f}  attr={cent_attr}")

        jsd = js_divergence(fed_attr, cent_attr)
        jsds.append(jsd)
        print(f"  JSD(fed, cent) = {jsd:.4f}")

        append_result(seed, fed_acc, cent_acc, jsd, fed_attr, cent_attr, n_params)

    print(f"\n=== Summary (n={len(jsds)}) ===")
    print(f"  Classical MLP JSD(fed, cent): {np.mean(jsds):.4f} ± {np.std(jsds, ddof=1):.4f}")
    print(f"  Compare to quantum Wisconsin IID JSD_cent (from table1_rq1_fidelity.csv):")
    print(f"    angle:       0.132 ± 0.046")
    print(f"    amplitude:   0.139 ± 0.028")
    print(f"    reuploading: 0.143 ± 0.034")
    print(f"    iqp:         0.164 ± 0.044  (n=4, seed=7 excluded — MN-09)")
    print("\n[classical baseline complete]")


if __name__ == "__main__":
    main()
