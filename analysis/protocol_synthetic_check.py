"""Post-hoc validation of the protocol with a known model difference.

Take each functional final IID global model (both datasets, four encodings, 30 seeds) and make a second
model by adding Gaussian noise to its weights at a known relative scale s (noise standard deviation
s * ||w|| / sqrt(P)). Then vary the two factors independently:
  model only     J(w on A ; w' on A)     same records and background, model changes by s
  records only   J(w on A ; w on A')     same model, two disjoint 50-record test sets
  both           J(w on A ; w' on A')
A = E_g, A' = a second disjoint, stratified 50-record test set, both explained against B_g. If the
protocol's quantities behave as intended, the model-only distance is zero at s = 0 and grows with s, the
records-only distance does not depend on s, and the combined distance is at least the larger of the two.
Not part of the frozen protocol.

    PYTHONPATH=. OMP_NUM_THREADS=1 xqfl-env/bin/python analysis/protocol_synthetic_check.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explain.metrics import js_divergence
from v2 import config
from v2.data import _rng, explanation_sets, make_data, stratified_pick
from v2.explain import explain_with_weights, importance
from v2.models import build_model, load_flat, predict_probabilities
from v2.train import initial_seed

ROOT = Path("results/protocol_v2/primary")
OUT = Path("results/protocol_v2/posthoc")
SCALES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def model_rows(folder):
    torch.set_num_threads(1)
    summary = json.loads((folder / "summary.json").read_text())
    spec = summary["spec"]
    data = make_data(spec["dataset"], spec["seed"], spec["n_features"], spec["zero_handling"])
    with np.load(folder / "weights.npz") as w:
        weights = w["global_weights"][-1]
        parts = [w[f"partition_{k}"] for k in range(spec["n_clients"])]
    sets = explanation_sets(data, parts, spec["seed"])
    others = np.setdiff1d(np.arange(len(data.y_test)), sets.E_global)
    second = others[stratified_pick(data.y_test[others], 50, 0, _rng(spec["seed"], 97))[0]]
    A, A2, B = data.X_test[sets.E_global], data.X_test[second], data.X_train[sets.B_global]
    model = build_model(spec["encoding"], spec["n_features"], spec["n_layers"], initial_seed(spec["seed"]))
    base_A = importance(explain_with_weights(model, weights, A, B))
    base_A2 = importance(explain_with_weights(model, weights, A2, B))
    load_flat(model, weights)
    base_pred = predict_probabilities(model, data.X_test) >= 0.5
    rng = _rng(spec["seed"], 96)
    rows = []
    for s in SCALES:
        noise = rng.standard_normal(len(weights)) * s * np.linalg.norm(weights) / np.sqrt(len(weights))
        perturbed = weights + noise
        I_A = importance(explain_with_weights(model, perturbed, A, B))
        I_A2 = importance(explain_with_weights(model, perturbed, A2, B))
        load_flat(model, perturbed)
        flips = float(((predict_probabilities(model, data.X_test) >= 0.5) != base_pred).mean())
        rows.append(dict(dataset=spec["dataset"], encoding=spec["encoding"], seed=spec["seed"], scale=s,
                         model_only=js_divergence(base_A, I_A), records_only=js_divergence(base_A, base_A2),
                         both=js_divergence(base_A, I_A2), prediction_flips=flips))
    return rows


def main():
    folders = []
    for e in config.ENCODINGS:
        for f in sorted(ROOT.glob(f"fed_*_{e}_iid_s*_t15_missing")):
            s = json.loads((f / "summary.json").read_text())
            if s["val_acc"] > s["val_majority"] + config.GATE_MARGIN:
                folders.append(f)
    with Pool(7) as pool:
        frame = pd.DataFrame([r for chunk in pool.map(model_rows, folders) for r in chunk])
    frame.to_csv(OUT / "synthetic_per_model.csv", index=False)
    summary = frame.groupby(["dataset", "encoding", "scale"])[["model_only", "records_only", "both",
                                                                "prediction_flips"]].mean().round(4)
    summary.to_csv(OUT / "synthetic_summary.csv")
    print(len(folders), "models"); print(summary.to_string())


if __name__ == "__main__":
    main()
