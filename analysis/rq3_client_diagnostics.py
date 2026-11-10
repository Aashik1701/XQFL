"""Post-hoc RQ3 diagnostics: client-level functionality, and the record and background parts of D_floor.

For every primary federated quantum run and every non-sparse client at rounds 3, 6, 9, 12 and 15:
  * the client model's validation accuracy, and whether it passes the same gate as the global model
    (validation accuracy > validation majority rate + 0.02), and whether the client holds one class only;
  * D_model, D_pop and D_floor per client and round, recomputed from the saved explanations;
  * D_E = J(global on E_k with B_g ; global on E_g with B_g): the explained records change, background fixed;
  * D_B = J(global on E_k with B_k ; global on E_k with B_g): the background changes, records fixed.
D_E needs one new explanation set per client and round; everything else comes from saved weights and
attributions. Not part of the frozen protocol.

    PYTHONPATH=. OMP_NUM_THREADS=1 xqfl-env/bin/python analysis/rq3_client_diagnostics.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explain.metrics import js_divergence
from v2 import config
from v2.data import explanation_sets, make_data
from v2.explain import explain_with_weights, importance
from v2.models import accuracy, build_model, load_flat
from v2.train import initial_seed

ROOT = Path("results/protocol_v2/primary")
OUT = Path("results/protocol_v2/posthoc")


def run_rows(folder):
    torch.set_num_threads(1)
    summary = json.loads((folder / "summary.json").read_text())
    spec = summary["spec"]
    data = make_data(spec["dataset"], spec["seed"], spec["n_features"], spec["zero_handling"])
    with np.load(folder / "weights.npz") as w:
        history, clients = w["global_weights"], w["client_weights"]
        parts = [w[f"partition_{k}"] for k in range(spec["n_clients"])]
    with np.load(folder / "explanations.npz") as z:
        phi_global, matched, population, floor = (z["phi_global"].astype(float), z["phi_matched"].astype(float),
                                                  z["phi_population"].astype(float), z["phi_floor"].astype(float))
        rounds = z["client_rounds"]
    sets = explanation_sets(data, parts, spec["seed"])
    model = build_model(spec["encoding"], spec["n_features"], spec["n_layers"], initial_seed(spec["seed"]))
    majority = float(max(data.y_val.mean(), 1 - data.y_val.mean()))
    B_g = data.X_train[sets.B_global]
    global_functional = summary["val_acc"] > summary["val_majority"] + config.GATE_MARGIN
    rows = []
    for i, t in enumerate(rounds):
        I_g = importance(phi_global[t])
        for k in range(spec["n_clients"]):
            if sets.E_local[k] is None:
                continue
            load_flat(model, clients[t - 1, k])
            client_val = accuracy(model, data.X_val, data.y_val)
            E_k = data.X_train[sets.E_local[k]]
            phi_e = explain_with_weights(model, history[t], E_k, B_g)  # global weights on E_k with B_g
            I_floor = importance(floor[i, k])  # global on E_k with B_k
            rows.append(dict(
                dataset=spec["dataset"], encoding=spec["encoding"], regime=spec["regime"], seed=spec["seed"],
                client=k, round=int(t), size=len(parts[k]),
                single_class=bool(len(np.unique(data.y_train[parts[k]])) < 2),
                client_val_acc=client_val, val_majority=majority,
                client_functional=bool(client_val > majority + config.GATE_MARGIN),
                global_functional=bool(global_functional),
                D_model=js_divergence(importance(matched[i, k]), I_g),
                D_pop=js_divergence(importance(population[i, k]), I_g),
                D_floor=js_divergence(I_floor, I_g),
                D_E=js_divergence(importance(phi_e), I_g),
                D_B=js_divergence(I_floor, importance(phi_e))))
    return rows


def main():
    folders = [f for e in config.ENCODINGS for f in sorted(ROOT.glob(f"fed_*_{e}_*_s*_t15_missing"))]
    with Pool(7) as pool:
        rows = [r for chunk in pool.map(run_rows, folders) for r in chunk]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "client_diagnostics.csv", index=False)
    print(len(folders), "runs,", len(frame), "client-rounds")


if __name__ == "__main__":
    main()
