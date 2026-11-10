"""Post-hoc check: record-sampling baseline at the same record count as D_model.

The planned D_sample compares two disjoint 25-record halves of E_g, while D_model uses all 50 records of
E_g. This check explains two further disjoint, stratified 50-record test sets per IID primary run with the
saved global weights and the same background B_g, at the client rounds 3, 6, 9, 12 and 15, and repeats the
planned D_model vs D_sample test against this 50-record baseline. It also recomputes the 25-record baseline
on halves of the first set, so both sizes come from the same records. Not part of the frozen protocol.

    PYTHONPATH=. OMP_NUM_THREADS=1 xqfl-env/bin/python analysis/rq3_sample_size_check.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explain.metrics import js_divergence
from v2 import config, stats
from v2.analysis import base_only, load_summaries, passes_gate
from v2.data import _rng, explanation_sets, make_data, partition_indices, stratified_pick
from v2.explain import client_explanation_rounds, explain_with_weights, importance
from v2.models import build_model
from v2.train import initial_seed

ROOT = Path("results/protocol_v2")
OUT = ROOT / "posthoc"
SIZE = 50


def check_run(folder):
    torch.set_num_threads(1)
    summary = json.loads((folder / "summary.json").read_text())
    spec = summary["spec"]
    data = make_data(spec["dataset"], spec["seed"], spec["n_features"], spec["zero_handling"])
    parts = partition_indices(data.y_train, spec["n_clients"], spec["regime"], spec["seed"])
    sets = explanation_sets(data, parts, spec["seed"])
    first, second = stratified_pick(data.y_test, SIZE, SIZE, _rng(spec["seed"], 99))
    half_a, half_b = stratified_pick(data.y_test[first], SIZE // 2, SIZE - SIZE // 2, _rng(spec["seed"], 98))
    model = build_model(spec["encoding"], spec["n_features"], spec["n_layers"], initial_seed(spec["seed"]))
    history = np.load(folder / "weights.npz")["global_weights"]
    background = data.X_train[sets.B_global]

    class Spec:  # client_explanation_rounds needs only these two fields
        explain_mode, n_rounds = spec["explain_mode"], spec["n_rounds"]

    d50, d25 = [], []
    for t in client_explanation_rounds(Spec):
        phi_a = explain_with_weights(model, history[t], data.X_test[first], background)
        phi_b = explain_with_weights(model, history[t], data.X_test[second], background)
        d50.append(js_divergence(importance(phi_a), importance(phi_b)))
        d25.append(js_divergence(importance(phi_a[half_a]), importance(phi_a[half_b])))
    return dict(dataset=spec["dataset"], encoding=spec["encoding"], seed=spec["seed"],
                D_sample50_I=float(np.mean(d50)), D_sample25_same_records_I=float(np.mean(d25)),
                records_overlapping_E_g=int(len(np.intersect1d(np.concatenate([first, second]), sets.E_global))))


def main():
    folders = [f for e in config.ENCODINGS for f in sorted((ROOT / "primary").glob(f"fed_*_{e}_iid_s*_t15_missing"))]
    assert len(folders) == 2 * len(config.ENCODINGS) * len(config.SEEDS), len(folders)
    with Pool(7) as pool:
        per_run = pd.DataFrame(pool.map(check_run, folders))
    OUT.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(OUT / "sample_size_per_run.csv", index=False)

    runs, clients = load_summaries(ROOT / "primary")
    fed = base_only(runs[(runs.kind == "fed") & (runs.regime == "iid")]).copy()
    fed = fed[fed.encoding.isin(config.ENCODINGS) & passes_gate(fed, config.GATE_MARGIN)]
    clients = base_only(clients[(clients.regime == "iid") & clients.encoding.isin(config.ENCODINGS)])
    model = clients.groupby(["dataset", "encoding", "seed"]).D_model_I.mean().rename("D_model_I").reset_index()
    joined = (fed[["dataset", "encoding", "seed", "D_sample_I"]].merge(model, on=["dataset", "encoding", "seed"])
              .merge(per_run, on=["dataset", "encoding", "seed"]))
    rows = []
    for (dataset, encoding), g in joined.groupby(["dataset", "encoding"]):
        for baseline in ("D_sample_I", "D_sample25_same_records_I", "D_sample50_I"):
            result = stats.paired_test(g.D_model_I.to_numpy(), g[baseline].to_numpy())
            rows.append(dict(dataset=dataset, encoding=encoding, baseline=baseline, D_model_mean=g.D_model_I.mean(),
                             baseline_mean=g[baseline].mean(), **result))
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby("baseline").p.transform(lambda p: stats.holm(p.to_numpy()))
    table.to_csv(OUT / "sample_size_tests.csv", index=False)
    print(table[["dataset", "encoding", "baseline", "n", "D_model_mean", "baseline_mean", "median_diff", "ci_low",
                 "ci_high", "p_holm"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
