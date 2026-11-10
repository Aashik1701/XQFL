"""Descriptive summaries of the protocol v2 sweeps, with matched primary baselines.

The frozen sweep table (v2.analysis.sweep_table) reports coverage, accuracy and tau only. This script adds
RQ1 and RQ3 quantities with the same gate rules as the primary tables. Sweeps explain clients at the final
round only, so RQ3 values here use final-round quantities on both the sweep and the baseline side. The
baseline rows are the primary runs restricted to the 15 sweep seeds. Nothing here is tested (protocol
section 9). Reads results only:
    PYTHONPATH=. xqfl-env/bin/python analysis/sweep_summary_v2.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2 import config, stats
from v2.analysis import (BASE, fidelity_functional, fidelity_table, load_summaries, passes_gate,
                         summarize_cell)
from v2.models import build_model

ROOT = Path("results/protocol_v2")
OUT = ROOT / "tables" / "sweep_descriptive.csv"
FACTORS = ["n_features", "n_layers", "n_clients", "n_rounds"]
CELL = ["dataset", "encoding", "regime", *FACTORS]


def sweep_name(row):
    changed = [k for k in FACTORS if row[k] != BASE[k]]
    if not changed:
        return "datasets" if row["dataset"] in ("spambase", "phoneme") else "baseline"
    assert len(changed) == 1, row
    return {"n_features": "features", "n_layers": "depth", "n_clients": "clients", "n_rounds": "rounds"}[changed[0]]


def parameter_count(encoding, n_features, n_layers):
    model = build_model(encoding, n_features, n_layers, 0)
    return int(sum(p.numel() for p in model.parameters()))


def describe(runs, clients):
    fed = runs[runs.kind == "fed"].copy()
    fed["n_sparse"] = [sum(n < config.SPARSE_CLIENT for n in sizes) for sizes in fed.partition_sizes]
    fed["functional"] = passes_gate(fed, config.GATE_MARGIN)
    fidelity = fidelity_table(runs)
    fidelity = fidelity[fidelity_functional(fidelity, config.GATE_MARGIN)]
    status = fed.set_index(CELL + ["seed"]).functional
    keys = list(zip(*[clients[k] for k in CELL + ["seed"]]))
    clients = clients[status.reindex(keys).fillna(False).to_numpy().astype(bool)]
    client_means = clients.groupby(CELL + ["seed"], as_index=False)[["D_model_I_final", "D_pop_I_final",
                                                                      "D_floor_I_final"]].mean()
    rows = []
    for key, group in fed.groupby(CELL):
        row = dict(zip(CELL, key))
        good = group[group.functional]
        low, high = stats.wilson_interval(len(good), len(group))
        row.update(runs=len(group), functional=len(good), coverage=len(good) / len(group),
                   wilson_low=low, wilson_high=high, test_acc_mean=group.test_acc.mean(),
                   test_acc_sd=group.test_acc.std(ddof=1), test_majority=group.test_majority.mean())
        for name, values in (("tau", good.stability_I_tau), ("tau_init", good.stability_I_tau_init)):
            row.update({f"{name}_{k}": v for k, v in summarize_cell(values).items()})
        part = fidelity[np.logical_and.reduce([fidelity[k] == v for k, v in row.items() if k in CELL])]
        for name in ("jsd_cent_I", "floor_cent_I", "excess_I"):
            row.update({f"{name}_{k}": v for k, v in summarize_cell(part[name]).items()})
        cm = client_means[np.logical_and.reduce([client_means[k] == row[k] for k in CELL])]
        joined = cm.merge(good[["seed", "D_sample_I_final"]], on="seed")
        for name in ("D_model_I_final", "D_pop_I_final", "D_floor_I_final"):
            row.update({f"{name}_{k}": v for k, v in summarize_cell(cm[name]).items()})
        row.update({f"D_sample_I_final_{k}": v for k, v in summarize_cell(joined.D_sample_I_final).items()})
        diff = joined.D_model_I_final - joined.D_sample_I_final
        row.update(model_minus_sample_median=diff.median() if len(diff) else np.nan,
                   model_minus_sample_q25=diff.quantile(0.25) if len(diff) else np.nan,
                   model_minus_sample_q75=diff.quantile(0.75) if len(diff) else np.nan,
                   model_below_sample=int((diff < 0).sum()), model_sample_pairs=len(diff),
                   runs_with_sparse_client=int((group.n_sparse > 0).sum()),
                   mean_sparse_clients=group.n_sparse.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    torch.set_num_threads(1)
    sweep_runs, sweep_clients = load_summaries(ROOT / "sweep")
    primary_runs, primary_clients = load_summaries(ROOT / "primary")
    seeds = set(config.SWEEP_SEEDS)
    keep = (primary_runs.seed.isin(seeds) & (primary_runs.zero_handling == "missing")
            & primary_runs.encoding.isin(config.ENCODINGS) & primary_runs.regime.isin(["iid", "alpha_0.5"])
            & (primary_runs.group == "primary"))
    primary_runs = primary_runs[keep]
    primary_clients = primary_clients[primary_clients.seed.isin(seeds) & primary_clients.encoding.isin(config.ENCODINGS)
                                      & primary_clients.regime.isin(["iid", "alpha_0.5"])
                                      & (primary_clients.zero_handling == "missing")]
    table = pd.concat([describe(primary_runs, primary_clients), describe(sweep_runs, sweep_clients)],
                      ignore_index=True)
    # Pima sweeps use IID only, so the Pima alpha 0.5 baseline has no sweep partner but is kept.
    table.insert(0, "sweep", table.apply(sweep_name, axis=1))
    table["parameters"] = [parameter_count(e, f, l) for e, f, l in zip(table.encoding, table.n_features, table.n_layers)]
    table.to_csv(OUT, index=False)
    print(f"{len(table)} cells written to {OUT}")


if __name__ == "__main__":
    main()
