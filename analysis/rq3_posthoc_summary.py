"""Summaries for the post-hoc RQ3 checks (not part of the frozen protocol).

Inputs: results/protocol_v2/posthoc/client_diagnostics.csv (analysis/rq3_client_diagnostics.py) and the
equal-size label-skew runs in results/protocol_v2/posthoc_label_equal (analysis/label_equal_control.py).
Outputs, in results/protocol_v2/posthoc:
  client_functionality.csv     share of client-rounds that pass the gate, by dataset, regime and encoding
  client_subset_tests.csv      RQ3 regime contrasts with all clients, functional clients only, and
                               multi-class clients only (same paired Wilcoxon and HL as the protocol)
  floor_decomposition.csv      mean D_model, D_pop, D_floor, D_E and D_B by dataset and regime
  label_equal_tests.csv        D_model under equal-size label skew vs IID and vs Dirichlet alpha = 0.5
  label_equal_coverage.csv     functional coverage and client-mix statistics for the equal-size regime

    PYTHONPATH=. xqfl-env/bin/python analysis/rq3_posthoc_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from v2 import config, stats
from v2.analysis import base_only, load_summaries, passes_gate

POST = Path("results/protocol_v2/posthoc")
PRIMARY = Path("results/protocol_v2/primary")
EQUAL = Path("results/protocol_v2/posthoc_label_equal")
KEYS = ["dataset", "encoding", "regime", "seed"]
NON_IID = ("qty_skew", "alpha_0.5", "alpha_0.1")
ENCODINGS = ("angle", "amplitude", "reuploading", "iqp")


def contrast(a, b, column):
    joined = pd.concat([a.set_index("seed")[column].rename("a"), b.set_index("seed")[column].rename("b")],
                       axis=1, join="inner").dropna()
    result = stats.paired_test(joined.a.to_numpy(), joined.b.to_numpy())
    return dict(mean_a=joined.a.mean(), mean_b=joined.b.mean(), **result)


def client_tables():
    d = pd.read_csv(POST / "client_diagnostics.csv")
    d = d[d.global_functional]
    functionality = (d.groupby(["dataset", "regime", "encoding"])
                     .agg(client_rounds=("client_functional", "size"),
                          functional_share=("client_functional", "mean"),
                          single_class_share=("single_class", "mean"))
                     .reset_index())
    pooled = (d.groupby(["dataset", "regime"])
              .agg(client_rounds=("client_functional", "size"), functional_share=("client_functional", "mean"),
                   single_class_share=("single_class", "mean")).reset_index().assign(encoding="all"))
    pd.concat([functionality, pooled]).to_csv(POST / "client_functionality.csv", index=False)

    subsets = {"all": d, "functional": d[d.client_functional], "multi_class": d[~d.single_class]}
    rows = []
    for name, part in subsets.items():
        means = part.groupby(KEYS, as_index=False)[["D_model"]].mean()
        for dataset in config.DATASETS:
            for encoding in ENCODINGS:
                cell = means[(means.dataset == dataset) & (means.encoding == encoding)]
                for regime in NON_IID:
                    r = contrast(cell[cell.regime == regime], cell[cell.regime == "iid"], "D_model")
                    rows.append(dict(subset=name, dataset=dataset, encoding=encoding, regime=regime, **r))
    tests = pd.DataFrame(rows)
    tests.to_csv(POST / "client_subset_tests.csv", index=False)

    columns = ["D_model", "D_pop", "D_floor", "D_E", "D_B"]
    run_means = d.groupby(KEYS, as_index=False)[columns].mean()
    floor = run_means.groupby(["dataset", "regime"])[columns].mean().reset_index()
    floor["runs"] = run_means.groupby(["dataset", "regime"]).size().to_numpy()
    floor.to_csv(POST / "floor_decomposition.csv", index=False)
    by_encoding = run_means.groupby(["dataset", "encoding", "regime"])[columns].mean().reset_index()
    by_encoding.to_csv(POST / "floor_decomposition_by_encoding.csv", index=False)
    return functionality, tests, floor


def run_level(group_dir):
    runs, clients = load_summaries(group_dir)
    runs = base_only(runs[runs.kind == "fed"]).copy()
    runs = runs[runs.encoding.isin(ENCODINGS)]
    runs["functional"] = passes_gate(runs, config.GATE_MARGIN)
    clients = base_only(clients)
    means = clients.groupby(KEYS, as_index=False)[["D_model_I"]].mean()
    good = runs[runs.functional][KEYS]
    return runs, means.merge(good, on=KEYS)


def client_mix(folder):
    """Client sizes and class-1 shares for one run, from its saved partitions."""
    import json
    from v2.data import make_data
    spec = json.loads((folder / "summary.json").read_text())["spec"]
    data = make_data(spec["dataset"], spec["seed"], spec["n_features"], spec["zero_handling"])
    with np.load(folder / "weights.npz") as w:
        parts = [w[f"partition_{k}"] for k in range(spec["n_clients"])]
    sizes = np.array([len(p) for p in parts])
    shares = np.array([data.y_train[p].mean() for p in parts])
    return dict(dataset=spec["dataset"], seed=spec["seed"], size_ratio=sizes.max() / sizes.min(),
                min_size=sizes.min(), share_gap=np.abs(shares - data.y_train.mean()).mean(),
                single_class=int((np.minimum(shares, 1 - shares) == 0).sum()))


def label_equal():
    equal_runs, equal = run_level(EQUAL)
    primary_runs, primary = run_level(PRIMARY)
    rows = []
    for dataset in config.DATASETS:
        for encoding in ENCODINGS:
            e = equal[(equal.dataset == dataset) & (equal.encoding == encoding)]
            for reference in ("iid", "alpha_0.5"):
                p = primary[(primary.dataset == dataset) & (primary.encoding == encoding)
                            & (primary.regime == reference)]
                rows.append(dict(dataset=dataset, encoding=encoding, reference=reference,
                                 **contrast(e, p, "D_model_I")))
    tests = pd.DataFrame(rows)
    tests["p_holm"] = stats.holm(tests.p.fillna(1).to_numpy())
    tests.to_csv(POST / "label_equal_tests.csv", index=False)

    coverage = []
    both = pd.concat([equal_runs, primary_runs[primary_runs.regime.isin(["iid", "alpha_0.5"])]])
    for (dataset, regime, encoding), g in both.groupby(["dataset", "regime", "encoding"]):
        low, high = stats.wilson_interval(int(g.functional.sum()), len(g))
        coverage.append(dict(dataset=dataset, regime=regime, encoding=encoding, runs=len(g),
                             functional=int(g.functional.sum()), low=low, high=high))
    coverage = pd.DataFrame(coverage)
    folders = sorted({f.parent for f in EQUAL.glob("fed_*_angle_*/summary.json")})
    mix = pd.DataFrame([client_mix(f) for f in folders])
    mix_summary = mix.groupby("dataset")[["size_ratio", "min_size", "share_gap", "single_class"]].mean()
    coverage.to_csv(POST / "label_equal_coverage.csv", index=False)
    mix_summary.to_csv(POST / "label_equal_mix.csv")
    return tests, coverage, mix_summary


def main():
    pd.set_option("display.width", 200)
    functionality, tests, floor = client_tables()
    pass
    print(pd.read_csv(POST / "client_functionality.csv").round(3).to_string(index=False))
    print(tests[["subset", "dataset", "encoding", "regime", "n", "median_diff", "p"]].round(4).to_string(index=False))
    print(floor.round(4).to_string(index=False))
    t, c, m = label_equal()
    print(t[["dataset", "encoding", "reference", "n", "mean_a", "mean_b", "median_diff", "ci_low", "ci_high", "p",
             "p_holm"]].round(4).to_string(index=False))
    print(c.round(3).to_string(index=False))
    print(m.round(3))


if __name__ == "__main__":
    main()
