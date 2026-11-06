"""Build every v2 table from per-run summaries. Protocol sections 5-7."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from explain.metrics import js_divergence
from v2 import config
from v2 import stats

FAMILY_ENCODINGS = config.ENCODINGS
NON_IID = ("qty_skew", "alpha_0.5", "alpha_0.1")
PAIRS = list(itertools.combinations(FAMILY_ENCODINGS, 2))
BASE = dict(n_features=4, n_layers=2, n_clients=4, n_rounds=15, zero_handling="missing")


def load_summaries(group_dir):
    rows, clients = [], []
    for path in sorted(Path(group_dir).glob("*/summary.json")):
        record = json.loads(path.read_text())
        row = dict(record.pop("spec"))
        client_rows = record.pop("clients", [])
        for block in ("stability_I", "stability_S"):
            for name, value in record.pop(block, {}).items():
                row[f"{block}_{name}"] = value
        row.update(record)
        rows.append(row)
        for entry in client_rows:
            if not entry.get("sparse"):
                clients.append({**{k: row[k] for k in ("dataset", "encoding", "regime", "seed",
                                                       *BASE)}, **entry})
    runs = pd.DataFrame(rows)
    return runs, pd.DataFrame(clients)


def passes_gate(frame, margin, column="val"):
    if margin is None:
        return pd.Series(True, index=frame.index)
    return frame[f"{column}_acc"] > frame[f"{column}_majority"] + margin


def fidelity_table(runs):
    """One row per federated run with its centralized pair."""
    fed = runs[runs.kind == "fed"]
    cent = runs[runs.kind == "cent"]
    keys = ["dataset", "encoding", "seed", *BASE]
    first = cent[cent.replicate == 1].set_index(keys)
    second = cent[cent.replicate == 2].set_index(keys)
    rows = []
    for _, run in fed.iterrows():
        key = tuple(run[k] for k in keys)
        if key not in first.index or key not in second.index:
            continue
        c1, c2 = first.loc[key], second.loc[key]
        row = {k: run[k] for k in keys + ["regime"]}
        row.update(val_acc=run.val_acc, val_majority=run.val_majority, test_acc=run.test_acc,
                   c1_val=c1.val_acc, c2_val=c2.val_acc, c_majority=c1.val_majority,
                   c1_test=c1.test_acc)
        for kind in ("I", "S"):
            distance = js_divergence(run[f"{kind}_final"], c1[f"{kind}_final"])
            floor = js_divergence(c1[f"{kind}_final"], c2[f"{kind}_final"])
            row.update({f"jsd_cent_{kind}": distance, f"floor_cent_{kind}": floor,
                        f"excess_{kind}": distance - floor})
        rows.append(row)
    return pd.DataFrame(rows)


def fidelity_functional(table, margin):
    """All three models in a fidelity comparison must pass the gate."""
    if margin is None:
        return pd.Series(True, index=table.index)
    limit = table.c_majority + margin
    return (table.val_acc > table.val_majority + margin) & (table.c1_val > limit) & (table.c2_val > limit)


def base_only(frame):
    mask = np.logical_and.reduce([frame[k] == v for k, v in BASE.items()])
    return frame[mask]


def cell_status(n):
    if n >= config.MIN_ESTIMATE_SEEDS:
        return "estimated"
    return "descriptive" if n >= config.MIN_DESCRIPTIVE_SEEDS else "coverage_only"


def summarize_cell(values):
    values = pd.Series(values).dropna()
    status = cell_status(len(values))
    if status == "coverage_only":
        return dict(n=len(values), status=status, mean=np.nan, sd=np.nan, median=np.nan)
    return dict(n=len(values), status=status, mean=values.mean(), sd=values.std(ddof=1),
                median=values.median())


def coverage_table(runs):
    fed = base_only(runs[runs.kind == "fed"])
    rows = []
    for key, group in fed.groupby(["dataset", "encoding", "regime"]):
        functional = int(passes_gate(group, config.GATE_MARGIN).sum())
        low, high = stats.wilson_interval(functional, len(group))
        rows.append(dict(zip(("dataset", "encoding", "regime"), key), runs=len(group),
                         functional=functional, rate=functional / len(group),
                         wilson_low=low, wilson_high=high,
                         gate_agrees_with_test=float(
                             (passes_gate(group, config.GATE_MARGIN)
                              == passes_gate(group, config.GATE_MARGIN, "test")).mean())))
    return pd.DataFrame(rows)


def _by_seed(frame, column):
    return frame.set_index("seed")[column]


def _paired(frame_a, frame_b, column_a, column_b=None):
    column_b = column_b or column_a
    joined = pd.concat([_by_seed(frame_a, column_a).rename("a"),
                        _by_seed(frame_b, column_b).rename("b")], axis=1, join="inner").dropna()
    return joined.a.to_numpy(), joined.b.to_numpy()


def _client_means(clients):
    return clients.groupby(["dataset", "encoding", "regime", "seed"], as_index=False).mean(
        numeric_only=True)


def family_tests(runs, clients, margin, kind="I", n_boot=stats.BOOTSTRAPS):
    """All pre-specified families at one gate margin. kind is I (primary) or S."""
    rows = []
    fed = base_only(runs[runs.kind == "fed"]).copy()
    fed = fed[fed.encoding.isin(FAMILY_ENCODINGS)]
    fed["functional"] = passes_gate(fed, margin)
    table = fidelity_table(runs)
    table = base_only(table) if len(table) else table

    def add(family, label, result):
        rows.append(dict(family=family, contrast=label, margin=margin, metric=kind, **result))

    # Coverage is always judged at the primary gate; it is not conditional on itself.
    if kind == "I" and margin == config.GATE_MARGIN:
        for dataset, regime in itertools.product(config.DATASETS, ("iid", "alpha_0.5")):
            for a, b in PAIRS:
                frame = fed[(fed.dataset == dataset) & (fed.regime == regime)]
                x, y = _paired(frame[frame.encoding == a], frame[frame.encoding == b], "functional")
                if len(x):
                    add("coverage", f"{dataset}/{regime}: {a} vs {b}",
                        stats.mcnemar_exact(x.astype(bool), y.astype(bool)))

    if len(table):
        table = table.copy()
        table["functional"] = fidelity_functional(table, margin)
        good = table[table.functional & (table.regime == "iid")]
        for dataset in config.DATASETS:
            frame = good[good.dataset == dataset]
            for encoding in FAMILY_ENCODINGS:
                values = frame[frame.encoding == encoding][f"excess_{kind}"].to_numpy()
                add("rq1_excess", f"{dataset}: {encoding} excess vs 0",
                    stats.paired_test(values, n_boot=n_boot))
            for a, b in PAIRS:
                x, y = _paired(frame[frame.encoding == a], frame[frame.encoding == b],
                               f"jsd_cent_{kind}")
                add("rq1_pairwise", f"{dataset}: {a} vs {b}", stats.paired_test(x, y, n_boot=n_boot))

    good = fed[fed.functional & (fed.regime == "iid")]
    for dataset in config.DATASETS:
        frame = good[good.dataset == dataset]
        for a, b in PAIRS:
            x, y = _paired(frame[frame.encoding == a], frame[frame.encoding == b],
                           f"stability_{kind}_tau")
            add("rq2", f"{dataset}: {a} vs {b}", stats.paired_test(x, y, n_boot=n_boot))

    if len(clients):
        base = base_only(clients)
        status = fed.set_index(["dataset", "encoding", "regime", "seed"]).functional
        keys = list(zip(base.dataset, base.encoding, base.regime, base.seed))
        base = base[status.reindex(keys).fillna(False).to_numpy().astype(bool)]
        means = _client_means(base)
        column = f"D_model_{kind}"

        def client_arrays(dataset, encoding, regime):
            part = base[(base.dataset == dataset) & (base.encoding == encoding) & (base.regime == regime)]
            return {seed: g[column].to_numpy() for seed, g in part.groupby("seed")}

        def select(dataset, encoding, regime):
            return means[(means.dataset == dataset) & (means.encoding == encoding) & (means.regime == regime)]

        for dataset in config.DATASETS:
            for encoding in FAMILY_ENCODINGS:
                reference = select(dataset, encoding, "iid")
                for regime in NON_IID:
                    x, y = _paired(select(dataset, encoding, regime), reference, column)
                    result = stats.paired_test(x, y, n_boot=n_boot)
                    if result["status"] == "tested" and n_boot >= 1000:
                        result["ci_low"], result["ci_high"] = stats.hierarchical_interval(
                            client_arrays(dataset, encoding, regime),
                            client_arrays(dataset, encoding, "iid"), n_boot=n_boot)
                    add("rq3_regime", f"{dataset}/{encoding}: {regime} vs iid", result)
                sample = fed[(fed.functional) & (fed.dataset == dataset) & (fed.encoding == encoding)
                             & (fed.regime == "iid")].set_index("seed")[f"D_sample_{kind}"]
                model = reference.set_index("seed")[column]
                joined = pd.concat([model.rename("a"), sample.rename("b")], axis=1, join="inner").dropna()
                add("rq3_floor", f"{dataset}/{encoding}: D_model vs D_sample",
                    stats.paired_test(joined.a.to_numpy(), joined.b.to_numpy(), n_boot=n_boot))

    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("family").p.transform(lambda p: stats.holm(p.to_numpy()))
    return out


def rq_tables(runs, clients):
    """Descriptive per-cell tables at the primary gate."""
    fed = base_only(runs[runs.kind == "fed"]).copy()
    fed["functional"] = passes_gate(fed, config.GATE_MARGIN)
    cells = ["dataset", "encoding", "regime"]
    stability, divergence, fidelity = [], [], []

    good = fed[fed.functional]
    for key, group in good.groupby(cells):
        row = dict(zip(cells, key))
        for name in ("I_tau", "S_tau", "I_tau_init", "I_cosine", "I_top_change_rate", "I_settle_round"):
            cell = summarize_cell(group[f"stability_{name}"])
            row.update({f"{name}_{k}": v for k, v in cell.items()})
        stability.append(row)

    if len(clients):
        base = base_only(clients)
        status = fed.set_index(cells + ["seed"]).functional
        keys = list(zip(base.dataset, base.encoding, base.regime, base.seed))
        base = base[status.reindex(keys).fillna(False).to_numpy().astype(bool)]
        means = _client_means(base)
        sample = good.set_index(cells + ["seed"])
        for key, group in means.groupby(cells):
            row = dict(zip(cells, key))
            seeds = sample.loc[key] if key in sample.index.droplevel(3) else None
            for name in ("D_model", "D_pop", "D_floor"):
                for kind in ("I", "S"):
                    cell = summarize_cell(group[f"{name}_{kind}"])
                    row.update({f"{name}_{kind}_{k}": v for k, v in cell.items()})
            if seeds is not None:
                for kind in ("I", "S"):
                    cell = summarize_cell(seeds[f"D_sample_{kind}"])
                    row.update({f"D_sample_{kind}_{k}": v for k, v in cell.items()})
            divergence.append(row)

    table = fidelity_table(runs)
    if len(table):
        table = base_only(table).copy()
        table = table[fidelity_functional(table, config.GATE_MARGIN)]
        for key, group in table.groupby(cells):
            row = dict(zip(cells, key))
            for name in ("jsd_cent_I", "floor_cent_I", "excess_I", "jsd_cent_S", "floor_cent_S"):
                cell = summarize_cell(group[name])
                row.update({f"{name}_{k}": v for k, v in cell.items()})
            fidelity.append(row)
    return pd.DataFrame(fidelity), pd.DataFrame(stability), pd.DataFrame(divergence)


def build_tables(group_dir, out_dir, n_boot=stats.BOOTSTRAPS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, clients = load_summaries(group_dir)
    coverage_table(runs).to_csv(out_dir / "coverage.csv", index=False)
    for name, table in zip(("rq1_fidelity", "rq2_stability", "rq3_divergence"), rq_tables(runs, clients)):
        table.to_csv(out_dir / f"{name}.csv", index=False)
    family_tests(runs, clients, config.GATE_MARGIN, "I", n_boot).to_csv(out_dir / "tests_primary.csv", index=False)
    family_tests(runs, clients, config.GATE_MARGIN, "S", n_boot).to_csv(out_dir / "tests_signed.csv", index=False)
    frames = [family_tests(runs, clients, margin, "I", min(n_boot, 2000))
              for margin in config.SENSITIVITY_MARGINS]
    pd.concat(frames).to_csv(out_dir / "tests_gate_sensitivity.csv", index=False)
    runs.drop(columns=["I_final", "S_final"]).to_csv(out_dir / "runs.csv", index=False)
    clients.to_csv(out_dir / "clients.csv", index=False)
    return sorted(p.name for p in out_dir.glob("*.csv"))


def zero_handling_sensitivity(runs, out_dir):
    """Pima: missing-value treatment vs v1 zero handling, paired by seed."""
    table = fidelity_table(runs)
    rows = []
    fed = runs[(runs.kind == "fed") & (runs.dataset == "pima") & (runs.regime == "iid")]
    for encoding in ("amplitude", "reuploading"):
        chosen = fed[fed.encoding == encoding]
        missing = chosen[chosen.zero_handling == "missing"]
        as_values = chosen[chosen.zero_handling == "as_values"]
        for column in ("test_acc", "stability_I_tau"):
            x, y = _paired(missing, as_values, column)
            rows.append(dict(encoding=encoding, metric=column, **stats.paired_test(x, y)))
        rows.append(dict(encoding=encoding, metric="functional_rate_missing_minus_as_values",
                         median_diff=passes_gate(missing, config.GATE_MARGIN).mean()
                         - passes_gate(as_values, config.GATE_MARGIN).mean(),
                         n=len(missing), status="descriptive"))
        if len(table):
            part = table[(table.dataset == "pima") & (table.encoding == encoding) & (table.regime == "iid")]
            x, y = _paired(part[part.zero_handling == "missing"], part[part.zero_handling == "as_values"],
                           "jsd_cent_I")
            rows.append(dict(encoding=encoding, metric="jsd_cent_I", **stats.paired_test(x, y)))
    pd.DataFrame(rows).to_csv(Path(out_dir) / "pima_zero_sensitivity.csv", index=False)


def classical_comparison(runs, out_dir):
    """Exploratory quantum vs classical contrasts. No claim of quantum advantage."""
    fed = base_only(runs[(runs.kind == "fed") & (runs.regime == "iid")])
    table = fidelity_table(runs)
    table = base_only(table[table.regime == "iid"]) if len(table) else table
    rows = []
    for dataset in config.DATASETS:
        for encoding in config.ENCODINGS + (config.VARIANT,):
            for control in config.CLASSICAL:
                select = lambda frame, name: frame[(frame.dataset == dataset) & (frame.encoding == name)]
                for metric, frame in (("test_acc", fed), ("stability_I_tau", fed), ("jsd_cent_I", table)):
                    if not len(frame):
                        continue
                    x, y = _paired(select(frame, encoding), select(frame, control), metric)
                    rows.append(dict(dataset=dataset, encoding=encoding, control=control, metric=metric,
                                     exploratory=True, **stats.paired_test(x, y)))
    pd.DataFrame(rows).to_csv(Path(out_dir) / "classical_exploratory.csv", index=False)


def estimator_table(group_dir, out_dir):
    rows = []
    for path in Path(group_dir).glob("*/estimators.json"):
        record = json.loads(path.read_text())
        for row in record["rows"]:
            rows.append({**{k: record["spec"][k] for k in ("dataset", "encoding", "seed")}, **row})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(out_dir) / "estimators_per_model.csv", index=False)
    summary = frame.groupby(["dataset", "encoding", "comparison"]).agg(
        n=("seed", "size"), jsd_I_mean=("jsd_I", "mean"), jsd_I_max=("jsd_I", "max"),
        tau_I_mean=("tau_I", "mean"), tau_I_min=("tau_I", "min"), jsd_S_mean=("jsd_S", "mean"),
        cosine_S_mean=("cosine_S", "mean"), mean_abs_error=("mean_abs_error", "mean"))
    summary.reset_index().to_csv(Path(out_dir) / "estimators.csv", index=False)


def privacy_tables(group_dir, out_dir):
    """RQ4 cells and the protocol's reading table. Exploratory throughout."""
    from v2.mia import CONTROL_VALID_AUC, bootstrap_mean_interval, interpret

    out_dir = Path(out_dir)
    targets = []
    for path in Path(group_dir).glob("fed_*/mia.json"):
        record = json.loads(path.read_text())
        targets.append({**record.pop("spec"), **record})
    controls = [json.loads(p.read_text()) for p in Path(group_dir).glob("controls/*/mia.json")]
    if not targets:
        return

    def describe(frame):
        out = dict(n=len(frame))
        for name in ("loss_auc", "explanation_auc", "explanation_auc_phi_only"):
            low, high = bootstrap_mean_interval(frame[name])
            out.update({f"{name}_mean": frame[name].mean(), f"{name}_sd": frame[name].std(ddof=1),
                        f"{name}_low": low, f"{name}_high": high})
        out["detects"] = bool(out["explanation_auc_low"] > 0.5)
        return out

    frame = base_only(pd.DataFrame(targets))
    frame["functional"] = passes_gate(frame, config.GATE_MARGIN)
    rows = []
    for key, group in frame.groupby(["dataset", "encoding", "regime"]):
        cell = dict(zip(("dataset", "encoding", "regime"), key))
        rows.append({**cell, "population": "all", **describe(group)})
        rows.append({**cell, "population": "functional", **describe(group[group.functional])})
    cells = pd.DataFrame(rows)
    cells.to_csv(out_dir / "rq4_targets.csv", index=False)
    detecting = cells[(cells.population == "all") & cells.detects]
    readings = []
    control_rows = []
    if controls:
        for kind, group in pd.DataFrame(controls).groupby("kind"):
            summary = describe(group)
            valid = bool(summary["loss_auc_low"] > CONTROL_VALID_AUC)
            control_rows.append(dict(kind=kind, valid=valid, train_acc=group.train_acc.mean(),
                                     test_acc=group.test_acc.mean(), **summary))
            readings.append(dict(control=kind, valid=valid, control_detects=summary["detects"],
                                 target_cells_detecting=len(detecting), target_cells=int((cells.population == "all").sum()),
                                 reading=interpret(valid, summary["detects"], len(detecting) > 0)))
        pd.DataFrame(control_rows).to_csv(out_dir / "rq4_controls.csv", index=False)
        pd.DataFrame(readings).to_csv(out_dir / "rq4_reading.csv", index=False)


def sweep_table(group_dir, out_dir):
    """Descriptive sweep summary; these are not tested (protocol section 9)."""
    runs, clients = load_summaries(group_dir)
    if not len(runs):
        return
    fed = runs[runs.kind == "fed"].copy()
    fed["functional"] = passes_gate(fed, config.GATE_MARGIN)
    factors = ["dataset", "encoding", "regime", "n_features", "n_layers", "n_clients", "n_rounds"]
    rows = []
    for key, group in fed.groupby(factors):
        good = group[group.functional]
        row = dict(zip(factors, key), runs=len(group), functional=len(good))
        row.update({f"tau_{k}": v for k, v in summarize_cell(good.stability_I_tau).items()})
        row["test_acc_mean"] = group.test_acc.mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(Path(out_dir) / "sweep_summary.csv", index=False)


def build_all(group_dir, out_dir, n_boot=stats.BOOTSTRAPS):
    names = build_tables(group_dir, out_dir, n_boot)
    runs, _ = load_summaries(group_dir)
    zero_handling_sensitivity(runs, out_dir)
    classical_comparison(runs, out_dir)
    estimator_table(group_dir, out_dir)
    privacy_tables(group_dir, out_dir)
    return sorted(p.name for p in Path(out_dir).glob("*.csv"))
