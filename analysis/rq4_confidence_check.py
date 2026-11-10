"""Post-hoc RQ4 check: exact correct-class confidence reconstructed from attributions.

By efficiency, sum_i phi_i(x) = f(x) - f(r), where r is the background mean (here the validation split).
The earlier post-hoc score (2y - 1) * sum_i phi_i(x) omits the class-dependent offset. The exact
correct-class confidence is
    C(x, y) = y * (f(r) + sum phi) + (1 - y) * (1 - f(r) - sum phi),
which is a monotone function of the per-record loss, so its AUC should equal the loss attack's.
f(r) is one model evaluation. Target weights are saved; control weights are not, so the controls are
retrained with their fixed seeds and checked against the saved attributions before use.

    PYTHONPATH=. OMP_NUM_THREADS=1 xqfl-env/bin/python analysis/rq4_confidence_check.py
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from v2 import config
from v2.data import _rng, make_data, stratified_pick
from v2.mia import sample_loss, train_control
from v2.models import build_model, load_flat, predict_probabilities
from v2.train import initial_seed

ROOT = Path("results/protocol_v2/primary")
OUT = Path("results/protocol_v2/posthoc")


def scores(phi, y, f_ref):
    total = phi.sum(axis=1)
    old = (2 * y - 1) * total
    confidence = y * (f_ref + total) + (1 - y) * (1 - f_ref - total)
    return old, confidence


def target_row(folder):
    torch.set_num_threads(1)
    meta = json.loads((folder / "mia.json").read_text())
    spec = meta["spec"]
    with np.load(folder / "mia_phi.npz") as saved:
        phi, labels, y = saved["phi"].astype(float), saved["labels"], saved["y"]
    data = make_data(spec["dataset"], spec["seed"], spec["n_features"], spec["zero_handling"])
    model = build_model(spec["encoding"], spec["n_features"], spec["n_layers"], initial_seed(spec["seed"]))
    with np.load(folder / "weights.npz") as source:
        load_flat(model, source["global_weights"][-1])
    f_ref = float(predict_probabilities(model, data.X_val.mean(axis=0, keepdims=True))[0])
    row = dict(group="target", dataset=spec["dataset"], encoding=spec["encoding"], regime=spec["regime"],
               seed=spec["seed"], loss_auc=meta["loss_auc"], f_ref=f_ref)
    if np.allclose(phi, 0):
        return dict(row, old_auc=np.nan, confidence_auc=np.nan)
    old, confidence = scores(phi, y, f_ref)
    return dict(row, old_auc=roc_auc_score(labels, old), confidence_auc=roc_auc_score(labels, confidence))


def control_row(args):
    kind, seed = args
    torch.set_num_threads(1)
    folder = ROOT / "controls" / f"{kind}_pima_s{seed}"
    meta = json.loads((folder / "mia.json").read_text())
    with np.load(folder / "mia_phi.npz") as saved:
        phi, labels = saved["phi"].astype(float), saved["labels"]
    data = make_data("pima", seed)
    model, chosen, _ = train_control(kind, data, seed)
    n = min(config.N_MIA_RECORDS, len(chosen))
    members = chosen[stratified_pick(data.y_train[chosen], n, 0, _rng(seed, 8))[0]]
    nonmembers, _ = stratified_pick(data.y_test, n, 0, _rng(seed, 9))
    X = np.vstack([data.X_train[members], data.X_test[nonmembers]])
    y = np.r_[data.y_train[members], data.y_test[nonmembers]]
    probabilities = predict_probabilities(model, X)
    f_ref = float(predict_probabilities(model, data.X_val.mean(axis=0, keepdims=True))[0])
    # The retrained model must reproduce the saved attributions through efficiency and the saved loss AUC.
    efficiency_error = float(np.abs(phi.sum(axis=1) - (probabilities - f_ref)).max())
    loss_auc = float(roc_auc_score(labels, -sample_loss(probabilities, y)))
    old, confidence = scores(phi, y, f_ref)
    return dict(group=kind, dataset="pima", encoding=kind, regime="-", seed=seed, loss_auc=meta["loss_auc"],
                retrained_loss_auc=loss_auc, efficiency_error=efficiency_error, f_ref=f_ref,
                old_auc=roc_auc_score(labels, old), confidence_auc=roc_auc_score(labels, confidence))


def main():
    targets = sorted(p.parent for p in ROOT.glob("fed_*/mia.json"))
    controls = [(k, s) for s in config.SEEDS for k in ("mlp_control", "vqc_control")]
    with Pool(7) as pool:
        rows = pool.map(target_row, targets) + pool.map(control_row, controls)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "confidence_per_model.csv", index=False)
    summary = frame.groupby("group")[["loss_auc", "old_auc", "confidence_auc"]].mean()
    summary["max_abs_confidence_minus_loss"] = frame.assign(
        d=(frame.confidence_auc - frame.loss_auc).abs()).groupby("group").d.max()
    print(summary.round(4))
    controls_frame = frame[frame.group != "target"]
    print("controls: max efficiency error", controls_frame.efficiency_error.max(),
          "max |retrained - saved loss AUC|", (controls_frame.retrained_loss_auc - controls_frame.loss_auc).abs().max())
    print(interval_summary().round(3).to_string(index=False))


if __name__ == "__main__":
    main()


def interval_summary():
    """Seed-level means and intervals, with the same bootstrap as analysis/rq4_posthoc_checks.py."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rq4_posthoc_checks import bootstrap_mean_interval
    frame = pd.read_csv(OUT / "confidence_per_model.csv")
    rows = []
    for name in ("mlp_control", "vqc_control", "target"):
        part = frame[frame.group == name].dropna(subset=["confidence_auc"])
        if name == "target":
            part = part.groupby(["dataset", "seed"], as_index=False)[["loss_auc", "old_auc", "confidence_auc"]].mean()
        row = dict(group=name, units=len(part))
        for metric in ("loss_auc", "old_auc", "confidence_auc"):
            low, high = bootstrap_mean_interval(part[metric])
            row.update({metric: part[metric].mean(), f"{metric}_low": low, f"{metric}_high": high})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "confidence_summary.csv", index=False)
    return out
