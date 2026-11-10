"""Post-hoc RQ4 checks on saved attributions (protocol v2, section 11).

Chosen after the primary results were seen, so everything here is exploratory.
Trains nothing and computes no new attributions.

Run from the repository root:
    PYTHONPATH=. xqfl-env/bin/python analysis/rq4_posthoc_checks.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from privacy.release_evaluation import classifier, evaluate_release
from v2 import config
from v2.data import _rng, make_data, stratified_pick
from v2.mia import attack_features, bootstrap_mean_interval

ROOT = Path("results/protocol_v2/primary")
OUT = Path("results/protocol_v2/posthoc")
SEED = 20261106
N_SHUFFLES = 40
ENCODINGS = ("amplitude", "angle", "iqp", "reuploading")


def cv_auc(X, labels, strata, seed):
    """Mean fold AUC of the protocol's classifier, with a chosen fold stratification."""
    scores = []
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    for train, test in folds.split(X, strata):
        model = classifier(seed).fit(X[train], labels[train])
        scores.append(roc_auc_score(labels[test], model.predict_proba(X[test])[:, 1]))
    return float(np.mean(scores))


def shuffle_within_class(labels, y, rng):
    """Shuffle membership inside each outcome class, keeping the sets' class balance."""
    shuffled = labels.copy()
    for value in np.unique(y):
        where = np.where(y == value)[0]
        shuffled[where] = rng.permutation(labels[where])
    return shuffled


def control_outcomes(meta):
    """Rebuild a control's attack-record outcomes; these were not saved with its attributions."""
    kind, seed = meta["kind"], meta["seed"]
    data = make_data(meta["dataset"], seed)
    n_records = 100 if kind == "mlp_control" else 30
    chosen, _ = stratified_pick(data.y_train, n_records, 0, _rng(seed, 7))
    n = min(config.N_MIA_RECORDS, len(chosen))
    members = chosen[stratified_pick(data.y_train[chosen], n, 0, _rng(seed, 8))[0]]
    nonmembers, _ = stratified_pick(data.y_test, n, 0, _rng(seed, 9))
    return np.r_[data.y_train[members], data.y_test[nonmembers]]


def load_records():
    """Yield group, metadata, attributions, membership labels and outcomes per model."""
    for path in sorted(ROOT.glob("*/mia_phi.npz")):
        meta = json.loads((path.parent / "mia.json").read_text())
        with np.load(path) as saved:
            yield "target", meta, saved["phi"].astype(float), saved["labels"], saved["y"]
    for path in sorted((ROOT / "controls").glob("*/mia_phi.npz")):
        meta = json.loads((path.parent / "mia.json").read_text())
        with np.load(path) as saved:
            phi, labels = saved["phi"].astype(float), saved["labels"]
        outcomes = control_outcomes(meta)
        if len(outcomes) != len(labels):
            raise ValueError(f"Rebuilt outcomes do not match {path}")
        yield meta["kind"], meta, phi, labels, outcomes


def gaussian_null():
    rng = np.random.default_rng(SEED)
    rows = []
    for n, k in ((50, 9), (50, 4), (30, 9)):
        labels = np.r_[np.ones(n, int), np.zeros(n, int)]
        aucs = [evaluate_release(rng.normal(size=(2 * n, k)), labels, r, n_features=k)[0]["auc_mean_fold"]
                for r in range(400)]
        rows.append(dict(per_class=n, features=k, reps=400, auc_mean=np.mean(aucs),
                         auc_se=np.std(aucs) / np.sqrt(400)))
    return pd.DataFrame(rows)


def per_model_table():
    rows = []
    for group, meta, phi, labels, y in load_records():
        spec = meta.get("spec", {})
        seed = spec.get("seed", meta.get("seed"))
        row = dict(group=group, dataset=spec.get("dataset", meta.get("dataset")),
                   encoding=spec.get("encoding", group), regime=spec.get("regime", ""),
                   zero_handling=spec.get("zero_handling", ""), seed=seed,
                   loss_auc=meta["loss_auc"], explanation_auc=meta["explanation_auc"],
                   label_aware_auc=np.nan, joint_stratified_auc=np.nan)
        if not np.allclose(phi, 0):
            # Efficiency: the attributions sum to f(x) - f(reference), so with the label
            # this score orders records by confidence on the correct class.
            row["label_aware_auc"] = roc_auc_score(labels, (2 * y - 1) * phi.sum(axis=1))
            row["joint_stratified_auc"] = cv_auc(attack_features(phi), labels, labels * 2 + y, seed)
        rows.append(row)
    return pd.DataFrame(rows)


def shuffle_null():
    """Observed AUC against repeated within-class shuffles, IID models only."""
    rows = []
    for group, meta, phi, labels, y in load_records():
        if group != "target" or np.allclose(phi, 0):
            continue
        spec = meta["spec"]
        if spec["regime"] != "iid" or spec["encoding"] not in ENCODINGS or spec["zero_handling"] != "missing":
            continue
        if spec["replicate"] != 1:
            continue
        rng = np.random.default_rng([spec["seed"], 11])
        X = attack_features(phi)
        null = []
        for _ in range(N_SHUFFLES):
            shuffled = shuffle_within_class(labels, y, rng)
            null.append(cv_auc(X, shuffled, shuffled, spec["seed"]))
        rows.append(dict(dataset=spec["dataset"], encoding=spec["encoding"], seed=spec["seed"],
                         observed=meta["explanation_auc"], null_mean=np.mean(null), null_sd=np.std(null),
                         percentile=np.mean(np.array(null) <= meta["explanation_auc"])))
    return pd.DataFrame(rows)


def seed_level_summary(models, null):
    """Intervals across dataset-seed pairs, because cells reuse records within a seed."""
    rows = []
    for name in ("mlp_control", "vqc_control", "target"):
        frame = models[models.group == name]
        if name == "target":
            frame = frame.groupby(["dataset", "seed"], as_index=False)[
                ["loss_auc", "explanation_auc", "label_aware_auc", "joint_stratified_auc"]].mean()
        row = dict(group=name, independent_units=len(frame))
        for metric in ("loss_auc", "explanation_auc", "label_aware_auc", "joint_stratified_auc"):
            low, high = bootstrap_mean_interval(frame[metric])
            row.update({metric: frame[metric].mean(), f"{metric}_low": low, f"{metric}_high": high})
        difference = frame.label_aware_auc - frame.loss_auc
        low, high = bootstrap_mean_interval(difference)
        row.update(label_aware_minus_loss=difference.mean(), label_aware_minus_loss_low=low,
                   label_aware_minus_loss_high=high)
        rows.append(row)
    pairs = null.groupby(["dataset", "seed"])[["observed", "null_mean"]].mean().reset_index()
    low, high = bootstrap_mean_interval(pairs.observed - pairs.null_mean)
    shuffle = dict(units=len(pairs), observed=pairs.observed.mean(), null=pairs.null_mean.mean(),
                   difference=(pairs.observed - pairs.null_mean).mean(), low=low, high=high,
                   mean_percentile=null.percentile.mean())
    return pd.DataFrame(rows), shuffle


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    gaussian = gaussian_null()
    models = per_model_table()
    null = shuffle_null()
    summary, shuffle = seed_level_summary(models, null)
    gaussian.to_csv(OUT / "gaussian_null.csv", index=False)
    models.to_csv(OUT / "per_model.csv", index=False)
    null.to_csv(OUT / "shuffle_null.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "shuffle_summary.json").write_text(json.dumps(shuffle, indent=1))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(gaussian.round(4).to_string(index=False))
    print(summary.round(3).T.to_string())
    print(json.dumps({key: round(float(value), 4) for key, value in shuffle.items()}))
    print("models with no attribution signal:", int(models.label_aware_auc.isna().sum()))


if __name__ == "__main__":
    main()
