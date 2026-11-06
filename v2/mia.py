"""Exploratory RQ4: membership attacks and positive controls. Protocol section 8."""
import json
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from explain.q_shap import exact_q_shap_values, make_quantum_predict_fn
from privacy.release_evaluation import evaluate_release, utility
from v2 import config
from v2.data import _rng, make_data, stratified_pick
from v2.models import accuracy, build_model, load_flat, predict_probabilities
from v2.train import child_seed, initial_seed, local_train, seed_rngs, write_json

NOISE_RATIOS = (0.25, 0.5, 1.0, 2.0)
CONTROL_VALID_AUC = 0.60


def attack_features(phi):
    return np.hstack([phi, np.abs(phi), np.linalg.norm(phi, axis=1, keepdims=True)])


def sample_loss(probabilities, y):
    p = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def attack_model(model, X_member, y_member, X_nonmember, y_nonmember, background, seed):
    """Run the loss baseline and the explanation attacks on one trained model."""
    X = np.vstack([X_member, X_nonmember])
    y = np.r_[y_member, y_nonmember]
    labels = np.r_[np.ones(len(X_member), int), np.zeros(len(X_nonmember), int)]
    phi = exact_q_shap_values(make_quantum_predict_fn(model), X, background)
    probabilities = predict_probabilities(model, X)
    loss_auc = float(roc_auc_score(labels, -sample_loss(probabilities, y)))
    if np.allclose(phi, 0):
        # A constant model yields no usable features for a classifier.
        result = dict(explanation_auc=np.nan, explanation_auc_phi_only=np.nan,
                      attack_advantage=np.nan)
    else:
        full, _, _ = evaluate_release(attack_features(phi), labels, seed, n_features=9)
        plain, _, _ = evaluate_release(phi, labels, seed)
        result = dict(explanation_auc=full["auc_mean_fold"],
                      explanation_auc_phi_only=plain["auc_mean_fold"],
                      attack_advantage=full["attack_advantage"])
    result["loss_auc"] = loss_auc
    return result, phi, labels


def _attack_sets(data, seed, n):
    members, _ = stratified_pick(data.y_train, n, 0, _rng(seed, 5))
    nonmembers, _ = stratified_pick(data.y_test, n, 0, _rng(seed, 6))
    return members, nonmembers


def target_task(spec, out_dir):
    """Attack the final global model of one saved federated run."""
    folder = out_dir / spec.key
    done = folder / "mia.json"
    if done.exists():
        return "skipped"
    meta = json.loads((folder / "train.json").read_text())
    data = make_data(spec.dataset, spec.seed, spec.n_features, spec.zero_handling)
    model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))
    with np.load(folder / "weights.npz") as source:
        load_flat(model, source["global_weights"][-1])
    members, nonmembers = _attack_sets(data, spec.seed, config.N_MIA_RECORDS)
    result, phi, labels = attack_model(
        model, data.X_train[members], data.y_train[members], data.X_test[nonmembers],
        data.y_test[nonmembers], data.X_val, spec.seed)
    np.savez_compressed(folder / "mia_phi.npz", phi=phi.astype(np.float32), labels=labels,
                        y=np.r_[data.y_train[members], data.y_test[nonmembers]])
    result.update(spec=spec.as_dict(), val_acc=meta["val_acc"][-1], val_majority=meta["val_majority"],
                  train_acc=meta["train_acc_final"], test_acc=meta["test_acc"][-1])
    write_json(done, result)
    return "attacked"


def train_control(kind, data, seed):
    """Deliberately overfit target. Returns model and its training positions."""
    if kind == "mlp_control":
        n_records, lr, max_epochs = 100, 0.01, 500
        model = build_model("mlp_control", init_seed=initial_seed(seed))
    elif kind == "vqc_control":
        n_records, lr, max_epochs = 30, config.LEARNING_RATE, 200
        model = build_model("reuploading", n_layers=6, init_seed=initial_seed(seed))
    else:
        raise ValueError(kind)
    chosen, _ = stratified_pick(data.y_train, n_records, 0, _rng(seed, 7))
    seed_rngs(child_seed(seed, 0, 2000))
    X, y = data.X_train[chosen], data.y_train[chosen]
    for epoch in range(1, max_epochs + 1):
        local_train(model, X, y, 1, lr=lr)
        if kind == "mlp_control" and accuracy(model, X, y) == 1.0:
            break
    return model, chosen, epoch


def control_task(kind, seed, out_dir, dataset="pima"):
    folder = out_dir / "controls" / f"{kind}_{dataset}_s{seed}"
    done = folder / "mia.json"
    if done.exists():
        return "skipped"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = make_data(dataset, seed)
    model, chosen, epochs = train_control(kind, data, seed)
    n = min(config.N_MIA_RECORDS, len(chosen))
    members = chosen[stratified_pick(data.y_train[chosen], n, 0, _rng(seed, 8))[0]]
    nonmembers, _ = stratified_pick(data.y_test, n, 0, _rng(seed, 9))
    result, phi, labels = attack_model(
        model, data.X_train[members], data.y_train[members], data.X_test[nonmembers],
        data.y_test[nonmembers], data.X_val, seed)
    np.savez_compressed(folder / "mia_phi.npz", phi=phi.astype(np.float32), labels=labels)
    result.update(kind=kind, dataset=dataset, seed=seed, epochs=epochs,
                  train_acc=accuracy(model, data.X_train[chosen], data.y_train[chosen]),
                  test_acc=accuracy(model, data.X_test, data.y_test),
                  seconds=time.monotonic() - started)
    write_json(done, result)
    return "attacked"


def laplace_release(phi, ratio, rng):
    """Add Laplace noise with scale = ratio x the mean per-sample attribution norm."""
    scale = ratio * float(np.linalg.norm(phi, axis=1).mean())
    return phi + rng.laplace(0.0, scale, phi.shape), scale


def noise_sweep(phi_file, seed):
    """Run only on models where the attack found a signal (protocol section 8)."""
    with np.load(phi_file) as source:
        phi, labels = source["phi"].astype(float), source["labels"]
    rows = []
    for i, ratio in enumerate(NOISE_RATIOS):
        noisy, scale = laplace_release(phi, ratio, _rng(seed, 10, i))
        summary, _, _ = evaluate_release(attack_features(noisy), labels, seed, n_features=9)
        rows.append(dict(ratio=ratio, scale=scale, auc=summary["auc_mean_fold"], **utility(phi, noisy)))
    return rows


def bootstrap_mean_interval(values, n_boot=10_000, seed=20261105):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def interpret(control_valid, control_detects, targets_detect):
    """The protocol's reading table. Returns the conclusion the paper may draw."""
    if not control_valid:
        return "pipeline unvalidated; no conclusion"
    if targets_detect:
        return "leakage detected on main targets; report effect size and interval"
    if control_detects:
        return "attack can work; no signal detected on low-capacity targets; not proof of privacy"
    return "explanations carried no detectable signal even when loss did; attack-specific"
