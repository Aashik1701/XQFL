"""Pilot checks for the IEEE Access review, run on saved RQ4 target models.

No training happens here. Each check reuses the final global models saved in
results/rq4_release_repair_v1, and writes per-model rows to results/review_pilots.
These are diagnostics to decide what the rerun must change, not paper results.

Run: xqfl-env/bin/python analysis/review_pilot_checks.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import load_breast_cancer_data, load_pima_data
from explain.metrics import js_divergence
from explain.q_shap import exact_q_shap_values, make_quantum_predict_fn, q_shap_values
from federation.run_federation import _partition_data
from models import encodings
from models.quantum_classifier import HybridQuantumClassifier

REPAIR = ROOT / "results/rq4_release_repair_v1"
OUT = ROOT / "results/review_pilots"
SEEDS = [1, 7, 42, 123, 999, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47]
ENCODINGS = ["angle", "amplitude", "reuploading", "iqp"]
TARGETS = [("wisconsin", e) for e in ENCODINGS] + [("pima", "amplitude")]
REGIMES = {"iid": ("iid", 0.5), "alpha_0.5": ("dirichlet", 0.5),
           "alpha_0.1": ("dirichlet", 0.1), "qty_skew": ("quantity", 0.5)}


def tau(a, b):
    value, _ = kendalltau(a, b)
    return float(value)


def load_target(dataset, encoding, seed):
    loader = load_breast_cancer_data if dataset == "wisconsin" else load_pima_data
    train, test, ytrain, ytest = loader(random_state=seed)
    with (REPAIR / f"{dataset}_{encoding}_iid_{seed}" / "target.pkl").open("rb") as f:
        saved = pickle.load(f)  # this project's own saved artifact
    model = HybridQuantumClassifier(
        n_qubits=4, n_layers=2, encoding_fn=getattr(encodings, f"{encoding}_encoding"))
    model.pca, model._scaler = saved["pca"], saved["scaler"]
    model.set_weights(saved["weights"])
    return model, train, test, ytrain, ytest


def check_model(dataset, encoding, seed):
    model, train, test, ytrain, ytest = load_target(dataset, encoding, seed)
    predict = make_quantum_predict_fn(model)
    X_expl = model._preprocess(test[:10])
    X_bg = model._preprocess(train[:20])

    # 1. Estimator error under the global-explanation protocol.
    exact = exact_q_shap_values(predict, X_expl, X_bg)
    phi_exact = exact.mean(axis=0)
    sampled = [q_shap_values(predict, X_expl, X_bg, n_permutations=8, seed=s).mean(axis=0)
               for s in (42, 1042)]
    accuracy = float((model.predict(test) == ytest).mean())
    majority = float(np.bincount(ytest.astype(int)).max() / len(ytest))
    estimator = dict(
        dataset=dataset, encoding=encoding, seed=seed, accuracy=accuracy,
        track_a=accuracy > majority + 0.02,
        jsd_m8_vs_exact=js_divergence(sampled[0], phi_exact),
        jsd_m8_vs_m8=js_divergence(sampled[0], sampled[1]),
        tau_m8_vs_exact=tau(sampled[0], phi_exact),
        norm_mean_phi=float(np.linalg.norm(phi_exact)),
        mean_norm_phi=float(np.linalg.norm(exact, axis=1).mean()),
        tau_signed_vs_magnitude=tau(phi_exact, np.abs(phi_exact)),
    )

    # 2. Local-to-global distance with zero model disagreement. The SAME model
    # plays both roles, so any distance comes from the explained records and
    # the background, never from a difference between models.
    floor_rows = []
    importance_g = np.abs(exact).mean(axis=0)
    for regime, (split, alpha) in REGIMES.items():
        parts = _partition_data(train, ytrain, 4, split, seed, dirichlet_alpha=alpha)
        signed, magnitude, class_share = [], [], []
        for X_k, y_k in parts:
            if len(X_k) == 0:
                continue
            X_q = model._preprocess(X_k)
            local = exact_q_shap_values(predict, X_q[:5], X_q[:10])
            signed.append(js_divergence(local.mean(axis=0), phi_exact))
            magnitude.append(js_divergence(np.abs(local).mean(axis=0), importance_g))
            class_share.append(float(y_k[:5].mean()))
        floor_rows.append(dict(
            dataset=dataset, encoding=encoding, seed=seed, regime=regime,
            track_a=estimator["track_a"], clients_used=len(signed),
            floor_jsd_signed_mean=float(np.mean(signed)),
            floor_jsd_abs_mean=float(np.mean(magnitude)),
            mean_positive_share_explained=float(np.mean(class_share)),
        ))

    # 3. Positive control: is there any membership signal in the model outputs?
    p_train, p_test = model.predict_proba(train), model.predict_proba(test)
    loss = lambda p, y: -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    scores = -np.r_[loss(p_train, ytrain), loss(p_test, ytest)]
    labels = np.r_[np.ones(len(train)), np.zeros(len(test))]
    control = dict(
        dataset=dataset, encoding=encoding, seed=seed,
        train_acc=float(((p_train >= 0.5) == ytrain).mean()),
        test_acc=float(((p_test >= 0.5) == ytest).mean()),
        loss_threshold_auc=float(roc_auc_score(labels, scores)),
    )
    return estimator, floor_rows, control


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    estimator, floor, control = [], [], []
    for dataset, encoding in TARGETS:
        for seed in SEEDS:
            e, f, c = check_model(dataset, encoding, seed)
            estimator.append(e)
            floor.extend(f)
            control.append(c)
            print(dataset, encoding, seed, f"acc={e['accuracy']:.3f}", flush=True)
        pd.DataFrame(estimator).to_csv(OUT / "estimator_error.csv", index=False)
        pd.DataFrame(floor).to_csv(OUT / "rq3_zero_disagreement_floor.csv", index=False)
        pd.DataFrame(control).to_csv(OUT / "mia_positive_control.csv", index=False)


if __name__ == "__main__":
    main()
