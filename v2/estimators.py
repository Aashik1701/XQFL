"""Estimator validation on final models. Protocol section 4."""
import json

import numpy as np

from explain.metrics import js_divergence
from explain.q_shap import exact_q_shap_values, make_quantum_predict_fn, q_shap_values
from scipy.stats import kendalltau
from v2.data import _rng, explanation_sets, make_data
from v2.models import build_model, load_flat
from v2.train import initial_seed, write_json


def _compare(reference, other):
    """Compare per-record matrices via both summary vectors."""
    I_ref, I_other = np.abs(reference).mean(axis=0), np.abs(other).mean(axis=0)
    S_ref, S_other = reference.mean(axis=0), other.mean(axis=0)
    denominator = np.linalg.norm(S_ref) * np.linalg.norm(S_other)
    return dict(jsd_I=js_divergence(I_other, I_ref), tau_I=float(kendalltau(I_other, I_ref).statistic),
                jsd_S=js_divergence(S_other, S_ref),
                cosine_S=float(S_ref @ S_other / denominator) if denominator > 1e-15 else np.nan,
                mean_abs_error=float(np.abs(reference - other).mean()))


def marginal_expectation_shap(predict, X, background):
    """Exact Shapley where absent features are averaged over background rows."""
    n, d = X.shape
    phi = np.zeros((n, d))
    for row in background:
        phi += exact_q_shap_values(predict, X, row[None, :])
    return phi / len(background)


def class_balanced_background(X, y):
    return np.stack([X[y == c].mean(axis=0) for c in np.unique(y)])


def kernel_shap(predict, X, reference):
    import shap

    explainer = shap.KernelExplainer(predict, reference.mean(axis=0, keepdims=True))
    return np.asarray(explainer.shap_values(X, silent=True, nsamples="auto"))


def validate_task(spec, out_dir, extended=False):
    """Sampled vs exact for one model; with extended, also Kernel and other references."""
    folder = out_dir / spec.key
    done = folder / "estimators.json"
    if done.exists():
        return "skipped"
    data = make_data(spec.dataset, spec.seed, spec.n_features, spec.zero_handling)
    model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))
    with np.load(folder / "weights.npz") as source:
        load_flat(model, source["global_weights"][-1])
    sets = explanation_sets(data, [], spec.seed)
    X, B = data.X_test[sets.E_global], data.X_train[sets.B_global]
    predict = make_quantum_predict_fn(model)
    exact = exact_q_shap_values(predict, X, B)
    rows = []
    for M in (8, 24, 64):
        sampled = q_shap_values(predict, X, B, n_permutations=M, seed=spec.seed)
        rows.append(dict(comparison=f"sampled_M{M}_vs_exact", **_compare(exact, sampled)))
    if extended:
        rows.append(dict(comparison="kernelshap_vs_exact", **_compare(exact, kernel_shap(predict, X, B))))
        rows.append(dict(comparison="marginal_reference_vs_mean_reference",
                         **_compare(exact, marginal_expectation_shap(predict, X, B))))
        balanced = class_balanced_background(data.X_train, data.y_train)
        rows.append(dict(comparison="class_balanced_reference_vs_mean_reference",
                         **_compare(exact, exact_q_shap_values(predict, X, balanced))))
    write_json(done, dict(spec=spec.as_dict(), rows=rows))
    return "validated"
