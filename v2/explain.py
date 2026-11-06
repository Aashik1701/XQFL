"""Exact Shapley explanations from saved weights, and the per-run metric summary."""
import json
import os
import time

import numpy as np
from scipy.stats import kendalltau

from explain.metrics import js_divergence
from explain.q_shap import exact_q_shap_values, make_quantum_predict_fn
from v2 import config
from v2.data import explanation_sets, make_data
from v2.models import build_model, load_flat
from v2.train import initial_seed, write_json


def importance(samples):
    """Mean absolute attribution over records, ignoring padded rows (primary)."""
    return np.nanmean(np.abs(samples), axis=-2)


def signed_mean(samples):
    """Mean signed attribution over records (secondary)."""
    return np.nanmean(samples, axis=-2)


def explain_with_weights(model, weights, X, background):
    load_flat(model, weights)
    return exact_q_shap_values(make_quantum_predict_fn(model), X, background)


def _pad(values, n, d):
    out = np.full((n, d), np.nan, dtype=np.float64)
    out[:len(values)] = values
    return out


def client_explanation_rounds(spec):
    """Rounds with client explanations: every third round, always including the last."""
    if spec.explain_mode != "full":
        return [spec.n_rounds]
    rounds = list(range(config.CLIENT_EXPLAIN_EVERY, spec.n_rounds + 1, config.CLIENT_EXPLAIN_EVERY))
    return rounds if rounds[-1] == spec.n_rounds else rounds + [spec.n_rounds]


def compute_explanations(spec, weights, data, sets):
    model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))
    d = spec.n_features
    E = data.X_test[sets.E_global]
    B = data.X_train[sets.B_global]
    history = weights["global_weights"]
    arrays = {}
    if spec.kind == "cent":
        arrays["phi_global"] = explain_with_weights(model, history[-1], E, B)[None]
        return arrays

    global_rounds = list(range(spec.n_rounds + 1))
    client_rounds = client_explanation_rounds(spec)
    arrays["phi_global"] = np.stack([explain_with_weights(model, history[t], E, B)
                                     for t in global_rounds])
    K = spec.n_clients
    shape = (len(client_rounds), K)
    matched = np.full(shape + (len(E), d), np.nan)
    population = np.full(shape + (config.N_LOCAL_EXPLAINED, d), np.nan)
    floor = np.full(shape + (config.N_LOCAL_EXPLAINED, d), np.nan)
    client_weights = weights["client_weights"]
    for i, t in enumerate(client_rounds):
        for k in range(K):
            if sets.E_local[k] is None:
                continue
            E_k = data.X_train[sets.E_local[k]]
            B_k = data.X_train[sets.B_local[k]]
            w_k = client_weights[t - 1, k]
            matched[i, k] = explain_with_weights(model, w_k, E, B)
            population[i, k] = _pad(explain_with_weights(model, w_k, E_k, B_k),
                                    config.N_LOCAL_EXPLAINED, d)
            floor[i, k] = _pad(explain_with_weights(model, history[t], E_k, B_k),
                               config.N_LOCAL_EXPLAINED, d)
    arrays.update(phi_matched=matched, phi_population=population, phi_floor=floor,
                  client_rounds=np.array(client_rounds))
    return arrays


def _tau(a, b):
    value = kendalltau(a, b).statistic
    return float(value) if np.isfinite(value) else np.nan


def _cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-15 else np.nan


def stability_metrics(vectors):
    """Metrics over rounds 1..T; vectors[0] is the initial model."""
    trained = vectors[1:]
    pairs = list(zip(trained[:-1], trained[1:]))
    orderings = [tuple(np.argsort(-v)) for v in trained]
    changes = [t for t in range(1, len(orderings)) if orderings[t] != orderings[t - 1]]
    return dict(
        tau=float(np.nanmean([_tau(a, b) for a, b in pairs])),
        tau_init=_tau(vectors[0], vectors[1]),
        cosine=float(np.nanmean([_cosine(a, b) for a, b in pairs])),
        l2=float(np.mean([np.linalg.norm(a - b) for a, b in pairs])),
        top_change_rate=float(np.mean([np.argmax(a) != np.argmax(b) for a, b in pairs])),
        # First trained round from which the full ordering no longer changes.
        settle_round=(changes[-1] + 1) if changes else 1,
    )


def summarize(spec, arrays, sets):
    """Per-run values for RQ1-RQ3. Everything uses importance (I) and signed (S)."""
    phi_global = arrays["phi_global"]
    summary = dict(I_final=importance(phi_global[-1]).tolist(),
                   S_final=signed_mean(phi_global[-1]).tolist())
    if spec.kind == "cent":
        return summary
    I = importance(phi_global)
    S = signed_mean(phi_global)
    # A signed stability read keeps the ordering of raw values, as in v1.
    summary["stability_I"] = stability_metrics(I)
    summary["stability_S"] = stability_metrics(S)

    # Global explanation rounds that pair with the client rounds.
    rounds = arrays["client_rounds"]
    half_a, half_b = sets.half_a, sets.half_b
    sample_I = [js_divergence(importance(phi_global[t][half_a]), importance(phi_global[t][half_b]))
                for t in rounds]
    sample_S = [js_divergence(signed_mean(phi_global[t][half_a]), signed_mean(phi_global[t][half_b]))
                for t in rounds]
    summary["D_sample_I"], summary["D_sample_S"] = float(np.mean(sample_I)), float(np.mean(sample_S))
    summary["D_sample_I_final"], summary["D_sample_S_final"] = sample_I[-1], sample_S[-1]

    clients = []
    for k in range(spec.n_clients):
        if sets.E_local[k] is None:
            clients.append(dict(client=k, sparse=True))
            continue
        entry = dict(client=k, sparse=False)
        for name, phi in (("model", arrays["phi_matched"]),
                          ("pop", arrays["phi_population"]),
                          ("floor", arrays["phi_floor"])):
            values_I = [js_divergence(importance(phi[i, k]), I[t]) for i, t in enumerate(rounds)]
            values_S = [js_divergence(signed_mean(phi[i, k]), S[t]) for i, t in enumerate(rounds)]
            entry[f"D_{name}_I"] = float(np.mean(values_I))
            entry[f"D_{name}_S"] = float(np.mean(values_S))
            entry[f"D_{name}_I_final"] = values_I[-1]
            entry[f"D_{name}_S_final"] = values_S[-1]
        clients.append(entry)
    summary["clients"] = clients
    return summary


def explain_task(spec, out_dir):
    folder = out_dir / spec.key
    done = folder / "summary.json"
    if done.exists():
        return "skipped"
    started = time.monotonic()
    meta = json.loads((folder / "train.json").read_text())
    data = make_data(spec.dataset, spec.seed, spec.n_features, spec.zero_handling)
    with np.load(folder / "weights.npz") as source:
        weights = {name: source[name] for name in source.files}
    n_partitions = sum(name.startswith("partition_") for name in weights)
    partitions = [weights[f"partition_{k}"] for k in range(n_partitions)]
    sets = explanation_sets(data, partitions, spec.seed)
    arrays = compute_explanations(spec, weights, data, sets)
    temporary = folder / "explanations.tmp.npz"
    np.savez_compressed(temporary, **{name: (a.astype(np.float32) if a.dtype == np.float64 else a)
                                      for name, a in arrays.items()},
                        E_global=sets.E_global, B_global=sets.B_global)
    os.replace(temporary, folder / "explanations.npz")
    summary = summarize(spec, arrays, sets)
    summary.update(spec=spec.as_dict(), val_acc=meta["val_acc"][-1], test_acc=meta["test_acc"][-1],
                   train_acc=meta["train_acc_final"], val_majority=meta["val_majority"],
                   test_majority=meta["test_majority"],
                   partition_sizes=meta.get("partition_sizes"),
                   seconds_explain=time.monotonic() - started, seconds_train=meta["seconds"])
    write_json(done, summary)
    return "explained"
