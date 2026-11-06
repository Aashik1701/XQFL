"""Auditable release evaluation; no training or files are written on import.

Raw mean fold AUC retains the old aggregation convention. Score orientation and
operating thresholds are chosen with inner out-of-fold predictions ONLY.
Evaluation-selected max(AUC,1-AUC) and ROC points are labelled diagnostics.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def classifier(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=500, random_state=seed))


def evaluate_release(X, y, seed=42, n_features=4):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if X.shape != (len(y), n_features) or not np.isfinite(X).all():
        raise ValueError(f"Expected finite (n,{n_features}) attributions")
    if set(np.unique(y)) != {0, 1} or min(np.bincount(y)) < 5:
        raise ValueError("Need both membership classes and at least five per class")
    raw = np.full(len(y), np.nan)
    oriented = raw.copy()
    decisions = np.zeros(len(y), dtype=int)
    folds = np.full(len(y), -1, int)
    details = []
    for fold, (train, test) in enumerate(StratifiedKFold(
            5, shuffle=True, random_state=seed).split(X, y)):
        inner = StratifiedKFold(4, shuffle=True, random_state=seed + fold + 1)
        validation = cross_val_predict(classifier(seed), X[train], y[train],
                                       cv=inner, method="predict_proba")[:, 1]
        reverse = bool(roc_auc_score(y[train], validation) < .5)
        calibrated = 1 - validation if reverse else validation
        fpr, tpr, thresholds = roc_curve(y[train], calibrated)
        threshold = float(thresholds[np.argmax(tpr - fpr)])
        clf = classifier(seed).fit(X[train], y[train])
        assert np.array_equal(clf.classes_, [0, 1])
        raw[test] = clf.predict_proba(X[test])[:, 1]
        oriented[test] = 1 - raw[test] if reverse else raw[test]
        decisions[test] = oriented[test] >= threshold
        folds[test] = fold
        details.append(dict(fold=fold, train_indices=train.tolist(),
                            test_indices=test.tolist(), reverse=reverse,
                            threshold=threshold if np.isfinite(threshold) else None,
                            auc_raw=float(roc_auc_score(y[test], raw[test])),
                            auc_oriented=float(roc_auc_score(y[test], oriented[test]))))
    assert np.isfinite(raw).all() and (folds >= 0).all()
    auc = float(np.mean([d['auc_raw'] for d in details]))
    tpr = float(decisions[y == 1].mean())
    fpr = float(decisions[y == 0].mean())
    # These pooled OOF ROC points describe the observed curve; they are NOT
    # independently validated deployment thresholds, especially at low FPR.
    fp, tp, _ = roc_curve(y, oriented)
    summary = dict(auc_mean_fold=auc,
                   auc_oriented_mean_fold=float(np.mean([d['auc_oriented'] for d in details])),
                   auc_orientation_oracle_diagnostic=max(auc, 1-auc),
                   auc_pooled_oof=float(roc_auc_score(y, raw)),
                   attack_advantage=tpr-fpr, attack_tpr=tpr, attack_fpr=fpr,
                   fpr_resolution=1 / int((y == 0).sum()))
    for rate in (.01, .02, .05):
        summary[f'tpr_at_fpr_{rate}_oof_diagnostic'] = float(tp[fp <= rate].max())
    return summary, dict(labels=y, score_raw=raw, score_oriented=oriented,
                         decision=decisions, fold=folds), details


def utility(clean, noisy):
    """Utility on exactly the per-sample releases supplied to the attack."""
    denom = np.linalg.norm(clean, axis=1) * np.linalg.norm(noisy, axis=1)
    valid = denom > 1e-15
    cos = np.sum(clean[valid] * noisy[valid], axis=1) / denom[valid]
    return dict(cosine_mean=float(cos.mean()) if len(cos) else None,
                cosine_valid_n=int(valid.sum()), cosine_undefined_n=int((~valid).sum()),
                l2_mean=float(np.linalg.norm(clean-noisy, axis=1).mean()))
