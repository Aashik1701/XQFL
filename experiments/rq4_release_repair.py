"""Corrected RQ4 rerun, isolated from every previously reported output.

Run: python experiments/rq4_release_repair.py --scope all
The initial sweep has 20 targets; all adds the existing clean-MIA conditions
(115 targets total, 195 release evaluations). Resume uses saved target models
and per-sample attributions, guarded by a code/protocol fingerprint.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = [1, 7, 42, 123, 999]
EXTRA = [11, 13, 17, 19, 23, 29, 31, 37, 43, 47]
ENC = ['angle', 'amplitude', 'reuploading', 'iqp']
OUT = ROOT / 'results/rq4_release_repair_v1'
PROTOCOL = dict(version=1, clients=4, rounds=15, local_epochs=3,
                qubits=4, layers=2, members=50, nonmembers=50,
                background='training rows 50:70, same for both groups; excluded from attack records',
                permutations=8, outer_folds=5, inner_folds=4,
                epsilon=[.1, .5, 1., 5.],
                orientation='inner OOF only; oracle max reported as diagnostic',
                threshold='inner OOF maximum TPR-FPR; evaluate on outer test',
                low_fpr='pooled OOF descriptive only; 50 negatives => 2% resolution',
                utility='same clean/noisy per-sample vectors as attack',
                shadow='not recomputed; old shadow scores excluded from corrected comparisons')


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def fingerprint():
    paths = ['privacy/dp_noise.py', 'privacy/release_evaluation.py',
             'experiments/rq4_release_repair.py', 'federation/run_federation.py',
             'federation/server.py', 'federation/client.py', 'explain/q_shap.py',
             'models/quantum_classifier.py', 'models/encodings.py', 'data/loaders.py',
             'data/partition.py', 'results/phase10_log.csv']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def tasks(scope):
    first = [('wisconsin', e, 'iid', s) for e in ENC for s in BASE]
    if scope == 'all':
        first += [('wisconsin', e, 'iid', s) for e in ENC for s in EXTRA]
        first += [('pima', 'amplitude', 'iid', s) for s in BASE + EXTRA]
        first += [('wisconsin', e, d, s) for d in ['alpha_0.1', 'qty_skew']
                  for e in ENC for s in BASE]
    return first


def calibration(enc):
    vals = []
    with (ROOT / 'results/phase10_log.csv').open() as f:
        for r in csv.DictReader(f):
            if (r['encoding'] == enc and r['run_type'] == 'federated'
                and r['distribution'] == 'iid' and r['client_id'] == 'global'
                and int(r['round']) >= 1 and r['track_class'] == 'A'):
                vals.append(float(r['norm_phi']))
    if len(vals) < 10:
        raise RuntimeError('Missing corrected empirical calibration; no fallback')
    return sum(vals) / len(vals)


def worker(task):
    import numpy as np
    import torch
    from data.loaders import load_breast_cancer_data, load_pima_data
    from federation.run_federation import run_federation, _fit_shared_preprocessor
    from models.quantum_classifier import HybridQuantumClassifier
    from models import encodings
    from explain.q_shap import q_shap_values, make_quantum_predict_fn
    from privacy.dp_noise import batch_dp_noise
    from privacy.release_evaluation import evaluate_release, utility

    dataset, enc, dist, seed = task
    key = f'{dataset}_{enc}_{dist}_{seed}'
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.monotonic()
    loader = load_breast_cancer_data if dataset == 'wisconsin' else load_pima_data
    train, test, ytrain, ytest = loader(random_state=seed)
    data_hash = hashlib.sha256(b''.join(a.tobytes() for a in [train, test, ytrain, ytest])).hexdigest()
    model = HybridQuantumClassifier(n_qubits=4, n_layers=2,
                                  encoding_fn=getattr(encodings, enc + '_encoding'))
    target = folder / 'target.pkl'
    if target.exists():
        with target.open('rb') as f:
            saved = pickle.load(f)  # only this runner's own local artifact
        assert saved['data_hash'] == data_hash
        model.pca, model._scaler = saved['pca'], saved['scaler']
        model.set_weights(saved['weights'])
    else:
        weights = []
        split = {'iid': 'iid', 'qty_skew': 'quantity', 'alpha_0.1': 'dirichlet'}[dist]
        _, rounds = run_federation(train, ytrain, test, ytest,
            encoding_fn=getattr(encodings, enc + '_encoding'),
            n_clients=4, n_rounds=15, local_epochs=3, n_qubits=4, n_layers=2,
            split=split, dirichlet_alpha=.1 if dist == 'alpha_0.1' else .5,
            strategy_name='fedavg', seed=seed, project_root=str(ROOT),
            n_permutations=0, model_out=weights)
        assert len(weights) == 1 and [r for r, _ in rounds] == list(range(16))
        model.pca, model._scaler = _fit_shared_preprocessor(train, 4)
        model.set_weights(weights[0])
        saved = dict(weights=weights[0], pca=model.pca, scaler=model._scaler,
                     rounds=rounds, data_hash=data_hash)
        with (folder / 'target.tmp').open('wb') as f:
            pickle.dump(saved, f)
        (folder / 'target.tmp').replace(target)
        import ray
        ray.shutdown()
    accuracy = float((model.predict(test) == ytest).mean())
    assert abs(accuracy - saved['rounds'][-1][1]) < 1e-8
    majority = float(np.bincount(ytest.astype(int)).max() / len(ytest))
    cleanfile = folder / 'clean.npz'
    if cleanfile.exists():
        with np.load(cleanfile) as f:
            clean, labels = f['clean'], f['labels']
    else:
        # The target training split is unchanged. The attack's fixed reference
        # uses training rows outside the 50 evaluated member records.
        bg = model._preprocess(train[50:70])
        explained = model._preprocess(np.concatenate([train[:50], test[:50]]))
        clean = q_shap_values(make_quantum_predict_fn(model), explained, bg,
                              n_permutations=8, seed=seed)
        labels = np.r_[np.ones(50, int), np.zeros(50, int)]
        np.savez_compressed(cleanfile, clean=clean, labels=labels,
                            background=bg, explained=explained,
                            member_indices=np.arange(50), nonmember_indices=np.arange(50),
                            background_train_indices=np.arange(50, 70))
    sensitivity = calibration(enc) if dataset == 'wisconsin' else None
    releases = [('clean', None)]
    if dataset == 'wisconsin' and dist == 'iid' and seed in BASE:
        releases += [(str(e), e) for e in PROTOCOL['epsilon']]
    rows = []
    for idx, (label, epsilon) in enumerate(releases):
        # One RNG draw for the combined member/nonmember matrix; no row reseeding.
        rng = np.random.default_rng(np.random.SeedSequence([seed, ENC.index(enc), idx, 20260928]))
        noisy = clean.copy() if epsilon is None else batch_dp_noise(
            clean, epsilon, sensitivity=sensitivity, seed=rng)
        metrics, scores, folds = evaluate_release(noisy, labels, seed)
        metrics.update(utility(clean, noisy))
        row = dict(dataset=dataset, encoding=enc, distribution=dist, seed=seed,
                   epsilon=epsilon, sensitivity=sensitivity, target_acc=accuracy,
                   majority=majority, track_class='A' if accuracy > majority + .02 else 'B',
                   n_members=50, n_nonmembers=50, **metrics)
        rows.append(row)
        np.savez_compressed(folder / f'release_{label}.npz', clean=clean,
                            noisy=noisy, **scores)
        dump(folder / f'folds_{label}.json', folds)
        print('EVALUATED', key, label, metrics, flush=True)
    dump(folder / 'complete.json', dict(task=task, data_hash=data_hash,
         elapsed_seconds=time.monotonic()-t0, rows=rows))


def aggregate(plan):
    import pandas as pd
    rows = []
    completed = []
    for task in plan:
        path = OUT / ('_'.join(map(str, task))) / 'complete.json'
        if path.exists():
            content = json.loads(path.read_text())
            rows.extend(content['rows'])
            completed.append(task)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / 'per_seed_results.csv', index=False)
        # Preserve all raw Track-B results; functional comparisons use A only.
        a = df[df.track_class == 'A'].copy()
        a['release'] = a.epsilon.fillna('clean')
        measures = ['auc_mean_fold', 'auc_oriented_mean_fold',
                    'auc_orientation_oracle_diagnostic', 'attack_advantage',
                    'cosine_mean', 'l2_mean']
        a.groupby(['dataset', 'encoding', 'distribution', 'release'])[measures].agg(
            ['count', 'mean', 'std']).to_csv(OUT / 'summary_track_a.csv')
    dump(OUT / 'status.json', dict(planned=len(plan), completed=len(completed),
                                 evaluations=len(rows), complete=len(plan)==len(completed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['sweep', 'all'], default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--worker', nargs=4)
    args = parser.parse_args()
    if args.worker:
        d, e, r, s = args.worker
        worker((d, e, r, int(s)))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol=PROTOCOL, source_hashes=fingerprint())
    file = OUT / 'manifest.json'
    if file.exists():
        if json.loads(file.read_text()) != manifest:
            raise RuntimeError('Code/protocol changed since run began; use a new version directory')
    else:
        dump(file, manifest)
        sources = ['phase13_mia_log.csv', 'rq4_extension_log.csv',
                   'rq4_dp_extension_log.csv', 'rq4_noniid_mia_log.csv',
                   'rq4_pima_mia_amplitude_log.csv']
        dump(OUT / 'historical_output_hashes.json', {
            p: hashlib.sha256((ROOT/'results'/p).read_bytes()).hexdigest() for p in sources})
    plan = tasks(args.scope)
    if args.limit:
        plan = plan[:args.limit]
    aggregate(plan)
    for i, task in enumerate(plan):
        folder = OUT / '_'.join(map(str, task))
        if (folder/'complete.json').exists():
            continue
        if shutil.disk_usage(OUT).free < 5 * 1024**3:
            raise RuntimeError('Less than 5 GiB free; stopped before training')
        folder.mkdir(exist_ok=True)
        print(f'START {i+1}/{len(plan)} {task}', flush=True)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
        # A short task-owned Ray directory prevents filling shared /tmp sessions.
        import tempfile
        raydir = Path(tempfile.mkdtemp(prefix='xqfl-rq4-', dir='/private/tmp'))
        env['RAY_TMPDIR'] = str(raydir)
        log = folder / 'run.log'
        with log.open('a') as f:
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--worker', *map(str, task)], cwd=ROOT, env=env,
                stdout=f, stderr=subprocess.STDOUT)
        # Only remove this task's newly created Ray scratch directory.
        shutil.rmtree(raydir)
        if run.returncode:
            raise RuntimeError(f'Task failed; see {log}')
        # The unchanged federation server logs the actual observed IDs.
        text = log.read_text()
        for round_id in range(1, 16):
            if f'[round {round_id}] partitions trained: [0, 1, 2, 3]' not in text:
                raise RuntimeError(f'Missing partition evidence for round {round_id}: {log}')
        dump(folder / 'partition_audit.json', dict(rounds=15, ids=[0,1,2,3], passed=True))
        aggregate(plan)
        print(f'COMPLETE {i+1}/{len(plan)} {task}', flush=True)


if __name__ == '__main__':
    main()
