"""Audit persisted corrected RQ4 artifacts, without training or changing old data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rq4_release_repair import OUT, tasks, fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--partial', action='store_true')
    p.add_argument('--results-only', action='store_true', help='Audit evidence after authorized downstream promotion; preserve original AUDIT.md')
    args = p.parse_args()
    protected = OUT/'protected_artifacts.json'
    if args.baseline:
        if protected.exists():
            raise RuntimeError('Baseline exists; do not overwrite')
        files = list((ROOT/'results/outputs').glob('table*.csv'))
        files += [ROOT/'XQFL_paper_humanized.docx', ROOT/'XQFL_paper.md']
        protected.write_text(json.dumps({str(f.relative_to(ROOT)): digest(f)
                                        for f in files if f.exists()}, indent=2))
        return
    errors, rows, complete = [], [], []
    manifest = json.loads((OUT/'manifest.json').read_text())
    if manifest['source_hashes'] != fingerprint():
        errors.append('Source/protocol fingerprint changed')
    if not args.results_only:
        for name, value in json.loads(protected.read_text()).items():
            if digest(ROOT/name) != value:
                errors.append('Protected artifact changed: '+name)
    for name, value in json.loads((OUT/'historical_output_hashes.json').read_text()).items():
        if digest(ROOT/'results'/name) != value:
            errors.append('Historical result changed: '+name)
    for task in tasks('all'):
        key = '_'.join(map(str, task))
        folder = OUT/key
        path = folder/'complete.json'
        if not path.exists():
            continue
        obj = json.loads(path.read_text())
        complete.append(key)
        if not (folder/'partition_audit.json').exists():
            errors.append(key+': partition audit missing')
        for r in obj['rows']:
            label = 'clean' if r['epsilon'] is None else str(r['epsilon'])
            with np.load(folder/f'release_{label}.npz') as data:
                y, raw = data['labels'], data['score_raw']
                assert len(y) == 100 and np.array_equal(y, np.r_[np.ones(50),np.zeros(50)])
                folds = json.loads((folder/f'folds_{label}.json').read_text())
                seen, aucs = [], []
                for fold in folds:
                    train, test = fold['train_indices'], fold['test_indices']
                    assert not set(train) & set(test)
                    seen.extend(test)
                    aucs.append(roc_auc_score(y[test], raw[test]))
                    expected = 1-raw[test] if fold['reverse'] else raw[test]
                    np.testing.assert_allclose(data['score_oriented'][test], expected)
                assert sorted(seen) == list(range(100))
                assert abs(np.mean(aucs)-r['auc_mean_fold']) < 1e-12
                if r['epsilon'] is not None:
                    noise = data['noisy']-data['clean']
                    assert not np.allclose(noise[1:50], noise[50:99])
                from privacy.release_evaluation import utility
                u = utility(data['clean'], data['noisy'])
                assert u == {k:r[k] for k in u}
            rows.append(r)
    if len(complete) != 115 and not args.partial:
        errors.append(f'Incomplete: {len(complete)}/115 targets')
    new = pd.DataFrame(rows)
    if len(new):
        assert not new.duplicated(['dataset','encoding','distribution','seed','epsilon']).any()
        old_specs = [('phase13_mia_log.csv','wisconsin','iid'),
                     ('rq4_extension_log.csv','wisconsin','iid'),
                     ('rq4_pima_mia_amplitude_log.csv','pima','iid'),
                     ('rq4_noniid_mia_log.csv','wisconsin',None),
                     ('rq4_dp_extension_log.csv','wisconsin','iid')]
        old = []
        for name, dataset, dist in old_specs:
            d = pd.read_csv(ROOT/'results'/name)
            d['dataset'] = dataset
            if dist: d['distribution'] = dist
            if 'epsilon' not in d: d['epsilon'] = np.nan
            old.append(d)
        historical = pd.concat(old,ignore_index=True)
        keys = ['dataset','encoding','distribution','seed','epsilon']
        merged = new.merge(historical[keys+['direct_mia_auc']], on=keys, how='left', validate='one_to_one')
        merged['auc_change'] = merged.auc_mean_fold-merged.direct_mia_auc
        merged.to_csv(OUT/'old_vs_new.csv',index=False)
        # Both all-target and functional views are retained, explicitly labelled.
        new['release'] = new.epsilon.fillna('clean')
        for name, selected in [('all_targets',new), ('track_a',new[new.track_class=='A'])]:
            selected.groupby(['dataset','encoding','distribution','release'])[
                ['auc_mean_fold','auc_oriented_mean_fold','attack_advantage',
                 'cosine_mean','l2_mean']].agg(['count','mean','std']).to_csv(
                     OUT/f'audited_summary_{name}.csv')
    status = 'PASS' if not errors and len(complete)==115 else 'PARTIAL' if not errors else 'FAIL'
    report = [f'# RQ4 repair audit — {status}', '',
              f'Completed targets: {len(complete)}/115. Release evaluations: {len(rows)}/195.',
              '', 'Historical outputs and protected manuscript/table hashes checked.',
              'Persisted scores, fold disjointness, AUC aggregation, noise non-reuse,',
              'matched utility, partition evidence and source hashes checked.', '',
              'Comparison: old_vs_new.csv. Changes combine attack repairs with target',
              'retraining and must not be attributed to a single correction.', '',
              'Oracle orientation AUC and pooled low-FPR ROC points are diagnostics.',
              'They are not validated deployment attack performance or privacy guarantees.',
              'Shadow-model results have not been re-evaluated.', '',
              'Scientific conclusions require review of completed results; a mechanical',
              'PASS is not evidence that mitigation works.', '', '## Errors', '']
    report += errors or ['None detected in completed artifacts.']
    if args.results_only:
        report = [line.replace('Historical outputs and protected manuscript/table hashes checked.',
                  'Historical raw outputs checked; authorized manuscript/table changes audited separately.') for line in report]
    (OUT/('RESULTS_ONLY_AUDIT.md' if args.results_only else 'AUDIT.md')).write_text('\n'.join(report)+'\n')
    print('\n'.join(report))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
