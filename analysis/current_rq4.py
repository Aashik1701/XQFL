"""Current RQ4 summaries: corrected release evaluation only, no legacy fallback."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/rq4_release_repair_v1/per_seed_results.csv'
ENCODINGS = ['angle', 'amplitude', 'reuploading', 'iqp']

def records():
    df = pd.read_csv(SOURCE)
    keys = ['dataset', 'encoding', 'distribution', 'seed', 'epsilon']
    if len(df) != 195 or df.duplicated(keys).any():
        raise ValueError('Expected 195 unique completed release evaluations')
    if df['auc_mean_fold'].isna().any():
        raise ValueError('Missing corrected AUC')
    return df

def mia_table():
    rows = []
    clean = records().query('epsilon != epsilon')
    for (dataset, encoding, distribution), all_runs in clean.groupby(['dataset', 'encoding', 'distribution'], sort=True):
        runs = all_runs if distribution == 'iid' else all_runs.query('track_class == "A"')
        n = len(runs)
        status = 'ok' if n >= 3 else ('insufficient' if n else 'absent')
        rows.append(dict(dataset=dataset, encoding=encoding, distribution=distribution,
                         n=n, auc_mean=runs.auc_mean_fold.mean() if n >= 3 else float('nan'),
                         auc_std=runs.auc_mean_fold.std() if n >= 3 else float('nan'),
                         collapse_rate=(all_runs.track_class == 'B').mean(), status=status,
                         n_total=len(all_runs), n_track_a=int((all_runs.track_class == 'A').sum()),
                         note='release_repair_v1; mean five outer-fold raw AUCs; ' +
                         ('all IID targets; Track-A sensitivity reported separately' if distribution == 'iid' else 'Track-A only; at least 3 seeds required')))
    return pd.DataFrame(rows)

def noise_table():
    rows = []
    for (encoding, epsilon), runs in records().dropna(subset=['epsilon']).groupby(['encoding', 'epsilon']):
        if len(runs) != 5 or not (runs.track_class == 'A').all():
            raise ValueError('Unexpected noise-sweep coverage')
        rows.append(dict(dataset='wisconsin', encoding=encoding, epsilon=epsilon, n=len(runs),
                         auc_mean=runs.auc_mean_fold.mean(), auc_std=runs.auc_mean_fold.std(),
                         status='ok', note='release_repair_v1; independent per-record noise; raw fold-mean AUC; no privacy guarantee'))
    return pd.DataFrame(rows)

def matched_utility():
    df = records().dropna(subset=['epsilon'])
    return df.groupby(['encoding', 'epsilon']).agg(cos_mean=('cosine_mean', 'mean'),
             cos_std=('cosine_mean', 'std'), n_vectors=('seed', 'size')).reset_index()

def diagnostics():
    df = records().copy()
    df['release'] = df.epsilon.map(lambda e: 'clean' if pd.isna(e) else f'epsilon={e:g}')
    metrics = ['auc_mean_fold', 'auc_oriented_mean_fold', 'auc_orientation_oracle_diagnostic',
               'attack_advantage', 'tpr_at_fpr_0.01_oof_diagnostic',
               'tpr_at_fpr_0.02_oof_diagnostic', 'tpr_at_fpr_0.05_oof_diagnostic', 'cosine_mean']
    return df.groupby(['dataset', 'encoding', 'distribution', 'release'])[metrics].agg(['mean', 'std', 'count'])
