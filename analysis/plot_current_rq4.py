"""Render current RQ4 figures from the completed, paired release records."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'analysis'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plots import ENC_COLOR, ENC_MARKER, ENC_LS, ENC_LABEL
from analysis.current_rq4 import records, mia_table, noise_table, matched_utility, diagnostics, ENCODINGS

def save(fig, stem, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(directory / f'{stem}.{ext}', dpi=300)
    plt.close(fig)

def membership(directory, stem='fig06_membership', panels=True):
    df = records().query('epsilon != epsilon and distribution == "iid"')
    groups = [('wisconsin', e) for e in ENCODINGS] + [('pima', 'amplitude')]
    fig, axes = plt.subplots(2 if panels else 1, 1, figsize=(6.9, 6.2 if panels else 4.1))
    axes = np.atleast_1d(axes)
    for i, (dataset, e) in enumerate(groups):
        rows = df.query('dataset == @dataset and encoding == @e')
        raw = rows.auc_mean_fold.to_numpy()
        axes[0].errorbar(i, raw.mean(), yerr=raw.std(ddof=1), fmt=ENC_MARKER[e],
                         color=ENC_COLOR[e], capsize=4, ms=7, lw=1.3)
        if panels:
            # Fixed horizontal offsets are presentation only, not data perturbation.
            axes[1].scatter(i + np.linspace(-.14, .14, len(raw)), raw,
                            marker=ENC_MARKER[e], color=ENC_COLOR[e], s=22, alpha=.8)
    labels = [ENC_LABEL[e] + ('\nPima' if d == 'pima' else '\nWisconsin') for d, e in groups]
    for ax in axes:
        ax.axhline(.5, ls='--', lw=1, color='#555555', label='Chance (AUC = 0.5)')
        ax.set_xticks(range(5), labels, fontsize=8)
        ax.set_ylabel('Direct-MIA AUC')
        ax.set_ylim(.2, .8)
        ax.grid(axis='x', visible=False)
        ax.set_xlim(-.5, 4.5)
    axes[0].set_title(('A  ' if panels else '') + 'Mean ± SD across 15 target seeds', loc='left', fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    if panels:
        axes[1].set_title('B  Per-seed mean of five held-out fold AUCs', loc='left', fontsize=10)
    fig.suptitle('RQ4  Membership-inference signal: corrected evaluation', fontsize=11)
    fig.tight_layout(rect=(0, .02, 1, .96), h_pad=2)
    save(fig, stem, directory)

def noise(directory, stem='fig07_privacy_utility', utility=True):
    a, u = noise_table(), matched_utility()
    fig, axes = plt.subplots(1, 2 if utility else 1, figsize=(6.9, 4.6 if utility else 4.1))
    axes = np.atleast_1d(axes)
    for e in ENCODINGS:
        sub = a.query('encoding == @e').sort_values('epsilon').merge(u.query('encoding == @e'), on=['encoding', 'epsilon'])
        style = dict(color=ENC_COLOR[e], marker=ENC_MARKER[e], ls=ENC_LS[e],
                     ms=5, lw=1.1, capsize=2, label=ENC_LABEL[e])
        axes[0].errorbar(range(4), sub.auc_mean, yerr=sub.auc_std, **style)
        if utility:
            axes[1].errorbar(sub.cos_mean, sub.auc_mean, xerr=sub.cos_std,
                             yerr=sub.auc_std, **style)
    axes[0].set_xticks(range(4), ['0.1', '0.5', '1', '5'])
    axes[0].set_xlabel(r'Noise-level index $\epsilon$ (categorical)')
    axes[0].set_ylabel('Direct-MIA AUC')
    axes[0].set_title(('A  ' if utility else '') + 'Noise response', loc='left', fontsize=10)
    if utility:
        axes[1].set_xlabel('Matched clean/noisy cosine similarity')
        axes[1].set_xlim(-.1, 1)
        axes[1].set_title('B  Matched release utility', loc='left', fontsize=10)
    for ax in axes:
        ax.axhline(.5, ls='--', color='#555555', lw=1)
        ax.set_ylim(.25, .75)
        ax.grid(axis='x', visible=False)
    fig.suptitle('RQ4  Calibrated Laplace noise: signal and utility', fontsize=11, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.text(.5, .885, 'Chance: dashed line at 0.5  |  mean ± SD, n = 5  |  no formal privacy guarantee',
             ha='center', fontsize=7.8)
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.78, wspace=.25)
    save(fig, stem, directory)

def main():
    out = ROOT / 'results/outputs'
    membership(out / 'final')
    noise(out / 'final')
    membership(out, 'figure5_mia_auc_comparison', panels=False)
    noise(out, 'figure6_dp_auc_vs_epsilon', utility=False)
    matched_utility().to_csv(out / 'final/fig07_utility_derived.csv', index=False)
    diagnostics().to_csv(out / 'rq4_release_diagnostics.csv')
    df = records().query('epsilon != epsilon and distribution == "iid" and track_class == "A"')
    df.groupby(['dataset', 'encoding']).auc_mean_fold.agg(['count', 'mean', 'std']).to_csv(out / 'rq4_iid_track_a_sensitivity.csv')

if __name__ == '__main__':
    main()
