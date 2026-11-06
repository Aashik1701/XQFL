"""One-time, backed-up downstream RQ4 source promotion; never runs training."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'results/pre_rq4_release_promotion_20260928'

def backup():
    if (BACKUP / 'manifest.json').exists():
        return
    paths = set()
    for pattern in ['*.md', '*.docx', '*.pdf', 'PAPER_FORMAT/*.docx', 'PAPER_FORMAT/*.md',
                    'report_build/*.js', 'report_build/*.docx', 'report_build/assets/*',
                    'analysis/aggregate_results.py', 'analysis/plots*.py', 'results/outputs/table*',
                    'results/outputs/figure[56]*', 'results/outputs/final/fig0[67]*',
                    'results/outputs/final/CAPTIONS.md', 'reproducibility/*.md']:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and not p.name.startswith('~$'))
    manifest = {}
    for p in sorted(paths):
        rel = p.relative_to(ROOT)
        dest = BACKUP / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        manifest[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    (BACKUP / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (BACKUP / 'README.md').write_text('# Historical snapshot\n\nPre-release-evaluation-repair presentation artifacts. Not current RQ4 evidence. Original audit history is preserved.\n')

def replace_function(path, start, end, body):
    text = path.read_text()
    a, b = text.index(start), text.index(end, text.index(start))
    path.write_text(text[:a] + body + '\n\n' + text[b:])

def promote_builders():
    p = ROOT / 'analysis/aggregate_results.py'
    replace_function(p, 'def build_rq4_mia()', '# ── RQ4b:',
        'def build_rq4_mia() -> pd.DataFrame:\n    from analysis.current_rq4 import mia_table\n    return mia_table()')
    replace_function(p, 'def build_rq4_dp()', '# ── output helpers',
        'def build_rq4_dp() -> pd.DataFrame:\n    from analysis.current_rq4 import noise_table\n    return noise_table()')
    text = p.read_text().replace('Table 5 — RQ4 Differential Privacy:', 'Table 5 — RQ4 Calibrated Laplace-Noise Evaluation:')
    p.write_text(text)

def promote_plots():
    p = ROOT / 'analysis/plots_final.py'
    replace_function(p, 'def make_fig06(', '# ─────────────────────────────────────────────────────────────────────────────\n# Figure 7',
        "def make_fig06(rq4_mia_table):\n    from analysis.plot_current_rq4 import membership\n    membership(out_dir())")
    replace_function(p, 'def _release_utility(', '# ─────────────────────────────────────────────────────────────────────────────\n# Figure 8',
        "def _release_utility(*args, **kwargs):\n    from analysis.current_rq4 import matched_utility\n    return matched_utility()\n\ndef make_fig07(dp_table, util):\n    from analysis.plot_current_rq4 import noise\n    noise(out_dir())")
    text = p.read_text().replace('dp_log = pd.read_csv(PROJECT_ROOT / "results" / "rq4_dp_extension_log.csv")', 'dp_log = None  # corrected persisted release records are read by current_rq4')
    text = text.replace("Fig. 6  RQ4 membership leakage: per-encoding AUC, per-seed spread, shadow control", "Fig. 6  RQ4 corrected direct-MIA AUC and per-seed spread")
    a = text.index('Two panels are derived rather than directly logged')
    b = text.index('Output: vector', a)
    text = text[:a] + 'Figure 7 uses utility measured on the same persisted noisy releases as the attack.\nS1 and S3 derive partitions and PCA from deterministic seeded paths.\n\n' + text[b:]
    p.write_text(text)
    p = ROOT / 'analysis/plots.py'
    replace_function(p, 'def make_figure4(', 'def make_figure5(',
        "def make_figure4(rq4_mia_table, out_dir):\n    from analysis.plot_current_rq4 import membership\n    membership(out_dir, 'figure5_mia_auc_comparison', panels=False)")
    replace_function(p, 'def make_figure5(', 'def make_figure6_coverage(',
        "def make_figure5(rq4_dp_table, out_dir):\n    from analysis.plot_current_rq4 import noise\n    noise(out_dir, 'figure6_dp_auc_vs_epsilon', utility=False)")

if __name__ == '__main__':
    backup()
    promote_builders()
    promote_plots()
