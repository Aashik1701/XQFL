# Reproducing XQFL

Every number in `XQFL_paper.md` traces to a file under `results/`. This document
gives the exact commands, in three tiers of cost.

All commands run **from the repository root** with the project virtualenv active.

---

## 0. Environment

```bash
python3.11 -m venv xqfl-env
source xqfl-env/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full pinned freeze of the environment the reported results
were produced in. Load-bearing pins:

| package | version | why it matters |
|---|---|---|
| `flwr` | 1.32.1 | `client_fn` reads `context.node_config["partition-id"]`; the partition-to-client binding depends on this API (paper §6.7) |
| `pennylane` | 0.45.1 | circuit simulation + autodiff |
| `pennylane_lightning` | 0.45.0 | `lightning.qubit` backend |
| `scikit-learn` | 1.9.0 | PCA, the MIA logistic-regression attack, `train_test_split` |
| `numpy` | 2.4.6 / `torch` | 2.13.0 | RNG streams; seed reproducibility |

Python 3.11. No GPU required (all runs are CPU + simulator).

PDF build additionally needs `pandoc` (≥ 3) and `tectonic` on `PATH`, plus a serif
font with Greek + math coverage (macOS ships *STIX Two Text* / *STIX Two Math*; on
Linux `apt install fonts-stix` or change `-V mainfont`/`-V mathfont` in `build_pdf.sh`).

---

## Tier A — rebuild tables and figures from the committed raw logs (seconds)

This is the fastest full check: it re-derives Tables 1–5 and Figures 2–5 from the
per-seed / per-round CSVs already in `results/`, with no training.

```bash
python analysis/aggregate_results.py     # -> results/outputs/table{1..5}_*.{csv,md}
python analysis/plots.py                  # -> results/outputs/figure{2..5}_*.png
python analysis/gen_figure1.py            # -> results/outputs/figure1_architecture.png
```

Then compare `results/outputs/table*.md` against the tables in `XQFL_paper.md`
§7.1–§7.4 and Appendix A. `results/FINAL_MANUSCRIPT_AUDIT.md` records the last
cell-by-cell verification (all PASS).

Inputs consumed by `aggregate_results.py`:

| table | source CSV(s) | builder function |
|---|---|---|
| Table 1 (RQ1) | `phase10_log.csv`, `phase11_log.csv`, `phase16_pima_log.csv` | `build_rq1_rq3` / `jsd_cell` |
| Table 2 (RQ2) | + `rq2_extension_log.csv` | `build_rq2` / `stability_cell` |
| Table 3 (RQ3) | same explanation logs (per-client + global rows) | `build_rq3_local_global` / `rq3_local_global_cell` |
| Table 4 (RQ4 MIA) | `phase13_mia_log.csv`, `rq4_extension_log.csv`, `rq4_noniid_mia_log.csv`, `rq4_pima_mia_amplitude_log.csv` | `build_rq4_mia` |
| Table 5 (RQ4 DP) | `rq4_dp_extension_log.csv` | `build_rq4_dp` |

## Tier B — rebuild the PDF (seconds)

```bash
bash build_pdf.sh        # -> XQFL_paper.pdf   (pandoc --pdf-engine=tectonic + fix_approx.lua)
```

`fix_approx.lua` maps the handful of Unicode math operators (≈ → ∈ ∪ ⊆ ≤ ≥ ≠) to
LaTeX math; Greek and the remaining symbols are covered by the STIX fonts. The
manuscript uses plain-text mathematical notation (`φ_g^(t)`, `JSD_cent`, `Kendall's τ`)
throughout — it renders as consistent technical prose, not typeset math environments.

## Tier C — regenerate the raw logs from scratch (~12 h CPU, checkpointed)

Only needed to reproduce `results/*.csv` themselves. Each sub-script is resume-safe
via its own `results/<name>_checkpoint.csv`, so the orchestrators can be killed and
restarted.

```bash
python experiments/rerun_corrected_stage1.py   # explanation sweeps (RQ1/RQ2/RQ3 logs)
python experiments/rerun_corrected_stage2.py   # MIA + DP sweeps      (RQ4 logs)
```

Stage 1 (`rerun_corrected_stage1.py`), in order:

1. `experiments/phase10_explanation_sweep.py` — Wisconsin IID, 5 seeds, all encodings + amplitude non-IID → `phase10_log.csv`
2. `experiments/rq2_seed_extension.py` — Wisconsin IID seeds 11–47 (brings RQ2/RQ4 to n = 15) → `rq2_extension_log.csv`
3. `experiments/phase11_encoding_noniid_sweep.py` — Wisconsin angle/reuploading/IQP × {α=0.5, α=0.1, qty_skew} × 5 seeds → `phase11_log.csv`
4. `experiments/phase16_pima_explanation_sweep.py` — Pima, all encodings × all distributions → `phase16_pima_log.csv`

Stage 2 (`rerun_corrected_stage2.py`), in order (aborts if < 5 GB free disk; purges stale `/tmp/ray/session_*` between steps):

1. `experiments/phase16_pima_explanation_sweep.py` — resume tail of Stage 1
2. `experiments/phase13_mia_sweep.py` — Wisconsin IID direct + shadow MIA, 5 seeds → `phase13_mia_log.csv`
3. `experiments/rq4_seed_extension.py` — Wisconsin IID direct MIA, seeds 11–47 → `rq4_extension_log.csv`
4. `experiments/rq4_noniid_mia.py` — Wisconsin non-IID MIA (α=0.1, qty_skew) → `rq4_noniid_mia_log.csv`
5. `experiments/rq4_dp_extension.py` — Wisconsin IID calibrated-Laplace DP sweep; Δf auto-derived at import from the current `phase10_log.csv` → `rq4_dp_extension_log.csv`

Not part of the orchestrators (run individually if needed):

- `experiments/rq4_pima_mia_amplitude.py` — Pima IID amplitude MIA, n = 15 → `rq4_pima_mia_amplitude_log.csv`
- `experiments/rq_learning_curve.py` — federated-vs-centralized per-round accuracy gap (paper §6.4) → `rq_learning_curve_log.csv`
- `experiments/rq_classical_baseline_extension.py` — 13-param classical MLP control, 15 seeds, Wisconsin + Pima (paper §8.2) → `results/rq_classical_baseline_extended.csv`

After Tier C, re-run Tier A to rebuild the tables/figures, then Tier B for the PDF.

---

## Determinism notes

- Weight aggregation is client-ID-sorted (`federation/server.py::DeterministicFedAvg`) so FedAvg is bit-reproducible given the same client updates.
- Residual run-to-run variation is mini-batch scheduling across Ray actors, bounded at ≤ 6.1 pp (Wisconsin) / ≤ 3.2 pp (Pima) within-seed final-accuracy range (paper §7.5). Table-level means are stable across this.
- `DeterministicFedAvg.aggregate_fit` raises if any round does not train exactly partitions `{0,1,2,3}` — a reproduction that hits this has a Flower version whose `node_config` shape differs; pin `flwr==1.32.1`.

## What was corrected (and what was not)

An earlier pipeline derived the client's data-partition index from Flower's internal
node id (`int(context.node_id) % n_clients`), which is not a stable bijection onto
`{0..3}`; some runs trained two clients on one partition. **Every federated result in
the paper was recomputed** under the fixed `node_config["partition-id"]` mapping
(paper §6.7, `results/CORRECTED_RERUN_AUDIT.md`). Centralized baselines and the
classical control do not federate and were retained unchanged. The pre-fix snapshot
is kept at `results/prefix_archive_20260906T194218/` for audit only — do not cite it.
