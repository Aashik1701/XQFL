# XQFL Completion Audit

**Date:** 2026-09-04
**Scope:** Full project completion assessment, grounded in files on disk.
**Data sources read:** `analysis/methods_notes.md` (MN-01 through MN-13, 1862 lines), `analysis/phase9b_part_c.md` (381 lines), `Idea/project_status.md` (273 lines), full `results/` directory (29 CSVs, 8 LOGs, 27,688 total CSV rows), and full `experiments/` directory (43 Python scripts, 6 CSVs).

---

## Step 1 — Inventory of What Exists on Disk

### Results CSVs (29 files, 27,688 rows)

| File | Rows | Content |
|------|------|---------|
| `phase9b_results.csv` | 81 | Phase 9b accuracy sweep (4 encodings × 4 distributions × 5 seeds) |
| `phase10_log.csv` | 2,681 | Explanation sweep — 4 encodings × 5 seeds × IID + non-IID (Wisconsin) |
| `phase10_checkpoint.csv` | 56 | Phase 10 summary |
| `phase11_log.csv` | 8,168 | Full encoding × distribution × seed explanation matrix (Wisconsin) |
| `phase11_checkpoint.csv` | 118 | Phase 11 summary |
| `phase12_log.csv` | 457 | Collapse replication (Wisconsin) |
| `phase12_checkpoint.csv` | 7 | Phase 12 summary |
| `phase13_mia_log.csv` | 21 | MIA sweep — 4 encodings × 5 seeds × IID (Wisconsin) |
| `phase13_mia_checkpoint.csv` | 21 | Phase 13 summary |
| `phase16_pima_log.csv` | 11,671 | Full Pima explanation sweep — 4 encodings × 4 distributions × 5 seeds |
| `phase16_pima_checkpoint.csv` | 193 | Phase 16 summary |
| `phase16_smoke_log.csv` | 734 | Phase 16 smoke test |
| `phase16_smoke_checkpoint.csv` | 12 | Smoke test summary |
| `rq2_extension_log.csv` | 3,041 | 10-seed extension for RQ2 stability (Wisconsin) |
| `rq2_extension_checkpoint.csv` | 41 | RQ2 extension summary |
| `rq3_anomaly_replication_log.csv` | 3 | RQ3 anomaly replication (seed=123 amplitude) |
| `rq3_anomaly_replication_checkpoint.csv` | 3 | RQ3 anomaly summary |
| `rq4_extension_log.csv` | 41 | MIA seed extension to n=15 (Wisconsin) |
| `rq4_extension_checkpoint.csv` | 41 | RQ4 extension summary |
| `rq4_dp_extension_log.csv` | 81 | DP cross-encoding sweep (4 encodings × 4 ε × 5 seeds, Wisconsin) |
| `rq4_dp_extension_checkpoint.csv` | 81 | DP extension summary |
| `rq4_noniid_mia_log.csv` | 41 | Non-IID MIA sweep (Wisconsin, qty_skew + α=0.1) |
| `rq4_noniid_mia_checkpoint.csv` | 41 | Non-IID MIA summary |
| `rq4_seed7_replication_log.csv` | 7 | MIA seed=7 replication (Wisconsin) |
| `rq4_seed7_replication_checkpoint.csv` | 7 | Seed=7 replication summary |
| `rq4_iqp_seed7_qty_replication_log.csv` | 4 | IQP/seed=7/qty_skew replication |
| `rq4_iqp_seed7_qty_replication_checkpoint.csv` | 4 | IQP seed=7 qty summary |
| `angle_falsification_results.csv` | 16 | 5 G1 partitions × 3 draws angle falsification |
| `angle_falsification_checkpoint.csv` | 16 | Falsification checkpoint |

### Experiments (43 Python scripts)

Phases 0–16 plus targeted RQ scripts (rq2_seed_extension, rq3_anomaly_replication, rq4_*, phase_angle_falsification, etc.). All scripts exist and are non-empty (except `run_experiment.py` which is 0 bytes). Six legacy CSVs exist in `experiments/` from early phases (phase3, phase6, phase7).

### Analysis

- `methods_notes.md` — 1,862 lines, MN-01 through MN-13
- `phase9b_part_c.md` — 381 lines
- `aggregate_results.py` — 0 bytes (empty)
- `plots.py` — 0 bytes (empty)

### Other notable files

- `Idea/project_status.md` — 273 lines, current status document
- `Idea/dataset_usage.md`, `Idea/implementation.md`, `Idea/EXQL.md` — design docs
- `paper.doc` — 442 KB (existing paper draft)
- `Review2_Progress_Slides.md` — 19 KB
- `XQFL_Research_Progress_Report.docx` — 1.4 MB

---

## Step 2 — RQ-by-RQ Completion Status

### RQ1: Does federating a quantum model make its explanations worse?

**Current best answer:** Yes, federation introduces small but measurable explanation distortion (JSD > 0), and the effect is encoding- and distribution-dependent. Amplitude encoding maintains the best fidelity (lowest JSD).

**Evidence base:**

| Dataset | Encodings with JSD data | Seeds | Distributions | Centralized baselines |
|----------|------------------------|-------|---------------|----------------------|
| Wisconsin (Phases 10/11) | All 4 (angle, amplitude, reuploading, IQP) | 5 each | IID + α=0.5 + α=0.1 + qty_skew | Yes, all 4 |
| Pima (Phase 16) | Amplitude (5/5 Track-A IID), Reuploading (5/5 Track-A IID), Angle (0/5), IQP (1/5 seed=7 only) | 5 each | IID complete; non-IID partial | Yes |

**Per-encoding coverage:**
- **Amplitude:** 2-dataset confirmed. Wisconsin IID JSD ≈ 0.05–0.08 (from phase11_checkpoint); Pima IID JSD = 0.037 ± 0.025 (n=5). Non-IID: full Wisconsin matrix, partial Pima (α=0.5 n=3 Track-A, α=0.1 n=3 Track-A, qty_skew n=5).
- **Reuploading:** 2-dataset confirmed for IID. Wisconsin IID JSD ≈ 0.1–0.2; Pima IID JSD = 0.080 ± 0.066 (n=5, high variance). Non-IID Pima: α=0.5 6/13 Track-A draws, α=0.1 only 2/15 Track-A (too sparse), qty_skew 4/5.
- **Angle:** Wisconsin only. Pima: 0 Track-A under any distribution. No second-dataset confirmation possible.
- **IQP:** Wisconsin only. Pima: 1 Track-A seed (seed=7, unreliable per MN-09). No second-dataset confirmation.

**Confidence grading:** Amplitude = **High** (2-dataset, consistent pattern). Reuploading = **Medium** (2-dataset IID, but Pima noisier and non-IID sparse). Angle = **Medium** (1-dataset, complete matrix). IQP = **Medium** (1-dataset, complete matrix, seed=7 excluded).

**Are the confidence grades in `project_status.md` justified?**
- Amplitude High: **Justified.** Two datasets, consistent JSD pattern across distributions.
- Reuploading Medium: **Justified.** Pima IID JSD has std nearly as large as the mean.
- Angle/IQP: `project_status.md` does not assign explicit grades for these — implicit "Wisconsin only" is correct.

**What's missing to upgrade:**
- Angle and IQP each need a working model on a second dataset. This requires either a different dataset or architectural changes — not achievable by re-running existing experiments.
- Reuploading non-IID divergence pattern on Pima needs α=0.1 Track-A data (currently 2/15 draws, too sparse).

---

### RQ2: Which encoding keeps its explanations most stable over training rounds?

**Current best answer:** All four encodings are stable under IID (mean τ ≥ 0.86), with amplitude and angle slightly ahead of reuploading and IQP. The metric saturates at n=4 features.

**Evidence base:**

| Dataset | Encodings | Seeds | Distributions |
|----------|-----------|-------|---------------|
| Wisconsin Phase 10/11 | All 4 | 5 each | IID (primary); non-IID rounds available in logs |
| Wisconsin rq2_extension | All 4 | +10 (n=15 total) | IID only |
| Pima Phase 16 | Amplitude, Reuploading | 5 each | IID (Track-A only) |

**IID stability rankings (n=15 Wisconsin, from `project_status.md`):**
- amplitude: 0.939 ± 0.051
- angle: 0.933 ± 0.040
- reuploading: 0.879 ± 0.049
- IQP: 0.868 ± 0.076 (n=14, seed=7 excluded per MN-09)

**Per-encoding coverage:**
- Amplitude: 2-dataset IID stability confirmed.
- Reuploading: 2-dataset IID stability confirmed (Pima data exists in Phase 16 logs).
- Angle: Wisconsin only.
- IQP: Wisconsin only.

**Confidence grading:** Amplitude = **Medium-high**. Reuploading = **Medium**. Angle = **Medium**. IQP = **Medium** (with MN-09 caveat).

**Are the grades justified?** Yes, with the caveat that the formal per-encoding × per-distribution stability table "has not yet been extracted" (project_status.md). The data is in the CSVs but no write-up-ready table exists.

**What's missing:**
- A formal extracted stability table covering all 4 encodings × 4 distributions from Wisconsin Phase 10/11 logs. This is an analysis task, not an experiment.
- Pima non-IID stability for amplitude and reuploading (data exists in phase16_pima_log.csv but not extracted).
- The MN-01 caveat that τ saturates at n=4 features should be prominently noted; cosine similarity and L2 distance should be co-reported.

---

### RQ3: Do local and global explanations diverge more as data gets more uneven?

**Current best answer:** Moderate label skew (α=0.5) produces the most explanation drift; quantity skew produces minimal drift; severe label skew (α=0.1) is complicated by model collapse.

**Evidence base:**

| Dataset | Encodings with divergence data | Seeds | Distributions |
|----------|-------------------------------|-------|---------------|
| Wisconsin (Phases 10/11) | All 4 (full 4×3×5 matrix) | 5 each | α=0.5, α=0.1, qty_skew |
| Pima (Phase 16) | Amplitude (3/5 Track-A at α=0.5, 6/15 at α=0.1, 5/5 qty_skew); Reuploading (6/13 α=0.5, 2/15 α=0.1, 4/5 qty_skew) | 5 each | Same three |

**Key JSD_cent values from `project_status.md`:**

Wisconsin amplitude:
- IID ≈ 0.05 (baseline)
- qty_skew ≈ similar to IID
- α=0.5 ≈ elevated
- α=0.1 ≈ complicated by collapse

Pima amplitude:
- IID: 0.037 ± 0.025
- α=0.5: 0.068 (n=3)
- α=0.1: 0.029 (n=3)
- qty_skew: 0.046 (n=5)

**Per-encoding coverage:**
- Amplitude: 2-dataset confirmed. Pattern partially replicates (qty_skew low, α=0.5 elevated on both). α ordering on Pima reversed from Wisconsin (α=0.1 lower than α=0.5), attributed to small-n.
- Reuploading: Partial 2-dataset. α=0.5 and qty_skew usable on Pima; α=0.1 too sparse (2/15).
- Angle: Wisconsin only. 0/41 Track-A on Pima non-IID.
- IQP: Wisconsin only. Pima: 4/11 Track-A qty_skew only.

**Confidence grading:** Amplitude = **High** (2-dataset, consistent). Reuploading = **Medium-low** (Pima α=0.1 unusable). Angle = **Medium** (1-dataset, complete). IQP = **Medium** (1-dataset, complete).

**Are the grades justified?** Yes. The project_status.md grades are appropriately conservative.

**What's missing:**
- Formal divergence-by-distribution table extracted from Wisconsin Phase 10/11 CSVs. Analysis task, not experiment.
- Reuploading α=0.1 on Pima is structurally limited (too few Track-A draws) — not fixable without more seeds or a different approach.

---

### RQ4: Can someone use the explanation to figure out if a specific person was in the training data?

**Current best answer:** Yes. All four encodings leak membership information via direct MIA (AUC > 0.55). Reuploading leaks the most. DP protection requires ε ≤ 0.5 for uniform cross-encoding protection.

**Evidence base:**

| Analysis | Dataset | Encodings | Seeds | Distributions |
|----------|---------|-----------|-------|---------------|
| IID MIA (Phase 13 + extension) | Wisconsin | All 4 | 15 each (14 for IQP) | IID |
| DP sweep (rq4_dp_extension) | Wisconsin | All 4 | 5 each × 4 ε levels | IID |
| Non-IID MIA (rq4_noniid_mia) | Wisconsin | All 4 | 5 each | qty_skew, α=0.1 |
| Pima MIA | — | — | — | **Not tested** |

**Key numbers (Wisconsin, from `project_status.md`):**

Direct MIA AUC at n=15:
- angle: 0.545 ± 0.088
- amplitude: 0.619 ± 0.115
- reuploading: 0.644 ± 0.134
- IQP: 0.612 ± 0.171 (n=14)

DP protection at ε=0.1: all encodings → AUC 0.309–0.332 (near-random).
DP protection at ε=1.0: angle well-protected (0.421), others still leak (0.567–0.597).

Non-IID MIA (qty_skew): angle 0.546, amplitude 0.639, reuploading 0.692, IQP 0.722.
Non-IID MIA (α=0.1): angle 2/5 Track-A, amplitude 0.549 (5/5), reuploading 0/5 (all Track-B), IQP 0.674 (4/5 Track-A corrected).

**Confidence grading:** All four encodings = **Medium-high** for Wisconsin IID. Non-IID MIA completed on Wisconsin. DP sweep completed. Pima MIA = **absent**.

**Are the grades justified?** Yes. The Wisconsin coverage is genuinely comprehensive (n=15 seeds, 4 encodings, DP × 4 ε, non-IID MIA). The absence of Pima MIA is correctly noted as a gap.

**What's missing:**
- Pima MIA (both IID and non-IID). Priority: low per project_status.md, justified because most Pima conditions have sparse Track-A data.

---

## Step 3 — Data Coverage Matrix

This is the core gap-visibility artifact. Rows = (dataset × encoding × distribution) combinations. Columns = data types present in the CSVs on disk.

### Legend

- **C** = Complete (full 5-seed matrix present in CSVs)
- **P** = Partial (some seeds/draws present, not all)
- **A** = Absent (no data)
- Seeds = number of unique seeds with data on disk
- Draws = total replication draws per cell (where applicable)

### Accuracy Data

| Dataset | Encoding | IID | α=0.5 | α=0.1 | qty_skew | TrimmedMean/IID |
|----------|----------|-----|-------|-------|----------|-----------------|
| Wisconsin | angle | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Wisconsin | amplitude | C (5s) | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | reuploading | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Wisconsin | IQP | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Pima | angle | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Pima | amplitude | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Pima | reuploading | C (5s) | C (5s) | C (5s) | C (5s) | — |
| Pima | IQP | C (5s) | C (5s) | C (5s) | C (5s) | — |

All 32 accuracy cells are **Complete**. The accuracy matrix is fully populated for both datasets across all four encodings and all four distributions. This is the strongest part of the project.

### Explanation Vectors (Q-SHAP logged, Track-A seeds)

| Dataset | Encoding | IID | α=0.5 | α=0.1 | qty_skew |
|----------|----------|-----|-------|-------|----------|
| Wisconsin | angle | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | amplitude | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | reuploading | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | IQP | C (5s*, MN-09 excl seed=7) | C (5s) | C (5s) | C (5s) |
| Pima | angle | **A** (0/5 Track-A) | **A** (0 Track-A) | **A** (0 Track-A) | **A** (0 Track-A) |
| Pima | amplitude | C (5s) | P (3/5 Track-A) | P (3/5 Track-A) | C (5s) |
| Pima | reuploading | C (5s) | P (6/13 draws) | **A** (2/15 draws, too sparse) | P (4/5) |
| Pima | IQP | P (1/5 seed=7 only, MN-09) | **A** (0/5) | P (1/15, MN-10 noise) | P (4/11) |

Key gaps:
- **Pima angle explanation vectors: entirely absent.** 0 Track-A under any condition. Angle encoding collapses on Pima even under IID.
- **Pima IQP explanation vectors: effectively absent.** 1 unreliable Track-A seed under IID (MN-09 excluded), 0 under α=0.5, 1 marginal draw under α=0.1 (MN-10 non-reproducible), 4/11 under qty_skew only.
- **Pima reuploading α=0.1:** 2/15 draws is below any reasonable threshold for distribution characterization.

### MIA Data

| Dataset | Encoding | IID (n seeds) | qty_skew | α=0.1 |
|----------|----------|---------------|----------|-------|
| Wisconsin | angle | C (15s) | C (5s) | P (2/5 Track-A) |
| Wisconsin | amplitude | C (15s) | C (5s) | C (5s) |
| Wisconsin | reuploading | C (15s) | C (5s) | **A** (0/5 Track-A) |
| Wisconsin | IQP | C (14s, MN-09) | C (5s, seed=7 replicated) | P (4/5 Track-A corrected) |
| Pima | *all encodings* | **A** | **A** | **A** |

Pima MIA is **entirely absent** across all four encodings.

### DP Data

| Dataset | Encoding | ε=0.1 | ε=0.5 | ε=1.0 | ε=5.0 |
|----------|----------|-------|-------|-------|-------|
| Wisconsin | angle | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | amplitude | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | reuploading | C (5s) | C (5s) | C (5s) | C (5s) |
| Wisconsin | IQP | C (5s) | C (5s) | C (5s) | C (5s) |
| Pima | *all encodings* | **A** | **A** | **A** | **A** |

Pima DP is **entirely absent**.

### Seed Counts Summary

| Dataset | Accuracy seeds | Explanation seeds (Track-A) | MIA seeds | DP seeds |
|----------|---------------|----------------------------|-----------|----------|
| Wisconsin, amplitude | 5 × 4 dists | 5 × 4 dists | 15 IID + 5×2 non-IID | 5 × 4 ε |
| Wisconsin, reuploading | 5 × 4 dists | 5 × 4 dists | 15 IID + 5×2 non-IID | 5 × 4 ε |
| Wisconsin, angle | 5 × 4 dists | 5 × 4 dists | 15 IID + 2/5 α=0.1 | 5 × 4 ε |
| Wisconsin, IQP | 5 × 4 dists | 5 × 4 dists (seed=7 excl) | 14 IID + 5 qty + 4/5 α=0.1 | 5 × 4 ε |
| Pima, amplitude | 5 × 4 dists | 5 IID, 3 α=0.5, 3 α=0.1, 5 qty | **0** | **0** |
| Pima, reuploading | 5 × 4 dists | 5 IID, 6/13 α=0.5, 2/15 α=0.1, 4/5 qty | **0** | **0** |
| Pima, angle | 5 × 4 dists | **0** (all Track-B) | **0** | **0** |
| Pima, IQP | 5 × 4 dists | **0** (effectively) | **0** | **0** |

---

## Step 4 — Methods Notes Status

| MN | Status | Contradictions | Unresolved Caveats |
|----|--------|----------------|-------------------|
| **MN-01** (τ saturation at n=4) | **Closed.** Confirmed Phase 6, protocol adopted Phase 7+. No contradictions. | None | Metric limitation is inherent to 4-feature design; noted but unavoidable. |
| **MN-02** (tied accuracy, hard core vs stochastic tail) | **Closed.** Confirmed Phase 9a checks. Within-seed noise floor ±1.75pp established. | None | Noise floor applies only outside attractor-bifurcation regime (MN-06 extends). |
| **MN-03a** (seed=7 encoding×partition interaction) | **Partially confirmed.** Three-way split confirmed (hurt/neutral/helped). Mechanistic hypothesis flagged "PENDING CONFIRMATION" — needs non-IID per-encoding data to test. | None | Hypothesis explicitly unconfirmed. Part C is amplitude-only; cross-encoding non-IID comparison not available. |
| **MN-03b** (Pima collapse-vs-boundary) | **Closed.** Structural distinction (classical MLP CAN collapse, VQC CANNOT) is robust. Per-seed assignments not robust. | None | None. |
| **MN-04** (α=0.1 Dirichlet degeneracy) | **Closed.** Scope limitation confirmed. α=0.1 describes partition degeneracy, not pure label heterogeneity. | None | None — this is a design constraint correctly documented. |
| **MN-04a** (Wisconsin collapse predictor) | **RETRACTED as deterministic (Phase 10, MN-08).** Predictor is a risk indicator, not a rule. Seeds 42 and 999 produce COLLAPSED/FUNCTIONAL/MIXED across replicates. | **Contradicted by Phase 10.** Original "confirmed collapse" → "bifurcation-sensitive." Retraction correctly documented in MN-04a notice. | Retraction is complete and properly flagged. |
| **MN-04b** (Pima α=0.1 behavioral taxonomy) | **Partially confirmed.** Seeds 42/999 collapse patterns reliable; seeds 1 and 7 revised to "stochastically variable" after replicate checks. | **Seed=7 original label contradicted** by replication (39.61% did not reproduce). Correctly revised to "stochastic bifurcation." Seed=1 "transient escape→regression" shape not reliably reproduced; correctly downgraded. | Within-cell variance 21–30pp dwarfs any seed-level signal at α=0.1. |
| **MN-04c** (seed confound experiment) | **Partially confirmed.** Finding 1 (init determines attractor start) confirmed 12/12. Finding 2 (partition determines final outcome) WITHDRAWN — within-cell variance 21–30pp makes signal undetectable at n=3. | **Finding 2 withdrawn.** Original claim that partition=42 locks attractors is contradicted by replicate data showing same cell ranges 21–30pp. Retraction correctly documented. | Only Finding 1 is robust. Final accuracy at Pima/α=0.1 is dominated by actor-RNG stochasticity. |
| **MN-04d** (Pima α=0.5 taxonomy) | **Confirmed with revision.** Seed=7 "below baseline" revised to "partial regression, ≈baseline" (3 reps: 64.9–67.5%, range 2.6pp). | None | None — revision is complete. |
| **MN-05** (governing framework) | **Closed.** Reporting modes correctly prescribed per condition. | None | None. |
| **MN-06** (actor-RNG noise non-uniform) | **Confirmed.** Noise floor varies from 3.5pp (IID) to 21–30pp (attractor bifurcation). α=0.5 adversarial partition noise measured at 22.81pp. | None | None. |
| **MN-07** (explanation sweep design) | **Closed.** Runtime estimates validated by subsequent phases. Design decisions executed. | None | None. |
| **MN-08** (unified actor-RNG-determined outcomes) | **Confirmed.** Unified mechanism: actor-RNG basin selection at attractor boundaries. Correctly subsumes MN-04a retraction and MN-04c findings. Applies to both Wisconsin and Pima. | None | Mechanism is well-characterized but mitigation directions (seeding actor RNG, variance-scaled init) are not implemented — only the protocol (≥3 reps) is in place. |
| **MN-09** (IQP/seed=7 bimodal instability) | **Confirmed.** 16 draws across 4 distributions: 50% anomaly rate under balanced, near-persistent under label skew. Excluded from aggregations. | None | Seed=7 exclusion from IQP aggregations is a standing rule; correctly applied in RQ2/RQ3/RQ4 summaries. |
| **MN-10** (within-seed reproducibility rule) | **Closed.** Standing rule. Applied to IQP/Pima/α=0.1/seed=999 (1/3 draws crossed, 0/2 others reproduced → non-reproducible). | None | None. |
| **MN-11** (angle/Pima α=0.5 attractor draws) | **Closed.** Seeds 123 and 999 resolve to MN-04c and MN-08c respectively. No new finding needed. | None | None. |
| **MN-12** (angle purity-driven collapse) | **Closed.** Phase 11 "angle uniquely all-B" claim falsified. 100% purity → collapse confirmed (0/6); 91–93% purity → dataset-dependent (Wisconsin escapes, Pima doesn't). Threshold untested in 93–99% range. | **Phase 11 claim contradicted and retracted.** The falsification experiment correctly identified the artifact. | The 93–99% purity range is untested. Do not extrapolate. |
| **MN-13** (R0 initialization architecture) | **Partially confirmed.** Reuploading R0 input-sensitivity confirmed on Wisconsin (all 5 seeds). NOT confirmed on Pima (all encodings within 1–2 predictions of floor). Finding is **dataset-specific.** | None | Revised caveat correctly notes dataset-specificity. Mechanism (per-layer re-embedding) is plausible but the R0 advantage does not transfer to Pima. |

### Summary of unresolved contradictions

Two MNs carry explicit retractions:
1. **MN-04a** — collapse predictor retracted as deterministic (replaced by MN-08 stochastic framing)
2. **MN-04c Finding 2** — partition determines final outcome withdrawn (within-cell noise too high)

Both retractions are correctly documented and flagged in the notes. No MN contains a claim that a later phase contradicted without the contradiction being noted.

### MNs with untested generalizations

- **MN-03a:** The "single-pass vs repeated/full-Hilbert" hypothesis for seed=7's encoding×partition interaction is flagged PENDING CONFIRMATION. Non-IID per-encoding data does not exist.
- **MN-12:** The purity collapse threshold is untested between 93–99%. Only two purity levels tested (100% and 91–93%).
- **MN-13:** The reuploading R0 advantage is dataset-specific (confirmed on Wisconsin, absent on Pima). Generalizability unknown.

---

## Step 5 — Open Items Inventory

| # | Item | Why flagged | What closes it | Est. compute | Blocks paper? |
|---|------|-------------|----------------|-------------|---------------|
| 1 | **Formal RQ1 JSD-by-encoding-by-distribution table** | Data exists in CSVs but no write-up-ready table extracted | Script to aggregate phase10/11 and phase16 logs into formatted table | None (analysis only) | **Yes** — paper needs this table |
| 2 | **Formal RQ2 stability table (per-enc × per-dist)** | Same: data in CSVs, no extraction | Same extraction script | None | **Yes** — paper needs this table |
| 3 | **Formal RQ3 divergence table** | Same | Same extraction script | None | **Yes** — paper needs this table |
| 4 | **RQ4 DP/privacy write-up tables** | Data complete on Wisconsin but not formatted | Extraction from rq4_dp_extension CSV | None | **Yes** — paper needs this |
| 5 | **Non-IID MIA on Pima** | Never tested | New experiment: 4 encodings × 2 distributions × 5 seeds | ~30 min compute | **No** — optional; project_status.md correctly de-prioritizes |
| 6 | **DP sweep on Pima** | Never tested | New experiment: 4 encodings × 4 ε × 5 seeds | ~30 min compute | **No** — optional |
| 7 | **Reuploading α=0.1 Pima Track-A data** | 2/15 draws too sparse for JSD estimate | More seeds (20+) or acceptance that this cell is not characterizable | High (20+ runs) | **No** — correctly identified as structurally limited |
| 8 | **MN-03a non-IID per-encoding cross-check** | Encoding×partition hypothesis pending | Run angle/reuploading/IQP under α=0.5 on Wisconsin with explanation logging | ~20 min | **No** — nice-to-have for mechanistic story |
| 9 | **Purity threshold location (MN-12)** | Only 2 purity levels tested (100% and 91–93%) | 3–5 additional purity levels in 93–99% range | ~15 min | **No** — the binary "100% pure → collapse" finding is sufficient for the paper |
| 10 | **`aggregate_results.py` and `plots.py` are empty** | Analysis scripts declared but never written | Write and run extraction scripts | None | **Yes** — these are likely the scripts for items 1–4 |
| 11 | **Paper draft (`paper.doc`)** | Exists at 442 KB but project_status.md indicates sections are incomplete | Complete write-up across all sections | None (writing) | **Yes** — the actual paper |
| 12 | **MNIST / Fashion-MNIST / MedMNIST** | Stretch goal from original plan; never touched | Full pipeline run on new datasets | Very high (days) | **No** — correctly deferred |

### Items 1–4 are the critical path

The single highest-impact remaining work is extracting write-up-ready tables from existing CSVs. No new experiments are needed for the core RQs. Items 1–4 are all analysis/extraction tasks that could be completed in a single session. The empty `aggregate_results.py` and `plots.py` scripts suggest this work was planned but not executed.

---

## Step 6 — Paper-Readiness Check

Assuming a standard paper structure (Introduction → Related Work → Method → Experiments → Results per RQ → Discussion → Conclusion → Future Work):

| Section | Underlying data | Status |
|---------|----------------|--------|
| **Introduction** | N/A (framing) | Needs writing |
| **Related Work** | Literature_Review.docx exists (30 KB) | Partially exists |
| **Method** | Architecture diagram exists (1.3 MB PNG); implementation.md (18 KB) | Partially exists |
| **Dataset description** | dataset_usage.md exists (3.5 KB); two datasets used | Partially exists |
| **Experimental setup** | Phase descriptions in methods_notes.md | Partially exists |
| **RQ1 results — fidelity table** | JSD data in phase10/11/16 CSVs | **Data exists; table not extracted** |
| **RQ1 results — encoding comparison** | Same | **Data exists; table not extracted** |
| **RQ1 results — distribution effect** | Same, per-distribution JSD | **Data exists; table not extracted** |
| **RQ2 results — stability table** | τ, cosine, L2 in phase10_log.csv, rq2_extension_log.csv | **Data exists; table not extracted** |
| **RQ2 results — encoding ranking** | Same | **Data exists; table not extracted** |
| **RQ2 Fig: stability over rounds** | Round-by-round data in CSVs | **Data exists; figure not produced** |
| **RQ3 results — divergence table** | JSD_cent per distribution in phase10/11/16 | **Data exists; table not extracted** |
| **RQ3 Fig: divergence vs non-IID level** | Same | **Data exists; figure not produced** |
| **RQ4 results — MIA table** | rq4_extension, rq4_noniid_mia CSVs | **Data exists; table not extracted** |
| **RQ4 results — DP table** | rq4_dp_extension CSV | **Data exists; table not extracted** |
| **RQ4 Fig: DP AUC vs ε** | Same | **Data exists; figure not produced** |
| **Collapse behavior discussion** | MN-04 through MN-08, phase9b_results.csv | Data exists; narrative in methods_notes |
| **Attractor failure analysis** | MN-04c, MN-06, MN-08 | Data exists |
| **Initialization architecture (MN-13)** | R0 data in phase10/11/16 round=0 rows | Data exists |
| **Limitations section** | Phase 9b Part C §5; MN-04, MN-06 | Data exists |
| **Future Work** | N/A | Needs writing |
| **Table 1: Dataset characteristics** | Two datasets, documented | Available |
| **Table 2: Encoding descriptions** | 4 encodings, documented | Available |
| **Table 3: RQ1 fidelity** | — | **Needs extraction** |
| **Table 4: RQ2 stability** | — | **Needs extraction** |
| **Table 5: RQ3 divergence** | — | **Needs extraction** |
| **Table 6: RQ4 MIA** | — | **Needs extraction** |
| **Table 7: RQ4 DP** | — | **Needs extraction** |
| **Figure 1: Architecture** | Architecture Diagram.png exists | Exists |
| **Figure 2: Stability over rounds** | Round data in CSVs | **Needs production** |
| **Figure 3: Divergence by distribution** | JSD data in CSVs | **Needs production** |
| **Figure 4: MIA AUC comparison** | MIA data in CSVs | **Needs production** |
| **Figure 5: DP AUC curves** | DP data in CSVs | **Needs production** |

**Summary:** Every core results table and figure has its underlying data in the CSVs. None of the tables or figures have been produced yet. The paper cannot be written in its current state because the extraction and formatting step has not been done. However, this is an analysis task, not an experiment — no new compute is needed.

---

## Step 7 — Honest Bottom Line

### (a) Could the paper be written today with what exists?

**No — but not for lack of data.** The experimental data is substantially complete for a two-dataset paper on amplitude and reuploading, and a one-dataset paper on angle and IQP. What does not exist is any formatted results table or figure. The CSVs contain the raw numbers, but `aggregate_results.py` and `plots.py` are both empty (0 bytes). A paper cannot be written until the data is extracted from 29 CSV files into structured tables.

**What the paper would have to caveat:**
1. Angle and IQP have no Pima explanation data — these two encodings are Wisconsin-only for all explanation questions. This must be stated upfront, not buried.
2. Pima reuploading α=0.1 Track-A data is too sparse (2/15 draws) for a reliable JSD estimate. This cell cannot be meaningfully reported.
3. The α=0.1 results on both datasets are confounded with partition degeneracy (MN-04) — they describe what happens when the effective federation shrinks to 1–2 clients, not pure label heterogeneity.
4. The Pima MIA and DP analyses were never run. RQ4 on Pima is entirely absent.
5. The Pima/angle and Pima/IQP models collapse under all tested conditions. These encodings cannot contribute explanation results from Pima.
6. Every result in the attractor-bifurcation regime (Pima/amplitude α=0.1, some α=0.5 seeds) carries 21–30pp within-cell uncertainty, not ±1.75pp.

### (b) What is the single highest-value remaining experiment?

**There are no remaining experiments needed for the core paper.** The single highest-value remaining *task* is writing the extraction scripts (`aggregate_results.py` / `plots.py`) to produce the 5 results tables and 4 figures from existing CSVs. This is the only thing between the current state and a writeable paper.

If an additional experiment were to be run despite diminishing returns, the highest value would be **Pima IID MIA for amplitude** — it would add a second-dataset privacy finding for the strongest encoding. Estimated: ~10 minutes. But this is not necessary for a complete paper; the Wisconsin RQ4 story is self-sufficient.

### (c) What is currently the weakest claim in the project?

**The weakest claim is that reuploading's Pima non-IID divergence pattern is characterizable.** The α=0.1 data has 2 Track-A draws out of 15 — this is not a sample from which any distributional claim can be made. Including this cell in a divergence-by-distribution table would require either (a) fabricating a mean from 2 draws (not acceptable), (b) reporting it as "insufficient data" (honest but uninformative), or (c) dropping the cell entirely.

The claim is weak enough to need **caveating, not removal.** The correct framing: "Reuploading's explanation fidelity under severe label skew on Pima cannot be characterized due to near-total model collapse (2/15 Track-A draws). This is itself a finding: reuploading's explanation vulnerability under extreme non-IID is more severe on Pima than on Wisconsin, where the same condition produced usable data for all 5 seeds." The sparseness is data, not a gap.

**The second-weakest claim** is that the α ordering of divergence (α=0.5 > qty_skew > α=0.1 on Pima amplitude) is a real pattern rather than a small-n artifact. With n=3 Track-A seeds for α=0.5 and α=0.1, the ordering could easily flip with more data. The Wisconsin data (n=5) shows a consistent pattern; the Pima data (n=3) is suggestive but not confirmed.

---

## Appendix: Contradictions Found During Audit

The audit instructions said to report contradictions rather than resolving them silently. Two were found:

1. **`project_status.md` claims Phase 10/11 "a write-up-ready summary table has not yet been extracted" for RQ1 and RQ3.** This is accurate — no such table exists in any file on disk. The extraction is an open task.

2. **MN-08 (unified mechanism) and Phase 9b Part C §4.1 present the same finding with slightly different framing.** Phase 9b Part C was written before MN-08 and contains the original "two mechanistically distinct patterns" framing. MN-08 correctly retracts this. Phase 9b Part C was never updated to reflect MN-08's retraction — it still contains the original framing in its introduction (which references MN-08) and section 4.1 (which has the revision). The file is internally inconsistent: the limitations section (§5, item 4) says "Part C is amplitude-only" which is correct, but the cross-cutting analysis (§4.1) has been updated with MN-08's unified account. A reader of only the early sections of Phase 9b Part C would get the pre-MN-08 framing. This is a documentation inconsistency, not a scientific one — MN-08 is the authoritative version.

---

*End of audit.*
