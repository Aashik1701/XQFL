# Hostile Reviewer Audit — XQFL

**Manuscript:** `XQFL_paper.md` (post-partition-fix revision, 2026-09-07)
**Authoritative outputs:** `results/outputs/table{1..5}.{csv,md}`, `results/outputs/figure{2..5}.png`, regenerated 2026-09-07 16:45 by `analysis/aggregate_results.py` + `analysis/plots.py`
**Raw logs:** `results/phase10_log.csv`, `phase11_log.csv`, `phase16_pima_log.csv`, `rq2_extension_log.csv`, `phase13_mia_log.csv`, `rq4_extension_log.csv`, `rq4_noniid_mia_log.csv`, `rq4_dp_extension_log.csv`, `rq4_pima_mia_amplitude_log.csv`
**Reviewer stance:** skeptical peer reviewer for a Q1 venue. Every claim is treated as guilty until traced to evidence.

---

## Overall Verdict

**Publishability:** Publishable as a *characterization study* after the manuscript changes listed in "Required Changes Before Submission". Not publishable as a claims-heavy "quantum encoding X is better" paper — the evidence does not support that framing, and the current manuscript mostly (not entirely) avoids it.

**Strongest aspect:** The methodological hygiene around collapse. The Track A / Track B split, the explicit treatment of near-zero attribution vectors as a silent failure mode for JSD, the classical parameter-matched control (§8.2), and the disclosed-and-corrected partition-assignment defect (§6.7) together make the pipeline unusually auditable for this subfield. The partition fix is real, is in the code, and is enforced at runtime (`federation/server.py` raises if any round does not train exactly `{0,1,2,3}`).

**Biggest vulnerability:** The RQ1 fidelity metric has a resolution floor that the paper measures itself (§5.5: two independent Shapley estimators disagree by JSD ≈ 0.171 on the *same* model) but then does not carry into the RQ1 results or limitations. Most of the Wisconsin between-encoding and between-distribution JSD_cent differences the paper interprets (0.08–0.25) sit at or below that floor. The one fidelity comparison that clearly clears it — amplitude 0.201 vs re-uploading 0.335 on Pima IID — is also the paper's headline, so the core result survives, but the surrounding Table-1 interpretation is over-resolved.

**Submission-ready?** **NOT YET — MINOR REVISION.** No result needs to be recomputed. Four manuscript changes (estimator floor, RQ3 survivorship, MIA caveats, per-split threshold) and the reproducibility package close the gap. None of them overturn a headline claim.

---

## Major Concerns

### MC-1 — RQ1 fidelity metric resolution floor is measured but not applied
- **Severity:** MAJOR
- **Issue:** §5.5 reports that Q-SHAP vs `shap.KernelExplainer` on the *identical* model, background, and samples disagree at JSD = 0.171 (cosine 0.978). That is the noise floor of the fidelity metric from estimator choice alone. Table 1's Wisconsin cells span JSD_cent 0.08–0.25, and §7.1 draws distribution-level conclusions from within that band ("divergence grows under label skew … 0.15 to 0.25", amplitude qty_skew 0.104 vs α=0.5 0.195). Those differences are not resolvable against an estimator floor of ≈0.17.
- **Evidence:** `XQFL_paper.md` §5.5 (JSD 0.171); Table 1 (`results/outputs/table1_rq1_fidelity.csv`). Wisconsin amplitude IID 0.152 ± 0.036 vs α=0.5 0.195 ± 0.040: gap 0.043, below the floor. Pima amplitude IID 0.201 ± 0.101 vs Pima re-uploading IID 0.335 ± 0.087: gap 0.134, above the floor and with non-overlapping ±1 SE bands (SE 0.045 vs 0.039) — this one is real.
- **Affected:** Abstract; §1.5 contribution 2; §7.1; Table 1 interpretation; §8.6; Appendix Table 6 RQ1 row.
- **Action:** **A (manuscript change).** Add the floor to §7.1 and §8.6. Restrict RQ1 interpretation to (i) "all functional encodings sit in a low, tight JSD_cent band on Wisconsin (0.08–0.25), comparable to the estimator's own cross-implementation disagreement" and (ii) the amplitude-vs-re-uploading Pima gap, which exceeds the floor. Stop interpreting sub-0.1 Wisconsin differences as distributional effects in RQ1 (they belong to RQ3, which uses a different and better-resolved metric).

### MC-2 — RQ3 α=0.5 > α=0.1 ordering is consistent with survivorship, not dose-response
- **Severity:** MAJOR
- **Issue:** For angle and amplitude the local-to-global divergence is *larger* at moderate skew (α=0.5) than at severe skew (α=0.1): angle 0.619 vs "insufficient" (1/5 Track-A); amplitude 0.461 vs 0.446. A monotone "more skew → more divergence" story does not predict a peak at the middle setting. The parsimonious explanation is selection: at α=0.1 the seeds whose clients diverge most from the global model are also the seeds that collapse to Track B (collapse rate: angle α=0.1 0.80, amplitude α=0.1 0.40, reuploading α=0.1 0.60) and are removed from the Track-A mean. The reported α=0.5 "peak" is then partly an artifact of which seeds survive filtering.
- **Evidence:** `results/outputs/table3_rq3_divergence.csv` (n_track_a_seeds and collapse_rate columns); `XQFL_paper.md` §7.3.
- **Affected:** Abstract ("moderate label skew … drives the largest …"); §1.5 contribution 5; §7.3; Figure 3; §8.1; §8.4; Appendix Table 6 RQ3 row.
- **Action:** **A + B.** Keep the Wisconsin observation (α=0.5 shows the largest Track-A local-to-global divergence, quantity skew ≈ IID) — it is on the correct metric and the quantity-skew half is clean (0.00 collapse). But state explicitly that the non-monotonic drop at α=0.1 is *expected under Track-A selection* and that RQ3 therefore cannot claim a monotone relationship between skew severity and divergence. The defensible claim is: "among functional runs, moderate label skew is the condition with the largest local-to-global explanation divergence; quantity skew is indistinguishable from IID." Do not imply severe skew is *less* divergent.

### MC-3 — Direct-MIA construction: preprocessing-in-CV leakage and asymmetric explanation backgrounds
- **Severity:** MAJOR (construct validity), but headline claims survive
- **Issue (a):** `privacy/membership_inference.py::run_mia_attack` calls `StandardScaler().fit_transform(X)` on the full stacked 100-row (50 member + 50 non-member) matrix *before* `cross_val_score`. Every CV fold's scaler has therefore seen the held-out fold. At n=100, d=4 the inflation is small and applies uniformly across encodings and seeds, so the *ranking* and the *angle-at-chance* result are unaffected, but the absolute AUCs (Table 4, Table 5) are mildly optimistic and this is undisclosed.
- **Issue (b):** Member explanations are computed with a training-set background (`_qshap_for_samples(model, X_train, …)` → background = `X_train[:20]`); non-member explanations with a test-set background (`X_test[:20]`). The two classes differ in the *explanation procedure*, not only in membership. Some of the measured AUC could be train/test input-distribution shift propagated through the background set rather than model memorization of members. Angle sitting at chance (0.533, 1.6 SE) bounds this effect as small, but it is a threat to what "membership-inference AUC" means here.
- **Evidence:** `privacy/membership_inference.py` lines 28–48 (`_qshap_for_samples`, background = first `n_bg` of the same array) and 222–240 (`run_mia_attack`, scaler outside `StratifiedKFold`). `experiments/phase13_mia_sweep.py` lines 220–240. Manuscript §6.5, §7.4.
- **Affected:** §6.5; §6.5b; §7.4; Tables 4–5; §8.1; §8.6; Appendix Table 6 RQ4 row; Abstract and §9 MIA sentences.
- **Action:** **A (manuscript change), no recompute required.** The paper already frames direct-MIA as a lower-bound signal-presence test and notes the shadow attack fails; it needs two added caveats: (1) attack-classifier standardization is fit on the pooled member/non-member set, so absolute AUCs are a mild upper estimate — cross-encoding comparison and the angle-vs-chance test are unaffected; (2) member/non-member explanations use train/test backgrounds respectively, so a component of the AUC may reflect input-distribution shift; the fact that angle remains at chance is the empirical bound on that component. If the authors want to *remove* the caveat rather than disclose it, that requires a re-run with (i) a `Pipeline` inside the CV and (ii) a common fixed background for both classes — this is a ~2–3 h job on the existing 20 IID models and is the single re-run I would consider defensible. It is not mandatory for publication if the caveats are stated.

---

## Minor Concerns

### mc-4 — RQ2 ranking is carried by a near-saturated metric
- **Severity:** MINOR
- **Issue:** §7.2 and Figure 2 show τ reaching ~1.0 within 2 rounds for all encodings. The 0.936 (amplitude) vs 0.858 (IQP) spread is therefore driven by the first one or two round-to-round transitions and by the minority of seeds that never saturate. The paper acknowledges the four-feature ceiling and corroborates with cosine/L2 (adequate), but the *interpretation* — "which encoding keeps its attributions most stable across rounds" — is really "which encoding's dominant component locks in fastest and cleanest".
- **Action:** **A.** One sentence in §7.2 reframing the ranking as lock-in speed/cleanliness rather than whole-trajectory stability. No number change. The two-tier structure and the >2 SE gap are genuine (`table2_rq2_stability.csv`: reuploading −4.9 pp at SE 0.017; IQP −7.7 pp at SE 0.026).

### mc-5 — Collapse threshold is per-split, presented as fixed
- **Severity:** MINOR
- **Issue:** `experiments/phase10_explanation_sweep.py:99` computes `majority_baseline = bincount(y_test).max() / len(y_test)` per run, so the Track-A threshold varies with the seed's stratified test split (≈62–66% Wisconsin, ≈63–65% Pima). The manuscript (§4.3) quotes 65.16% / 66.94% as if fixed. Those correspond to the seed-42 split (Wisconsin 63.16% + 2 pp; Pima 64.94% + 2 pp, matching `phase16_smoke_test.py:7`).
- **Action:** **A.** §4.3: note the threshold is the per-split majority rate + 2 pp; the quoted values are representative (seed 42). No result changes — the code already does the right per-split thing.

### mc-6 — Stale code comments/docstrings (behaviour is correct)
- **Severity:** MINOR
- **Issue:** `analysis/aggregate_results.py::rq3_local_global_cell` docstring still lists "MN-09: Wisconsin IQP/seed=7 excluded" and the RQ1 anomaly exclusions as active rules, though `IQP_SEED7_EXCLUDE_DATASET = None` and `RQ3_EXTRA_EXCLUSIONS = {}` disable them (verified: IQP Wisconsin IID is n=15 in `table2`/`table4`). `experiments/rq4_dp_extension.py` docstring shows Δf = 0.2258 / 0.2607 / 0.3547 / 0.2536 as the calibration, but the runtime `_compute_sensitivity()` derives 0.229 / 0.262 / 0.344 / 0.287 from the corrected `phase10_log.csv` (and the manuscript §6.5b uses the derived values). A reviewer reading the code will think a retired exclusion is live.
- **Action:** **A (code-comment cleanup, no logic change).** Update the two docstrings to match current behaviour.

### mc-7 — n = 3 cells reported with three-decimal means
- **Severity:** MINOR
- **Issue:** Tables 1 and 3 print e.g. Pima IQP IID 0.257 ± 0.062 (n=3), Wisconsin angle α=0.1 0.153 ± 0.016 (n=3), Wisconsin reuploading/IQP α=0.1 RQ3 (n=3/4). The `MIN_VALID_DRAWS = 3` cutoff is itself thin, and 3-decimal precision oversells them.
- **Action:** **B (analysis clarification) / optional A.** Either raise the "insufficient" cutoff to 4, or add a "n ≤ 4: indicative only" footnote to both tables. The manuscript already calls the Pima RQ3 skewed cells "inconclusive"; extend that discipline to every n ≤ 3 mean it prints.

### mc-8 — Revision-brief numbers vs authoritative CSVs
- **Severity:** MINOR (documentation only)
- **Issue:** The Phase 1 "scientific freeze" list in the revision brief gives IQP Wisconsin IID MIA = 0.632 and IQP DP ε=1.0 = 0.551. The authoritative regenerated CSVs and the raw logs give **0.647** (`phase13_mia_log.csv` + `rq4_extension_log.csv`, 15-seed mean = 0.64693) and **0.588** (`rq4_dp_extension_log.csv`, seeds 1/7/42/123/999 at ε=1.0 → 0.5884). The manuscript currently uses 0.647 and 0.588 — i.e. it correctly follows the CSVs, not the brief.
- **Action:** **D (no action) — flagged so a later pass does not "correct" the manuscript backwards toward the brief.** Authoritative outputs win.

### mc-9 — Reproducibility gaps
- **Severity:** MINOR (blocks the repro package, not the science)
- **Issue:** `README.md` is a 7-byte stub. No `references.bib` (references are hand-numbered in the Markdown — fragile, and a reviewer cannot check them mechanically). No single documented entry point to reproduce Tables 1–5 / Figures 2–5 (the chain is `experiments/rerun_corrected_stage{1,2}.py` → `analysis/aggregate_results.py` → `analysis/plots.py`, but this is only in session notes). No PDF build script; the repo's `XQFL_paper.pdf` predates the revision.
- **Action:** **A/B — Phase 7 package.** README with the twelve required questions answered, an explicit "reproduce the tables" command block, and the PDF rebuilt from the current `.md`.

---

## RQ1 Audit — Fidelity

**Claim:** "Federating a quantum model does not by itself change its feature attributions under balanced client data; amplitude has the lowest two-dataset divergence (Pima IID JSD_cent 0.201 ± 0.101)."

- **Metric validity:** Final-round JSD between federated-global and centralized global attribution vectors, Track-A only, on the shared test set. This does answer RQ1 as stated. Direction is correct (lower = more faithful). Hidden assumption: the centralized 45-epoch model is the right reference — §6.4 defends this as a gradient-budget match and concedes it does not match update *diversity*; acceptable and disclosed.
- **Statistical support:** Pima amplitude IID n=5, JSD 0.201 ± 0.101 (SE ≈ 0.045). The ±std nearly spans 0.10–0.30; "0.201" implies precision that n=5 does not carry. The *comparative* claim (amplitude < re-uploading on Pima, gap 0.134) is sound: SE 0.045 vs 0.039, gap > 2× the pooled SE, and the gap exceeds the §5.5 estimator floor. The Wisconsin cell-to-cell differences mostly do not clear the floor (MC-1).
- **Track filtering:** Correct and necessary. Collapsed models have ‖φ‖→0 and JSD→0, which would masquerade as perfect fidelity; §5.6 and §7.1 both state this. Consistently applied in `jsd_cell`.
- **Verdict:** Core claim **holds** with MC-1 applied. The two-dataset amplitude result is the paper's most transferable finding and it is defensible. The distribution-level RQ1 reading in §7.1 is over-resolved and must be trimmed.

## RQ2 Audit — Stability

**Claim:** "Amplitude ≈ angle (τ 0.94) > re-uploading (0.89) ≈ IQP (0.86); two-tier, between-tier gap > 2 SE; n = 15 Wisconsin IID."

- **Metric validity:** Mean consecutive-round Kendall τ of the global attribution vector. Answers RQ2 literally. But the four-item τ saturates fast (mc-4): the ranking reflects lock-in behaviour more than sustained stability.
- **Statistical support:** Verified against `table2_rq2_stability.csv`: amplitude 0.9363 ± 0.0556, angle 0.9348 ± 0.0456 (gap 0.15 pp — correctly called a tie); reuploading 0.8859 ± 0.0489 (−4.9 pp, gap/SE ≈ 2.8); IQP 0.8578 ± 0.0915 (−7.7 pp, gap/SE ≈ 3.0). The two-tier split is real at 2 SE. n = 15 confirmed (5 phase10 seeds + 10 `rq2_extension` primes).
- **Corroboration:** Cosine (0.93–0.96) and L2 columns move with τ and are not ceilinged — this is what rescues the ranking from the τ saturation objection.
- **Pima:** amplitude 0.853, reuploading 0.840, IQP 0.800 (n=3); angle Track-B throughout. Reported with the right hedges.
- **Verdict:** **Holds.** Add the lock-in reframing (mc-4). No number issues.

## RQ3 Audit — Local-to-Global Divergence

**Claim:** "On Wisconsin, moderate label skew (α=0.5) produces the largest local-to-global explanation divergence; quantity skew stays close to IID. Pima amplitude is flat/inconclusive."

- **Metric validity:** `rq3_local_global_cell` (analysis/aggregate_results.py:243) computes the mean over rounds t ≥ 1 and clients k of `JSD(φ_k^(t), φ_g^(t))` — per seed, then mean of per-seed means, Track-A seeds only (final-round global track class), round 0 skipped. **This is the Methods-defined RQ3 metric** (§4.4) and is genuinely distinct from RQ1's JSD_cent. The prior `rq3_view()` (which re-sorted RQ1 values) is gone. Confirmed.
- **Statistical support:** `table3_rq3_divergence.csv` verified cell-by-cell against §7.3 and Table 3 — all match to 3 dp, all Δ-vs-IID values match. Wisconsin α=0.5: angle +0.181, amplitude +0.173, reuploading +0.070, IQP +0.037. Quantity skew: |Δ| ≤ 0.031 for all four, 0.00 collapse — the clean half of the result.
- **Threat:** MC-2 (survivorship drives the α=0.5 > α=0.1 non-monotonicity). Also: several skewed cells are n = 3 (mc-7). The Wisconsin angle α=0.5 cell is n = 3 Track-A of 5, so its +0.181 (the largest single effect in the paper) rests on three seeds.
- **Pima:** amplitude flat (IID 0.367 → α=0.5 0.325 → α=0.1 0.291), explicitly *below* IID — reported as inconclusive at 3 Track-A seeds, which is the honest call. It does not replicate Wisconsin and the paper says so.
- **Verdict:** **Holds as a Wisconsin-only, functional-runs-only statement** with MC-2 applied. The "quantity skew ≈ IID" half is the strongest part and is clean. Do not generalize to "heterogeneity causes divergence" or to a monotone dose-response.

## RQ4 Audit — Privacy

**Claim:** "Direct MIA AUC 0.53 (angle) → 0.67 (re-uploading) at n=15; amplitude/IQP/re-uploading each resolvably above angle; angle not resolved above chance; no single Laplace budget brings all four to near-chance."

- **Metric validity:** 5-fold CV logistic regression separating 50 member from 50 non-member Q-SHAP vectors. Reasonable as a *signal-presence* proxy. Not a realistic-attacker estimate (the attacker is handed labelled members) — the paper mostly frames it correctly but should be sharper. MC-3 (a) and (b) are the construct-validity issues.
- **Statistical support:** verified against `table4_rq4_mia.csv` and raw logs. Wisconsin IID n=15: angle 0.533 ± 0.080, amplitude 0.620 ± 0.112, IQP 0.647 ± 0.158, reuploading 0.665 ± 0.137. Gaps vs angle: +8.7 pp (2.5 SE), +11.4 pp (2.5 SE), +13.3 pp (3.2 SE) — all resolved. vs chance: angle 1.6 SE (**not** resolved — correctly stated), others 3.6–4.7 SE. Shadow attack means 0.455–0.554 (fails) — verified.
- **DP sweep:** `table5_rq4_dp.csv` verified. ε=0.1: all 0.311–0.331. ε=1.0: angle 0.415, others 0.562–0.588. Monotone in 19/20 encoding-seed pairs (angle seed 123 dips 0.002 at ε=5.0). Δf calibration = mean ‖φ‖ over IID Track-A global rounds 1–15, derived at runtime (0.229/0.262/0.344/0.287) — matches the `sensitivity` column in `rq4_dp_extension_log.csv` and §6.5b. **Not** a certified (ε,δ) bound — §6.5b and §8.6 state this clearly and no "protects/guarantees" language survives (grep-verified).
- **Non-IID (Wisconsin, prose only):** qty_skew means angle 0.556 / amplitude 0.643 / reuploading 0.700 / IQP 0.700; qty > α=0.1 for amplitude (+7.0) and reuploading (+4.9), **not** IQP (−0.1). The retired "IQP leads quantity skew (0.722)" claim is gone. reuploading α=0.1 now measurable (4/5, 0.651). All verified against `rq4_noniid_mia_log.csv`.
- **Pima:** amplitude IID only, 0.564 ± 0.109, n=15 — resolved above chance (2.3 SE), tied with Wisconsin amplitude. Correctly not used to sharpen the ranking. Pima non-IID MIA and all Pima DP untested — stated as a limitation.
- **Verdict:** **Holds** with the MC-3 caveats added. The angle-at-chance result is the load-bearing privacy claim and it is robust to both MC-3 effects (a uniform inflation and a bounded distribution-shift term both leave angle at chance).

---

## Statistical Audit

| Question | Finding |
|---|---|
| Is n=15 adequate for the IID claims? | Yes for the two-tier RQ2 split and the MIA-vs-angle gaps (all > 2 SE). Marginal for angle-vs-chance (deliberately reported as unresolved). |
| Are mean ± std used appropriately? | Mostly. Tables print 3 dp for n=3 cells (mc-7). "±" is SD not SE throughout — consistent, but the text sometimes slides between them; the "2 SE" resolution rule is applied correctly where invoked. |
| Are "resolved/significant" claims justified? | Yes where stated. Spot-checked every "resolved" in §7.2 and §7.4 against the CSVs — all correct. The paper is careful to say "not resolved" for angle-vs-chance and for the Pima amplitude MIA gap. |
| Effect sizes vs uncertainty communicated honestly? | Adequate, will be good after MC-1 (RQ1 floor) and mc-7. |
| Small-n Pima RQ3 explicitly limited? | Yes — "inconclusive rather than a contradiction", and §8.6 limitation. Extend to Wisconsin n=3 cells. |
| Multiple comparisons? | No correction applied across the ~30 Table-1 cells / ~26 Table-3 cells. Given the paper now interprets only a handful of differences and hedges the rest, formal correction is not required, but §6.6 should acknowledge that per-cell tests are not family-wise controlled. |

---

## Methodological Threats (could a result be an artifact of …)

| Candidate artifact | Assessment |
|---|---|
| PCA→4 / four-feature attribution | **Partly, and disclosed.** τ ceiling (§7.2), "PCA components are not clinical features" (§8.6), classical control shares the same 4-D space and shows the low-JSD regime is not automatic (§8.2). Fair. |
| 2-layer, 13-param VQC | **Yes for the collapse results** — §8.2 Pima classical control collapses the same way, so "capacity, not encoding" is the right read and the paper makes it. §8.3 and §8.6 both frame collapse as circuit-size-bound. |
| Q-SHAP 8-permutation estimator | **Bounded.** §5.5 exact-24 check (τ=1.0, JSD 0.061) and independent-estimator check (JSD 0.171). The latter *is* the MC-1 floor. |
| Local epochs (3) / rounds (15) / FedAvg | Not stress-tested (trimmed-mean reduces to FedAvg at 4 clients — §5.4 concedes this). Single aggregation rule is a stated limitation. |
| Seed selection | 15 IID seeds = 5 original + 10 primes; 5 for other cells + extra draws for high-collapse label-skew cells. Extra draws are pooled into the mean (Table 1 "n Track-A draws" ≠ distinct seeds) — this is disclosed in the Table 1 note but is an unusual construction a reviewer will question; the distinct-seed count is in Table 3. Acceptable given the disclosure. |
| Track-A filtering | **This is MC-2.** Conditioning on non-collapse is necessary for the metric to mean anything, but it induces selection that the RQ3 α-ordering does not fully account for. |
| Partition assignment | **Fixed and verified.** `node_config["partition-id"]` + range check (`run_federation.py:227`) + per-round `{0,1,2,3}` assertion in `DeterministicFedAvg.aggregate_fit` (`server.py:39`), live on every logged run via `LoggingFedAvg`. `CORRECTED_RERUN_AUDIT.md` records that the assertion never fired across ~435 retrains. |

---

## Federated Validity

- **Partition correction:** verified in source. The old `int(context.node_id) % _n_clients` is gone; the replacement reads Flower's per-client `partition-id`, hard-errors if absent, range-checks, and uses that same index as the logged `client_id`.
- **Every round sees {0,1,2,3}:** enforced at runtime (raises otherwise). Not merely asserted in prose.
- **No hidden duplication/drop:** the server guard counts distinct `client_id`s per round and requires exactly `range(min_fit_clients)`. A failed/dropped client → 3 results → raise, not silent 3-way average.
- **Deterministic aggregation:** `DeterministicFedAvg` sorts results by integer CID before `super().aggregate_fit`, addressing FP non-associativity from Ray arrival order. Residual non-determinism (mini-batch scheduling) is disclosed in §7.5 and bounded (≤6.1 pp Wisconsin / ≤3.2 pp Pima within-seed).
- **Server never sees raw data or raw explanations except the object of study:** confirmed — only weights cross in `client_fn`/`XQFLClient`; the per-round Q-SHAP vectors the server logs *are* the research artifact, and §5.1 says local explanations are retained locally "except where they are logged for offline analysis". A privacy-purist reviewer may note that logging per-client φ centrally is itself a disclosure channel; the paper's RQ4 is precisely about that channel, so this is coherent, but §5.1 could say so directly.

---

## Reproducibility Audit

| Item | Status |
|---|---|
| Reproduce Tables 1–5 | Possible: `analysis/aggregate_results.py` reads only `results/*.csv` and writes `results/outputs/table*.{csv,md}`. Ran clean this session; outputs match the manuscript. **Not documented anywhere a reader would find.** |
| Reproduce Figures 2–5 | `analysis/plots.py` (same imports). Figure 1 is `analysis/gen_figure1.py` (static architecture). Undocumented. |
| Exact commands | Absent from README. Present only in session notes / `experiments/rerun_corrected_stage{1,2}.py` orchestrators. |
| Seeds / config | In each `experiments/phase*.py` as module constants; not centralized. `experiments/config.py` exists but is thin. |
| Dependencies | `requirements.txt` (root) + `xqfl-env/`. No pinned lockfile shown; `flwr` 1.32.1 is load-bearing (partition-id semantics) and should be pinned explicitly. |
| Partition construction | `data/partition.py` (`iid_split`, `dirichlet_split`, `quantity_skew_split`); described in §4.2 / §6.2. Adequate. |
| Corrected partition behaviour | §6.7 in the manuscript; `CORRECTED_RERUN_AUDIT.md`; code comments. Good. |
| PDF build | No script. `tectonic` is available on this host; pandoc 3.11 present; `fix_approx.lua` filter exists. Needs a one-line documented command. |
| `references.bib` | **Missing.** 37 hand-numbered refs in the `.md`. A reviewer cannot mechanically verify them and any renumber will silently break in-text `[n]` cites. |

---

## Claim Overstatement Audit

Grep-verified **absent** from `XQFL_paper.md`: "scheduler stochasticity" as a contribution, "attractor bifurcation" (except the §7.5 sentence that explicitly disavows it), "20–30 point" within-seed range (same), MN-08 / MN-09 / MN-12 / MN-13, IQP seed-7 exclusion, "0.722", "every encoding leaks", "quantity skew is harmless", "protects" / "guarantees" / "formal DP".

Residual softenings still needed:
- Abstract "drives the largest local-to-global explanation divergence" → keep, but the body must carry the MC-2 selection caveat so "drives" is not read as monotone causation. (The revision brief's own preferred wording — "produces the largest … in the Wisconsin experiments" — is acceptable; the issue is the missing survivorship sentence, not the verb.)
- §1.5 contribution 2 "keeps its attributions closest to the centralized reference" — true on the point estimates, but "closest" across the Wisconsin encodings is within the estimator floor. Scope it to "lowest two-dataset divergence", which is the defensible version and is already the wording in the contribution's second half.
- §8.3 "The clearest encoding effect is that angle collapses on Pima entirely, and IQP is functional there only under IID" — this is well-supported (angle 0/20 Track-A on Pima; IQP 3/5 IID, ≤2/5 elsewhere) and correctly distinguishes encoding-specific from architecture-specific. Keep.
- Appendix Table 6 grades RQ1 amplitude "High". After MC-1, "High" is defensible *for the two-dataset direction and the Pima gap*, not for the Wisconsin distribution ordering — tighten the cell text.

---

## Required Changes Before Submission

**Manuscript (A) — none overturn a headline claim, no recompute:**
1. **MC-1:** Add the §5.5 estimator floor (JSD ≈ 0.17) to §7.1 and §8.6. Restrict RQ1 interpretation to the tight-band observation plus the amplitude-vs-re-uploading Pima gap. Re-scope §1.5 contribution 2 and Appendix Table 6 RQ1 cell.
2. **MC-2:** In §7.3 and near the abstract/contribution-5 claim, state that the α=0.5 > α=0.1 ordering is expected under Track-A selection and that RQ3 claims no monotone skew→divergence relationship. Keep the clean "quantity skew ≈ IID" half.
3. **MC-3:** Add two caveats to §6.5/§7.4/§8.6: (a) attack-classifier standardization is fit on the pooled member/non-member set — absolute AUCs are a mild upper estimate, ranking and angle-vs-chance unaffected; (b) member/non-member explanations use train/test backgrounds, so part of the AUC may be input-distribution shift, bounded by angle remaining at chance.
4. **mc-5:** §4.3 — threshold is per-split majority + 2 pp; quoted values are the seed-42 split.
5. **mc-4:** §7.2 — one sentence reframing the RQ2 ranking as dominant-component lock-in speed/cleanliness.
6. **mc-7:** Tables 1 & 3 — footnote every n ≤ 3 printed mean as indicative only (or raise `MIN_VALID_DRAWS` to 4 and regenerate — a table-only regeneration, no experiments).

**Code hygiene (A, no logic change):**
7. **mc-6:** Fix the `rq3_local_global_cell` and `rq4_dp_extension` docstrings to match current (post-fix) behaviour.

**Reproducibility (Phase 7):**
8. **mc-9:** Real `README.md`; `references.bib`; documented "reproduce Tables 1–5 / Figures 2–5" command block; pin `flwr==1.32.1`; rebuild `XQFL_paper.pdf` from the current `.md`.

**Optional (C) — only if the authors want to remove rather than disclose MC-3:**
9. Re-run the 20 Wisconsin IID MIA models with an in-CV `Pipeline` and a common fixed Q-SHAP background for members and non-members (~2–3 h). Not required for publication.

**Not defensible as currently written:**
- Any RQ1 statement that reads sub-0.1 Wisconsin JSD_cent differences as distributional effects (MC-1).
- Any RQ3 phrasing implying severe skew produces *less* divergence than moderate skew, or that skew severity monotonically drives divergence (MC-2).
- "High" confidence for RQ1 amplitude in Appendix Table 6 without scoping it to the two-dataset direction.

**Defensible and should be kept:**
- Amplitude's two-dataset fidelity direction and the Pima amplitude-vs-re-uploading gap.
- The RQ2 two-tier ranking at n=15.
- Angle's MIA AUC at the boundary of chance; the other three resolvably above angle.
- "No single Laplace budget brings all four encodings to near-chance."
- "Quantity skew ≈ IID for local-to-global divergence" (Wisconsin, functional runs).
- The §6.7 disclosure as a subsection, not a footnote.
