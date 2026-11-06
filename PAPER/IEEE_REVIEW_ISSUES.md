# XQFL paper: IEEE Access review issue tracker

Manuscript: `PAPER/xqfl_elsievier.docx`, "Enhanced Attribution Dynamics in Explainable Quantum Federated Learning Across Encoding Strategies".

Source: simulated strict IEEE Access review, scored 5.1/10 on its weighted scorecard (its headline says 5.2), verdict "reject, resubmission advised". I checked each claim against the manuscript and the code. Issues marked **[new]** were not in the review and came from that check.

**Fix types:** `Text` = manuscript edit. `Reanalysis` = existing logs. `Rerun` = same experiment, corrected protocol. `New` = new experiment.

**Status:** `Open`, `Started`, `Done`, `Decision` (needs your call, see section 3).

---

## 1. What the pilot checks found

I ran three checks on the 75 final models saved by the RQ4 repair (Wisconsin IID, four encodings, and Pima amplitude IID, 15 seeds each). No training was involved.

- Script: `analysis/review_pilot_checks.py`
- Output: `results/review_pilots/`

These numbers are diagnostics for planning the rerun. They are not paper results.

### 1.1 Most of the RQ3 distance appears without any model disagreement

The same model played both the "local" and the "global" role, using the paper's explained records and backgrounds. The true model disagreement is therefore zero, yet the distance stays large.

| Wisconsin encoding | Published IID (Table 3) | Same-model floor, IID | Published α=0.5 | Same-model floor, α=0.5 |
|---|---|---|---|---|
| Angle | 0.438 | 0.352 | 0.619 | 0.398 |
| Amplitude | 0.289 | 0.367 | 0.461 | 0.366 |
| Re-uploading | 0.336 | 0.300 | 0.406 | 0.354 |
| IQP | 0.369 | 0.357 | 0.405 | 0.385 |

- **IID:** published values are at or below the floor, so they cannot be read as model disagreement.
- **Pima amplitude:** the floor is 0.36–0.39 and the published values are 0.29–0.37. This probably explains why Pima looked "inconclusive".
- **α=0.5:** angle and amplitude still sit above the floor (by about 0.22 and 0.10), so a real label-skew effect may survive. Re-uploading and IQP are close to the floor.
- **Limits of the comparison:** the floor uses final IID-trained models, per-seed splits and exact Shapley, while published values use one fixed split (F-10) and average rounds 1–15 of real client models. Treat it as an order-of-magnitude check. The redesigned RQ3 (S-02) gives the proper answer.

### 1.2 Signed averaging causes much of that noise

Eq. (6) averages signed attributions across records, so positive and negative values cancel. `‖mean φ‖` is only about half of `mean ‖φ‖` (Wisconsin amplitude: 0.191 vs 0.424). Using the standard `mean |φ|` importance cuts the IID floor roughly in half, to 0.15–0.20.

### 1.3 RQ2 ranks signed values, not importance

`kendalls_tau` is applied to raw signed vectors (`explain/metrics.py`, `analysis/aggregate_results.py:369`). On the same vector, the signed ordering and the magnitude ordering disagree strongly: mean τ between them is −0.60 for amplitude, −0.56 for re-uploading, −0.51 for IQP and −0.29 for angle. The paper's "feature ranking" and "dominant-component lock-in" language describes magnitude, but the statistic does not measure magnitude.

### 1.4 M=8 is noisier than the paper's single check suggests, and costs more than exact

| Wisconsin encoding | JSD, M=8 vs exact | JSD, M=8 vs another M=8 | Mean τ, M=8 vs exact |
|---|---|---|---|
| Angle | 0.069 | 0.108 | 0.911 |
| Amplitude | 0.061 | 0.080 | 0.911 |
| Re-uploading | 0.046 | 0.060 | 0.956 |
| IQP | 0.083 | 0.106 | 0.889 |

- The paper reports τ = 1.0 from one model. Across all 75 models, the worst τ is 0.33 and the worst JSD is 0.233.
- This estimator error is larger than the 0.031 spread between encodings on Wisconsin IID fidelity.
- Exact enumeration takes 16 model evaluations per sample. M=8 as implemented takes 40. Exact is 2.5 times cheaper and has no sampling noise. This is covered by a unit test.

### 1.5 The RQ4 null probably has a structural cause

The models do not overfit, so there may be no membership signal for any explanation attack to find.

| Target | Train accuracy | Test accuracy | Loss-threshold MIA AUC (mean ± SD) |
|---|---|---|---|
| Wisconsin angle | 0.752 | 0.746 | 0.519 ± 0.031 |
| Wisconsin amplitude | 0.953 | 0.950 | 0.503 ± 0.027 |
| Wisconsin re-uploading | 0.934 | 0.938 | 0.497 ± 0.027 |
| Wisconsin IQP | 0.892 | 0.885 | 0.506 ± 0.039 |
| Pima amplitude | 0.724 | 0.726 | 0.502 ± 0.022 |

This changes the review's advice. Stronger attacks (random forest, gradient boosting, shadow models) are unlikely to change the result on a 13-parameter model that does not memorize its training data. The paper needs a positive control first (S-07). This check is a simple average-case attack, so it does not rule out rare per-record leakage.

---

## 2. Issues

### 2.1 Statements that are wrong today (fix whatever else happens)

| ID | Issue | Evidence | Type | Status |
|---|---|---|---|---|
| F-01 **[new]** | "Amplitude performs best on fidelity across both datasets" contradicts Table 1. On Wisconsin, amplitude is never lowest in any regime, and its IID value (0.152) is the highest of the four. Section 5.1 itself says "No encoding comes out ahead". Appendix A's "lowest JSD_cent point estimates" is also false. | Abstract, §1 (ii), §6, Appendix A vs Table 1 | Text | Open |
| F-02 **[new]** | The gate is described as "fixed before any result was inspected" and "fixed in advance". The project notes define it next to a list of known-collapse seeds, so it was written after collapses had been seen. It can honestly be described as fixed before the corrected rerun. | Abstract, §3.6 vs `analysis/methods_notes.md:867-889` | Text | Open |
| F-03 **[new]** | "The embedding reaches the full Hilbert space" is false for this implementation. Four non-negative inputs are zero-padded, so only amplitudes 0–3 of 16 are populated and qubits 0 and 1 stay in \|0⟩ (P = 1.0). Normalization also removes scale, leaving 3 degrees of freedom from 4 inputs. | §3.2, Appendix B Fig. 10, `models/encodings.py:27` | Text, Rerun? | Open |
| F-04 **[new]** | Pima zero-coded missing values are used as real measurements. Insulin has 374 zeros (48.7%), skin thickness 227, blood pressure 35, BMI 11, glucose 5, and 376 rows (49%) are affected. This goes further than the review, which only said it could not tell from the text. | `data/loaders.py:27-55`, OpenML 37 | Rerun | Built in v2, awaiting primary run |
| F-05 | The Pima link in the data statement points to xgboosting.com, while §4.1 cites OpenML 37. The Wisconsin link also differs from §4.1. | Data availability section | Text | Open |
| F-06 **[new]** | The public repo cannot reproduce Tables 4 and 5. `experiments/rq4_release_repair.py` and `analysis/current_rq4.py` are untracked, `target.pkl` files are gitignored, and `*.png`/`*.pdf` are ignored. The paper cites this repo as its code source. | `git ls-files`, `.gitignore` | Repo | Open |
| F-07 **[new]** | Table 2 reports τ gaps in "pp", but τ is not a percentage. | Table 2 | Text | Open |
| F-08 **[new]** | Appendix A's "High" ratings have no stated method and conflict with the text. RQ3 Wisconsin is rated High despite the confound, and the α=0.5 "most divergent" claim rests on 0.461 vs 0.446 with SD above 0.07. | Table 6 | Text | Open |
| F-10 **[new]** | §3.6 says each run has "its own stratified test split" and that the threshold moves about ±1.5 points across seeds. RQ1–RQ3 actually use one split (`random_state=42`) for every seed, so all those seeds explain the same 10 test records against the same 20 background records. Seeds vary only initialization and partition, so the reported SDs leave out data-split variation. Only the RQ4 targets split per seed. | `experiments/phase10_explanation_sweep.py:98`, `phase11_encoding_noniid_sweep.py:148`, `rq2_seed_extension.py:78`, `phase16_pima_explanation_sweep.py:99`; `results/phase10_log.csv` has one threshold | Rerun, Text | Built in v2, awaiting primary run |
| F-09 **[new]** | Reference [5] names the journal as "IEEE Journal on Selected Topics…". Crossref gives "of Selected Topics". The docx metadata title also differs from the printed title, and figure alt text still says "Fig. S1" to "Fig. S4". | References, docx properties | Text | Open |

### 2.2 Critical science

| ID | Issue | Required fix | Type | Status |
|---|---|---|---|---|
| S-01 | Q-SHAP samples 8 permutations when exact enumeration is feasible and cheaper. It was validated on only one model. | Use exact for all 4-feature explanations. Report sampling error across encodings and datasets, compare signed vectors, and test other background references. | Rerun | Built in v2, awaiting primary run |
| S-02 | RQ3 mixes model, record and background differences (confirmed in 1.1). | Two evaluations. Model disagreement uses the same records and same background on both sides. Population disagreement uses client records and client backgrounds. Report the same-model floor beside them. | Rerun | Built in v2, awaiting primary run |
| S-03 **[new]** | Signed averaging across records cancels attributions and inflates the distance noise (1.2). | Report `mean \|φ\|` as primary importance. Keep signed values for direction analysis. | Rerun | Built in v2, awaiting primary run |
| S-04 **[new]** | RQ2's τ does not measure importance rank (1.3). | Compute τ on magnitudes. Report signed τ separately and revise the interpretation. | Reanalysis first, then Rerun | Built in v2, awaiting primary run |
| S-05 **[new]** | Explained sets are very small and are simply the first rows: 10 test records, and 5 local records against 10 background records. The first 5 local records are also inside their own background. | Use larger, stratified, disjoint explained and background sets. Since exact is cheaper, enlarge the sets. | Rerun | Built in v2, awaiting primary run |
| S-06 **[new]** | RQ1 has no noise floor. A JSD_cent value has no meaning without the distance between two centralized models that differ only by seed. The reference can also be less accurate than the model it judges (amplitude: centralized 92.81% vs federated 95.44%). | Add centralized-vs-centralized floor and report JSD_cent relative to it. Address reference quality. | Rerun | Built in v2, awaiting primary run |
| S-07 **[new]** | RQ4 has no positive control, so its null result cannot be interpreted (1.5). | Add a loss-based baseline attack, plus a deliberately overfit target (small training set, more capacity) to show the attack can succeed when leakage exists. | New | Built in v2, awaiting primary run |
| S-08 | Statistics are too weak: 3–5 seeds in most cells, no confidence intervals, no multiple-comparison correction, and the two-SE rule stands in for a test. | 30 independent seeds on primary comparisons, 95% CIs, paired tests on shared seeds (Wilcoxon), Holm correction, effect sizes, hierarchical bootstrap over seeds. Separate seeds from repeat draws in every table. | Rerun, Reanalysis | Built in v2, awaiting primary run |
| S-09 | Track-A filtering conditions results on success. Coverage differs by encoding (Pima angle has none). | Make functional coverage a primary outcome with failure rates per cell. Rerun analyses at other thresholds. Report explanation quality together with failure probability. | Reanalysis | Built in v2, awaiting primary run |
| S-10 **[new]** | The gate uses test accuracy, and the same test set supplies the reported accuracy and explained records. | Add a validation split for gating. | Rerun | Built in v2, awaiting primary run |
| S-11 | Novelty is not established. All components are existing methods. | Define a reusable protocol (inputs, outputs, metrics, validity conditions) with pseudocode. Show one phenomenon that standard federated XAI evaluation would miss. The noise-floor result in 1.1 is a strong candidate. | Text, Reanalysis | Open |
| S-12 | The architecture is too small to generalize: 4 qubits, 13 parameters, 4 clients, 15 rounds. | Sweep features, depth, clients and rounds (see 2.4). | New | Open |

### 2.3 Quantum design and implementation

| ID | Issue | Required fix | Type | Status |
|---|---|---|---|---|
| Q-01 | "Data re-uploading" runs both embedding blocks before the trainable circuit, not interleaved. | Implement true interleaving at the same 13 parameters, or rename. | Rerun or Text | Built in v2, awaiting primary run |
| Q-02 | Amplitude dimensionality, populated amplitudes and preparation cost are conflated. | Correct the text (F-03) and include the table below. Decide whether to change the padding. | Text, Rerun? | Started: costs measured |
| Q-03 | No circuit cost comparison. | Table below, then add to the paper. | Text | Started |
| Q-04 **[new]** | Mean imputation is applied in PCA and scaled coordinates. For amplitude encoding, replacing one input also changes every other amplitude through normalization, so "feature absence" means something different per encoding. | Evaluate alternative references and state this limitation. | Rerun, Text | Open |
| Q-05 | Simulator only, with no shot or device noise. | Finite-shot and noise-model simulation. | New | Open |

Embedding block costs, measured from the implemented circuits with PennyLane's decomposition to RY/RZ/RX/CNOT/H (no optimization). All four share the same trainable circuit, with 13 parameters in total.

| Encoding | Depth | CNOTs | Other gates |
|---|---|---|---|
| Angle | 1 | 0 | 4 RY |
| Amplitude | 24 | 14 | 12 RY |
| Re-uploading (2 blocks) | 7 | 6 | 8 RY |
| IQP | 18 | 12 | 8 H, 10 RZ |

The amplitude figure overstates what is needed. Only 2 qubits are actually in use, so an optimized preparation would be far cheaper. Say so if the table goes in the paper.

### 2.4 Experimental scope

| ID | Review proposal | My note | Priority |
|---|---|---|---|
| E-01 | Add 3–5 datasets | Choose datasets with more samples, because client scaling needs them. | High |
| E-02 | Features 4, 8, 16 | **16 is impossible on Pima, which has 8 raw features.** Angle-type encodings at 16 features need 16 qubits with a slow per-sample loop, and exact Shapley needs 65,536 coalitions. Suggest 4, 6, 8, and sampled Shapley with a stated justification beyond that. | High |
| E-03 | Depth 2, 4, 6 | Also separates capacity from encoding, which §6.1 leaves open. | High |
| E-04 | Clients 4, 8, 10, 20 | 20 clients leaves about 23 Wisconsin records each. Check ghost clients before committing. | High |
| E-05 | Rounds 15, 30, 50 | Answers whether "lock-in" lasts. | High |
| E-06 | Classical baselines: LR, MLP, another model | Existing 13-parameter MLP is a start. | Medium |
| E-07 | Estimators: exact, KernelSHAP, Q-SHAP | Run across the population, not on one model. | Medium |
| E-08 | Attack comparison: LR, RF, GB, shadow | Only after S-07 shows there is a signal. | Low until S-07 |
| E-09 | Pima non-IID privacy, more evaluation records, confidence intervals | Depends on the RQ4 decision. | Depends |
| E-10 | Formal (ε, δ)-DP: neighbor relation, sensitivity bound, mechanism, accounting | Large piece of work. Only worthwhile if privacy stays a central claim. | Depends |

### 2.5 Privacy reporting

| ID | Issue | Fix | Type | Status |
|---|---|---|---|---|
| P-01 | The empirical noise index uses the symbol ε, inviting confusion with a DP budget. | Rename to a noise-scale parameter such as σ throughout. | Text | Built in v2, awaiting primary run |
| P-02 **[new]** | Noise scale comes from the norm of dataset-averaged vectors but is applied per coordinate to per-sample vectors, whose norm is larger by a different factor per encoding (angle 0.229 vs about 0.22; amplitude 0.262 vs about 0.42). The same index means different relative noise per encoding. | Calibrate on per-sample scale, or report the signal-to-noise ratio alongside. | Rerun | Built in v2, awaiting primary run |
| P-03 | IID includes all targets, while non-IID requires three Track-A targets. Shadow-attack results were not re-evaluated. | Use one inclusion rule. Rerun shadow attacks or drop them from the paper. | Reanalysis | Built in v2, awaiting primary run |
| P-04 | Privacy takes up too much of the stated contribution for the evidence behind it. | Depends on the RQ4 decision. | Text | Decision |

### 2.6 Reproducibility and documentation

| ID | Issue | Fix | Status |
|---|---|---|---|
| R-01 | Dataset versions, split procedure, PCA fitting and class distributions after partitioning are not fully documented. | Add a methods table. Note that Wisconsin is loaded from scikit-learn. | Open |
| R-02 **[new]** | Weights per round are not saved for RQ1–RQ3, so any change to the explainer means retraining. | Save per-round global and client weights in the rerun, so later explanation changes need no retraining. | Built in v2, awaiting primary run |
| R-05 **[new]** | Client training RNG depends on Ray scheduling, so the same seed can give different runs. | Seed each client from seed, round and partition ID, and test for identical weights. | Built in v2, awaiting primary run |
| R-03 **[new]** | The README is out of date on tests, on shadow-model MIA and on the default of 64 permutations. | Update after the protocol is settled. | Open |
| R-04 | Reference coverage needs checking. | 16 of 16 DOIs resolve on Crossref with matching metadata. The other 20 references have no DOI and are unchecked. A search did not settle whether "Q-SHAP" clashes with an existing method name, so check this by hand. | Started |

### 2.7 Writing, structure and format (do last)

| ID | Issue | Status |
|---|---|---|
| W-01 | Grammar errors. **Six of the seven I traced are absent from `XQFL_paper.md`, so the last humanization pass introduced them.** Use that file as the language baseline rather than re-editing the current text. | Open |
| W-02 | Sentences needing repair: "This study address", "to what extend", "form non-members", "how does calibrate", "was offered", "is trace to", "was tested", "In each run, log…", "In this paper the comparison of…" (no verb), "with amplitude gives", "test separate", "spread is measured", "And compared", "also evaluated", "Retained max", "it describes as modest", "so, no claims", "On the other hands", "will also reveals", "comparison was studied", "models were tested", "pima-IID", the fragment "Evaluation of agreement…" in §2.4. | Open |
| W-03 | Rewrite the abstract around the actual contribution and remove the crowding. | Open |
| W-04 | Strengthen the novelty case in the introduction, and add a related-work comparison table. | Open |
| W-05 | Add pseudocode for the evaluation pipeline and improve the Q-SHAP explanation. | Open |
| W-06 | Consolidate repeated limitations. Remove the trimmed-mean mention, since no result uses it. | Open |
| W-07 | Improve the "why encodings differ" discussion. Rewrite the conclusion around findings, and make future work specific. | Open |
| W-08 | Retitle. The review suggests "An Empirical Framework for Evaluating Feature-Attribution Reliability in Quantum Federated Learning". Make the framework C5 the central contribution, with C1–C4 as its validation. | Open |
| W-09 | Move to the IEEE Access template: authors, abstract, headings, references, figure sizing, equation numbering, declarations. | Open |

### 2.8 Figures

| Fig. | Required action | Status |
|---|---|---|
| 1 Architecture | Improve legibility and consistency | Open |
| 2 Coverage | Tighter layout, clearer annotations | Open |
| 3 Fidelity | Clearer uncertainty and sample counts; add noise floor | Open |
| 4 Stability | Keep; improve axes and legend; redo with magnitude τ | Open |
| 5 Divergence | Show matched vs population results and the floor | Open |
| 6 Privacy | Add confidence intervals and the positive control | Open |
| 7 Noise utility | Clarify the noise parameter and utility reading | Open |
| 8 Attribution profiles | Add uncertainty across seeds | Open |
| 9 Client distribution | Keep | Open |
| 10 Circuits | Correct amplitude and re-uploading descriptions | Open |
| 11 PCA loadings | Improve label readability | Open |
| 12 Correction audit | Keep as supplementary | Open |

### 2.9 Review scores below 5.5, mapped to issues

| Section | Score | Issues |
|---|---|---|
| §4.6 Privacy evaluation, §5.4 Privacy | 4.0 | S-07, P-01 to P-04 |
| §4 Methodology, §4.7 Statistics, Writing | 4.5 | S-05, S-08, S-10, W-01 to W-09 |
| §3.6 Gate, §4.2 Partitions, §4.3 Model, §5.3 Divergence | 5.0 | F-02, S-02, S-09, S-10, S-12 |

---

## 3. Decisions

All five recommendations were approved on 2026-11-05 and are written into [the v2 protocol](../results/XQFL_PROTOCOL_V2.md). Items marked `Decision` above now follow these answers.

| ID | Decision | Approved answer | Why |
|---|---|---|---|
| D-1 | Role of RQ4 (S-07, E-08 to E-10, P-04) | Keep it as a limited exploratory check with a positive control. Remove privacy from the headline contributions and skip formal DP. | 1.5 suggests more attacks will not change the result, and formal DP is a separate project. |
| D-2 | Pima missing values (F-04) | Treat zeros as missing, use median imputation fitted on training data only, and add a missingness sensitivity check. | Standard handling, and cheap. All Pima results change, so it has to happen before the rerun. |
| D-3 | Re-uploading (Q-01) | Implement true interleaving at 13 parameters, and keep the current version as a named variant. | The rerun is happening anyway, and this removes the objection rather than explaining it. |
| D-4 | Amplitude padding (F-03, Q-02) | Keep the current circuit, correct the text, and state the 2-qubit subspace openly. | Avoids creating a fifth configuration. A proper 16-dimensional encoding belongs in the feature sweep. |
| D-5 | Experiment scope (S-12, section 2.4) | Do the Priority 1 rerun first, then one-factor sweeps of depth, features, clients and rounds on one or two extra datasets. | A full cross of all factors would be very large. |

---

## 4. Order of work

1. **Protocol v2. Done.** Frozen at [XQFL_PROTOCOL_V2.md](../results/XQFL_PROTOCOL_V2.md), which gives F-02 a defensible claim.
2. **Code. Done, in `v2/`.** Per-seed splits (F-10), exact Shapley (S-01), matched RQ3 and floors (S-02, S-06), magnitude metrics (S-03, S-04), larger stratified sets (S-05), validation gate (S-10), Pima fix (F-04), interleaved circuit (Q-01), per-round weights (R-02), deterministic clients (R-05).
3. **Primary rerun.** 30 seeds × 4 encodings × 4 regimes × 2 datasets = 960 runs, plus a variant and a Pima sensitivity check, separating training from explanation. The dry run puts this at about 22 hours of machine time at 7 workers, with client explanations every third round, including attacks and estimator validation. Run with `xqfl-env/bin/python -m v2.run primary --workers 7`.
4. **Statistics.** Build S-08 and S-09 as one aggregation pipeline that regenerates every table.
5. **Generalization sweeps** (section 2.4).
6. **RQ4 as decided.**
7. **Commit the code and rebuild the repo** (F-06).
8. **Rewrite last.** Start from the `XQFL_paper.md` language baseline, then the IEEE template. Apply the F-series corrections at that point, or sooner if the manuscript is shared before then.

---

## 5. Change log

| Date | Change |
|---|---|
| 2026-11-05 | Tracker created. Added `exact_q_shap_values` to `explain/q_shap.py` with `tests/test_q_shap.py`. Added `analysis/review_pilot_checks.py`, with results in `results/review_pilots/`. Measured circuit costs and Pima zeros, and checked DOIs. Manuscript not edited, and nothing committed. |
| 2026-11-05 | D-1 to D-5 approved. Protocol v2 drafted at `results/XQFL_PROTOCOL_V2.md`, not yet frozen. Added F-10 (fixed split in RQ1–RQ3) and R-05. No code changes for v2 yet. |
| 2026-11-05 | Built the v2 pipeline in `v2/`, with `tests/test_v2_quick.py` and `tests/test_v2_stats.py` (45 tests in total). Acceptance passed, the 1-seed dry run finished with no failures, and the protocol is frozen. Study outcomes were not inspected. Primary run not started, and nothing committed. |
