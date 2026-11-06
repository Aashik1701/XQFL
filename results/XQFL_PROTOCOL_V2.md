# XQFL experiment protocol v2

**Status:** frozen on 2026-11-05, with one logged pre-run deviation (section 11), after all acceptance tests in section 10 passed. Decisions D-1 to D-5 come from `PAPER/IEEE_REVIEW_ISSUES.md`. The file hash and code hash are in `results/protocol_v2/manifest.json`, and `v2.run` refuses to start if either has changed.

**Output:** `results/protocol_v2/`. Nothing from v1 is overwritten or pooled with v2.

**After the freeze:** any change is written to the deviation log (section 11) with its date and reason, and states whether results had been seen.

**How this may be described in the paper:** v2 was designed after seeing the v1 results and the review pilots, so it is not blind. The honest wording is "analysis plan fixed before the v2 runs". Do not write "fixed before any result was inspected".

---

## 1. What changes from v1

| Area | v1 | v2 | Tracker |
|---|---|---|---|
| Data split, RQ1–RQ3 | One split (`random_state=42`) for every seed | A new split per seed | F-10 |
| Shapley estimator | 8 sampled permutations | Exact, 16 coalitions | S-01 |
| Explained records | First 10 test rows; first 5 local rows | 50 stratified test records; up to 25 local | S-05 |
| Background | First 20 train rows; local background contains the explained rows | 50 stratified train records; disjoint from explained records | S-05 |
| Importance vector | Absolute value of the signed mean | `mean \|φ\|` primary, signed mean secondary | S-03 |
| RQ2 ranks | Signed values | Magnitudes primary, signed secondary | S-04 |
| RQ3 | Mixed model, record and background effects | Model, population and floor measured separately | S-02 |
| RQ1 | No baseline | Compared with a centralized-pair noise floor | S-06 |
| Gate | Test accuracy | Validation accuracy, plus sensitivity analysis | S-10, S-09 |
| Pima | Zeros used as values | Zeros treated as missing | F-04 |
| Re-uploading | Two blocks before the trainable circuit | Interleaved; old circuit kept as a named variant | Q-01 |
| Seeds | 3–15, with repeat draws pooled | 30 independent seeds, one draw each | S-08 |
| Client training RNG | Depends on Ray scheduling | Fixed by seed, round and client | R-05 |
| Weights | Final round only, RQ4 only | Saved every round; explanation runs afterwards | R-02 |
| RQ4 | Main contribution; noise sweep | Exploratory, with positive controls | S-07, P-04 |

---

## 2. Data

**Primary datasets:** Wisconsin Diagnostic Breast Cancer (scikit-learn, 569 records, 30 features) and Pima Indians Diabetes (OpenML 37, version 1, 768 records, 8 features).

**Split, per seed:** stratified 70% train, 10% validation, 20% test. For Wisconsin that is about 398/57/114, and for Pima about 537/77/154.

**Preprocessing, fitted on the training split only, in this order:**
1. Pima only: in `plas`, `pres`, `skin`, `insu` and `mass`, treat 0 as missing and impute the training median. `preg` zeros are real counts and stay.
2. Standardize.
3. Fit PCA and keep the first 4 components.
4. Scale to [0, π] with MinMax. Values from validation and test are not clipped. Record the fraction outside the range.

As in v1, this preprocessing is fitted centrally. The paper states this as an assumption.

**Pima sensitivity check:** rerun amplitude and re-uploading, IID, 30 seeds, with v1 zero handling. This measures how much the missing-value treatment matters.

---

## 3. Models and training

Four qubits, two `BasicEntanglerLayers`, four Pauli-Z readouts, `Linear(4,1)` with sigmoid: 13 parameters. FedAvg, 4 clients, 15 rounds, 3 local epochs, Adam (lr 0.05), batch size 32, binary cross-entropy.

**Execution:** runs use a direct, single-process FedAvg loop (`v2/train.py`), because client RNGs no longer depend on the backend and this allows one run per core. The same client code run through Flower and Ray matched it to within 6e-7 in every weight at every round, on two test cases (`v2/acceptance.py`). The paper should describe the training as FedAvg simulation validated against Flower.

| Encoding | Circuit | Runs |
|---|---|---|
| Angle | RY per feature, then trainable layers | Full matrix |
| Amplitude | Zero-padded to 16, normalized, then trainable layers. Uses 4 of 16 amplitudes, and qubits 0 and 1 stay in \|0⟩. The paper must say so. | Full matrix |
| Re-uploading | [embed, layer, embed, layer], where embed is the v1 RY + CNOT chain block. Only the placement changes. | Full matrix |
| IQP | v1 circuit, then trainable layers | Full matrix |
| Repeated embedding (variant) | v1 "re-uploading": [embed, embed, layer, layer] | IID only, both datasets |

**Partition regimes:** IID, quantity skew (factor 0.5), Dirichlet α = 0.5, Dirichlet α = 0.1.

**Seeds (30):** 1, 7, 42, 123, 999, 11, 13, 17, 19, 23, 29, 31, 37, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113. Each seed sets the split, the partition and the initial weights, so encodings on one seed share all three and form a matched pair. There are no repeat draws.

**Determinism:** before each client's local training, set the torch and numpy RNGs from `SeedSequence([seed, round, partition_id])`, with one torch thread. This should remove the scheduling variation seen in v1, which section 10 tests.

**Centralized reference:** 45 epochs on pooled training data. Train two models per dataset, encoding and seed, on the same split with different initial weights. The first is the RQ1 reference and the pair gives the noise floor.

**Classical controls:** the 13-parameter MLP and logistic regression, through the same pipeline, IID, both datasets, 30 seeds, with centralized pairs.

**Saved per run:** global weights for rounds 0–15, client weights for rounds 1–15, accuracies on train, validation and test, partition sizes and class counts, record indices, code hash. Explanations are computed afterwards from saved weights, so a later change to the explainer needs no retraining.

---

## 4. Explanations

**Estimator:** `exact_q_shap_values`. Absent features take the mean of the background.

| Set | Contents | Used for |
|---|---|---|
| E_g | 50 stratified test records, fixed per seed | Global, centralized and matched-client explanations |
| B_g | 50 stratified training records, fixed per seed | Reference for E_g |
| E_k, B_k | For client k, up to 25 explained and up to 25 background records, stratified where two classes exist, disjoint | Population comparison |

A client with fewer than 10 records is **sparse**: it is excluded from RQ3 means and counted in every table.

**Vectors per explained set:** `I = mean |φ|` (importance, primary) and `S = mean φ` (signed, secondary).

**Estimator validation, on all final models of the IID runs:**
- Sampled M = 8, 24 and 64 against exact: JSD, τ, cosine.
- On 5 seeds per encoding and dataset, also KernelSHAP against exact.
- On the same subset, two other references: full marginal expectation over B_g, and a class-balanced mean.

---

## 5. Gate and coverage

**Functional:** final validation accuracy > validation majority rate + 0.02.

**Functional coverage is a primary outcome.** Report the rate with a Wilson 95% interval for every dataset, encoding and regime cell.

**Sensitivity analysis:** repeat all primary contrasts with margins 0, 0.01, 0.05 and 0.10, with all runs ungated, and report how often the gate agrees with the same rule applied to test accuracy.

**Cell reporting:** 10 or more functional seeds are estimated and tested. 3–9 are descriptive only and flagged. Fewer than 3 are not shown as estimates, and the cell reports its coverage.

---

## 6. Research question metrics

JSD is the base-2 Jensen–Shannon distance of normalized vectors. Each metric is computed on `I` as primary and on `S` as secondary. Cosine and L2 are reported alongside.

**RQ1, fidelity (final round):**
- `JSD_cent = JSD(global, centralized_1)` on E_g and B_g.
- `Floor_cent = JSD(centralized_1, centralized_2)`.
- Primary statistic: `Excess = JSD_cent − Floor_cent`, per seed.
- A fidelity comparison counts as functional only if the federated model and both centralized models pass the gate. Validation accuracy of each centralized model is reported with it.

**RQ2, stability:**
- Mean Kendall τ-b between consecutive rounds, from round 1 to 15 (14 pairs), on `I`.
- Initialization to round 1 is reported separately, because a random initial model is not part of training stability.
- Also: rate of change in the top-ranked feature, cosine, and the round after which the ranking no longer changes.

**RQ3, divergence (client explanations at rounds 3, 6, 9, 12 and 15; their mean and the final round are reported separately):**

| Quantity | Models compared | Records and background | Reads as |
|---|---|---|---|
| `D_model` (primary) | Client vs global | E_g, B_g on both sides | Model disagreement |
| `D_pop` | Client vs global | E_k, B_k vs E_g, B_g | Model and population together |
| `D_floor` | Global vs itself | E_k, B_k vs E_g, B_g | Population only |
| `D_sample` | Global vs itself | Two disjoint halves of E_g, B_g on both sides | Record-sampling noise |

Client averages are taken within a seed first, then across seeds.

---

## 7. Statistics

These replace the two-SE rule and the three-point accuracy rule.

**Inference:** two-sided Wilcoxon signed-rank tests on seed-paired differences. Report the Hodges–Lehmann median difference, the matched-pairs rank-biserial correlation, and a 95% bootstrap interval from 10,000 seed resamples with a fixed RNG seed. RQ3 intervals use a hierarchical bootstrap: seeds, then clients within seeds.

**A pair enters a test only if it is functional in both arms,** and a test needs at least 10 such pairs. Otherwise the contrast is reported as not testable.

**Holm correction at 0.05, separately within each family:**

| Family | Contrasts | Tests |
|---|---|---|
| Coverage | Pairwise encodings per dataset, IID and α = 0.5, using exact McNemar in place of Wilcoxon | 24 |
| RQ1-A | `Excess` vs 0, per dataset and encoding, IID | 8 |
| RQ1-B | Pairwise encodings on `JSD_cent`, per dataset, IID | 12 |
| RQ2 | Pairwise encodings on τ, per dataset, IID | 12 |
| RQ3-A | `D_model` vs IID for each non-IID regime, per dataset and encoding | 24 |
| RQ3-B | `D_model` vs `D_sample`, per dataset and encoding, IID | 8 |

Anything else is exploratory and labelled so. Quantum-versus-classical contrasts are exploratory and carry no claim of quantum advantage.

**Wording rules for the paper:**
- A non-significant result is "not resolved". It is never "equal" or "no effect", since no equivalence test is run.
- Seeds are the unit of inference. Clients and rounds are never treated as independent samples.
- Report every contrast, whatever its direction.

---

## 8. RQ4: exploratory, with positive controls

Privacy is not a headline contribution. No formal differential privacy is attempted. "ε" and "budget" are not used anywhere, and the noise parameter is called σ.

**Targets:** final global models from all primary runs, all encodings and regimes, 30 seeds. This costs little because the weights already exist.

**Attack data:** 50 training members and 50 test non-members, with the validation split as the reference background, so the reference is independent of both groups.

**Attacks:**
1. Loss-threshold attack on model outputs, as the baseline for whether any membership signal exists.
2. Logistic regression on `φ`, `|φ|` and `‖φ‖`, using the v1 repair's nested evaluation in `privacy/release_evaluation.py` (5 outer folds, 4 inner folds).

**Positive controls, on Pima because its labels are noisy (30 seeds):**
- Classical: MLP with two 64-unit hidden layers, Adam lr 0.01, trained on 100 records until training accuracy reaches 1.0 or 500 epochs. Attack uses 50 of those records.
- Quantum: interleaved re-uploading, 4 qubits, 6 layers (29 parameters), Adam lr 0.05, trained on 30 records for 200 epochs. Attack uses all 30.

A control counts as valid only if the lower bound of the 95% bootstrap interval of its mean loss-attack AUC across seeds is above 0.60. Non-members come from the test split, and members are the control's training records.

**"Detects"** in the table below means the 95% bootstrap interval of the mean AUC across seeds lies above 0.5 for that model group. RQ4 is exploratory, so no multiple-comparison correction applies, and any detection is reported as a lead to confirm.

**How results will be read:**

| Control, loss attack | Control, explanation attack | Main targets | Conclusion allowed |
|---|---|---|---|
| Invalid | Any | Any | The pipeline is unvalidated. Report as such and draw no conclusion. |
| Valid | Detects | Null | The attack can work. No signal was detected on these low-capacity models, which is not proof of privacy. |
| Valid | Null | Null | Explanations carried no detectable signal even when loss did. Attack-specific result. |
| Valid | Any | Detects | Report leakage, with the effect size and interval. |

**Noise:** Laplace noise with scale σ relative to the per-sample attribution norm, at ratios 0.25, 0.5, 1 and 2, runs only on models where the attack detects a signal. Without a signal there is nothing to suppress. Shadow attacks are dropped.

---

## 9. Run order and budget

Timing comes from the one-seed dry run: 72 runs at 7 workers on this 8-core machine, with no failed tasks. A run is 2–3 times slower with all workers busy than alone (45–80 seconds of training alone against about 140 under load).

| Stage | Content | Tasks | Measured for 1 seed | Estimate for 30 seeds |
|---|---|---|---|---|
| 0 | Acceptance and 1-seed dry run | done | about 65 min | done |
| 1, 2 | Training: 1,200 federated runs and 960 centralized | 2,160 | 20 min | about 10 hours |
| 3 | Explanations from saved weights, clients every third round | 2,160 | 35 min at every round | about 7 hours |
| 3 | Estimator validation | 240 models | 9 min (all extended) | about 2–3 hours |
| 4 | Statistics, tables | one pipeline | minutes | minutes |
| 5 | RQ4 targets and positive controls | 960 + 60 | 2 min | about 1 hour |
| 6 | Sweeps | see below | | re-estimate after stage 5 |

The primary stages total about 22 hours of machine time. Weights are saved for every round, so skipped client rounds can be explained later without retraining. They resume after an interruption.

**Sweeps** change one factor from baseline and run after stage 4. Each has 15 seeds, four encodings, and IID and α = 0.5. They run in this order, so stopping early still leaves a complete result for what finished.

1. **New datasets at baseline:** Spambase (OpenML 44) and Phoneme (OpenML 1489), each capped at a stratified 1,500 records. 240 runs.
2. **Depth 4 and 6:** Wisconsin 240 runs, and Pima IID 120 runs.
3. **Rounds 30 and 50:** Wisconsin. 240 runs.
4. **Clients 8, 10 and 20:** Wisconsin. 360 runs. Check sparse clients first, as 20 clients leaves about 20 records each.
5. **Features 6 and 8:** Wisconsin 240 runs, and Pima IID 120 runs. Angle, re-uploading and IQP use one qubit per feature, so parameter counts rise. Amplitude stays on 4 qubits. Parameter counts are reported, and these contrasts are not called capacity-matched.

Sweeps use explanations at the final round, plus global explanations every round for τ. Exact Shapley still applies, with 256 coalitions at 8 features. Sweep results are descriptive, with no Holm family.

---

## 10. Acceptance tests before the freeze

All of these must pass:

1. **Determinism:** two runs with the same seed, on two encodings, produce identical weights at every round. If they don't, record the size of the difference and treat it as a deviation.
2. **Partition integrity:** IDs are exactly {0, 1, 2, 3} every round (existing guard).
3. **Shapley:** existing tests, plus efficiency on a real circuit for each encoding.
4. **Set construction:** E, B, E_k and B_k are disjoint where required, and stratified.
5. **Floor check:** `D_floor` and `D_sample` are 0 when records and background are identical.
6. **Pima:** no zeros remain in the five columns after imputation, and the medians come from the training split only.
7. **Interleaving:** the circuit drawing shows [embed, layer, embed, layer] with 13 parameters.
8. **Dry run:** the full pipeline runs on one seed end to end, and all tables build. Dry-run values are used for checking validity only and are not reported. This seed is run again in stage 1.

---

## 11. Deviation log

| Date | Change | Reason | Results seen? |
|---|---|---|---|
| 2026-11-05 | Client explanations reduced from every round to rounds 3, 6, 9, 12 and 15, to cut about 10 hours of machine time. Global explanations stay at every round. | Run time on one laptop | No. Only dry-run timing, failures and table shapes were inspected, and no primary run existed. |
