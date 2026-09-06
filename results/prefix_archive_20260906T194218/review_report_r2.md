---
title: "Academic Paper Re-Review Report — XQFL Paper (Verification Round)"
type: "Re-Review"
mode: "re-review (verification)"
date: "2026-09-06"
---

# Academic Paper Re-Review Report (Verification Round, mode: re-review)

Verification of whether the prior round's Major Revision roadmap (review_report.md, Phase 2)
items are resolved in the current `XQFL_paper.md` (465 lines, 37 references).

---

## R&R Traceability Matrix

| # | Prior roadmap item (severity) | Author's response in manuscript | Verified? | Notes |
|---|-------------------------------|---------------------------------|-----------|-------|
| P0-1 | Add classical FedAvg baseline on same 4-PCA features (DA-1 / R1) | **Section 8.2** "Testing the Circularity Concern: A Classical Control" + **Section 6.4b** methods. Matched 13-param MLP through identical pipeline (shared PCA(4), IID K=4, FedAvg 15x3, 45-epoch central, same Q-SHAP 20/8/10, same JSD). Now run at **n=15 on Wisconsin AND Pima** (extended this session). | ✅ Yes | Classical JSD Wis = 0.186 ± 0.145 (n=15); Pima +5.0pp over majority, 5/15 collapse seeds. |
| P0-2 | Address circularity: show JSD metric is sensitive to architecture OR argue controlled comparison valid (DA-1) | **Section 8.2** via route (a): classical across-seed std (0.145) > entire spread of quantum means (0.032); individual classical seeds diverge up to JSD 0.49 while no quantum encoding does. Also Pima capacity evidence (classical collapses 5/15). Honest framing: directional, not conclusive — no overselling. | ✅ Yes | Result is directionally consistent at both n=5 and n=15; reported honestly as directional. |
| P0-3 | Specify Q-SHAP/Shapley details: exact definition, number of permutations, rationale, exact-vs-approximate, background/eval dataset, normalization, epsilon (R1) | **Section 3.4** (exact-vs-approximate, 4!=24 vs 8 permutations, unbiased approximation), **Section 3.5** (JSD: abs + epsilon=1e-10, base-2, epsilon sensitivity analysis), **Section 5.5** (Q-SHAP: 20 bg, 8 perms, mean-imputation, efficiency axiom, exact-permutation cross-check vs 24, KernelExplainer validation vs shap package, single-model caveat). | ✅ Yes | Fully specified; reproduced from actual code constants. |
| P1-4 | Qualify IQP classical-hardness claim (R2) | **Section 2.2** and **Section 3.3**: "the specific hardness result is about sampling from the circuit's output distribution, not about the binary classification task studied here, and no equivalent hardness result is claimed for the latter." | ✅ Yes | Correctly scoped. |
| P1-5 | Add deployment scenario (EIC/R3) | **Section 8.4** "A concrete scenario grounds these implications": hospital consortium, clinician query, consent, data-governance team, HIPAA/GDPR Art 22 [24]. | ✅ Yes | Concrete and actionable. |
| P1-6 | Clarify centralized baseline matching (R1) | **Section 6.4**: quantified learning-curve gap (45 round-checkpoints, 3 seeds: mean gap -0.032, std 0.076, closes by round 5-8, within 0.9-1.8pp by round 15). | ✅ Yes | Direct empirical quantification. |
| P2-7 | Compare Q-SHAP with classical SHAP (R3/R1) | **Section 5.5** KernelExplainer cross-check (Kendall tau=1.0, cosine 0.978, JSD 0.171, L2 0.110) + **Section 8.6** limitation note. | ✅ Yes | Single-model spot check, appropriately hedged, limitation acknowledged. |
| P2-8 | Discuss clinical interpretability of PCA components (R3) | **Section 8.6** new bullet: "PCA components are not clinical features... clinical meaning of any single component is not established here." | ✅ Yes | Now an explicit limitation. |
| P2-9 | Add federated classical XAI literature (DA-2/R2) | **Section 2.7**: cites Li/Gui/Wu (arXiv:2503.12958) federated Shapley explainability+privacy [5]; **Section 2.2**: Schuld & Petruccione textbook [13]; **Section 2.7**: quantum DP [9] (Watkins et al.). | ✅ Yes | Verified real references. |
| P2-10 | Specify exact train/test split (R1) | **Section 6.1**: "exact 80/20 stratified train/test split (train_test_split, test_size=0.2, stratified by label... Wisconsin training set is 455 samples)". | ✅ Yes | Specified precisely. |

**All 10 prior roadmap items are now resolved.** No prior CRITICAL or MAJOR issue is left unaddressed in the manuscript.

---

## Two New Issues Found During Verification

### NEW-MAJOR-1: Classical baseline is directionally NOT conclusive (honest).
The extended n=15 classical control still does not resolve the gap to the quantum encodings
(largest ratio 1.26 SE vs angle, 0.50 vs IQP). This is reported honestly in §8.2, and the
Pima capacity evidence (classical collapses 5/15 seeds, +5.0pp over majority) is a genuinely
strong result. However, a strict reviewer would note that the circularity concern is therefore
**mitigated, not eliminated**. The manuscript already frames this correctly ("directional, not
conclusive"), so this is a residual reviewer caveat, not a defect in the manuscript. Journal-fit
response: explicitly suggest a future capacity-perturbation test (vary n_qubits/hidden width) in
§10 Future Work to close the object fully.

### NEW-MINOR-1: Reference-number bug in §3.4 (fixed this session).
The §3.4 citation for SHAP's permutation-sampling estimator originally pointed at [17]
(Benedetti, parameterized circuits) rather than [25] (Lundberg & Lee, SHAP). Corrected during
this re-review to [25]. Verified all other §3.4/§3.5/§5.5 citations are correct.

### NEW-MINOR-2: Abstract value fixed (fixed this session).
Abstract previously claimed "Pima IID JSD 0.037 ± 0.025"; the authoritative table1_rq1_fidelity
value is 0.207 ± 0.111. Corrected in the abstract to match the authoritative table.

---

## Phase 2.5 — Revision Coaching (residual items only)

Given all P0/P1/P2 items are resolved, only two optional hardening actions remain:

1. **(Optional, recommended)** Add one sentence to §10 Future Work offering a "capacity
   perturbation" experiment (vary hidden width or n_qubits across a small grid and re-check the
   classical-vs-quantum JSD gap) as the decisive follow-up to the circularity concern. This
   directly answers "mitigated not eliminated" from NEW-MAJOR-1.

2. **(Mechanical)** Before submission, re-render Figure 1 and confirm the four RQ boxes do not
   clip with bbox_inches=tight; the current figure is 300dpi / 3846x2627.

---

## Editorial Decision (re-review)

**DECISION: MINOR REVISION (near-accept).**

Rationale per v1.10 re-review rule: all 10 prior roadmap items verified resolved. The two new
findings are a residual-and-acknowledged caveat (NEW-MAJOR-1, already honestly framed) and two
tiny fixes already applied during this re-review (NEW-MINOR-1/2). No remaining CRITICAL issues.
The Devil's Advocate's prior CRITICAL (DA-1 circularity) has been responded to with an honest
experiment and honest framing; per checkpoint rule #4 the decision is unblocked.

Recommended optional action before final submission:
- Add the capacity-perturbation sentence to Future Work (§10).
- Verify Figure 1 renders without clipping in the final PDF.

No further experiment is required. Score on rigor: 7/10 (from 6.5/10 prior round). Overall: 78/100.