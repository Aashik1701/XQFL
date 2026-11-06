# RQ4 release evaluation repair

Status: all 115 targets and 195 release evaluations completed. Corrected RQ4 outputs are promoted to current tables and deliverables by explicit user authorization.

The historical RQ4 raw CSVs remain unchanged; current aggregate tables now use corrected results. The
new output directory is `results/rq4_release_repair_v1`. Its manifest records
source hashes and the protocol; changing that protocol requires a new version.

## Scope

- Initial release sweep: Wisconsin IID, four encodings, existing five seeds
  1, 7, 42, 123, 999; clean baseline and epsilon 0.1, 0.5, 1, 5.
- Remaining clean direct-MIA evaluation: Wisconsin IID existing ten extension
  seeds per encoding, Pima amplitude IID existing 15 seeds, and Wisconsin
  quantity skew/alpha 0.1 existing five seeds per encoding.
- Total: 115 trained targets and 195 clean/noisy release evaluations. These
  replace existing RQ4 conditions; no new datasets, training seeds, architecture,
  or RQ1–RQ3 experiments are introduced.
- Old shadow-model AUCs are not mixed with the corrected direct-attack results.
  Shadow-model reevaluation is not included in this direct-MIA repair.

## Corrections and interpretation

1. Generate a whole release matrix from one RNG stream, without per-row
   reseeding. Member/non-member records do not share noise vectors. Separate
   deterministic streams are used for each noise level and target.
2. Fit StandardScaler inside each training fold using a sklearn pipeline.
3. Use the same background for all 100 attack records: target-training rows
   50–69, excluded from the 50 evaluated members. The target training data are
   unchanged. This is a fixed training-derived reference, not an independent
   public dataset or a formal privacy mechanism.
4. Preserve the five-fold raw-AUC aggregation; additionally save pooled OOF AUC.
5. Select score direction and TPR−FPR-maximizing threshold using inner OOF
   predictions restricted to the outer training fold. Evaluate these decisions
   on the untouched outer fold. Inner threshold selection is a repair-protocol
   addition, not part of the historical experimental protocol.
6. Report max(AUC,1−AUC) only as an evaluation-selected orientation diagnostic.
   It is not an unbiased attack-performance estimate.
7. Report TPR−FPR at the validation-selected threshold, plus descriptive pooled
   OOF TPR at FPR ≤1%, ≤2%, and ≤5%. With 50 negatives, empirical FPR resolution
   is 2%; sub-resolution operating points are not validated deployment claims.
8. Measure cosine and L2 utility on the SAME clean/noisy per-sample explanation
   vectors used by each attack. Zero-norm cosine is undefined and counted,
   never imputed as perfect utility. These utility measurements were absent
   from the old release logs and are new derived measurements in this rerun.

No numerical threshold for declaring privacy has been introduced. Chance-level
performance of this particular attack cannot certify privacy. The empirical
attribution scale remains non-certified. Multiple-query composition is not
evaluated. Any outcome, including unexpectedly low AUC, will be retained.

## Preservation and resume

For each target save model weights, fitted preprocessing, data fingerprint,
global accuracy by round, the background/record indices, clean explanations,
every noisy release, membership labels, raw/oriented held-out scores, fold
assignments, threshold/orientation decisions, and a completion record.

Capture all training logs and verify the observed partition IDs are exactly
`[0,1,2,3]` in each of rounds 1–15. Track A is determined anew from the same rule:
final accuracy strictly exceeds the run's test-set majority rate plus 0.02.
Never copy old Track-A classifications onto retrained targets. Preserve all
Track-B rows; functional summaries are separately labelled.

The runner stops below 5 GiB free disk, stops on any target failure, and removes
only its own newly created Ray scratch directory after each worker exits.

## Commands

```sh
./xqfl-env/bin/python -m unittest discover -s tests -p 'test_release_evaluation.py' -v
./xqfl-env/bin/python -u experiments/rq4_release_repair.py --scope sweep --limit 1
./xqfl-env/bin/python -u experiments/rq4_release_repair.py --scope all
```

The first command is a diagnostic test suite, not a scientific experiment.
The one-target pilot is part of the requested sweep and is reused on resume.
Do not run legacy sweep scripts to produce replacement results: they retain
historical output paths and legacy explanation-background construction.

## Final audit and promotion

Before promoting any replacement tables: verify completion counts, source and
historical-output hashes, all partition logs, disjoint folds and held-out
predictions, label orientation, matched utility provenance, actual Track-A
counts, and old-versus-new results. If the data no longer support earlier
rankings or mitigation claims, report that explicitly. The user authorized downstream updates after completion. See `RQ4_PROJECT_UPDATE_AUDIT.md` for the promotion record. The original `rq4_release_repair_v1/AUDIT.md` is retained: its only failure was a user-modified Word-file hash. `RESULTS_ONLY_AUDIT.md` rechecks experimental evidence without requiring intentionally updated presentation files to retain old hashes.
