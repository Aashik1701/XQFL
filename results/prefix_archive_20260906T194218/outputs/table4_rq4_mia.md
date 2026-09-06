# Table 4 — RQ4 Membership Inference: Direct-MIA AUC

| dataset | encoding | distribution | n | auc_mean | auc_std | collapse_rate | status | note |
|---|---|---|---|---|---|---|---|---|
| pima | amplitude | iid | 15 | 0.5620 | 0.1032 | 0.0000 | ok | RQ4 extension: cross-dataset check for the strongest encoding, n=15 — matches Wisconsin's n=15, same statistical footing. Stability check vs the n=5 preliminary read (0.579 +/- 0.130): shift = -0.017, well within the n=5 SE (0.058) — not a meaningful move. Pima (0.562) vs Wisconsin/amplitude (0.619) gap = 0.057, 1.43 SE — NOT resolved (tied, consistent with Wisconsin); vs random guess 2.33 SE — RESOLVED (genuine leakage). |
| wisconsin | angle | iid | 15 | 0.5449 | 0.0883 | — | ok |  |
| wisconsin | amplitude | iid | 15 | 0.6188 | 0.1145 | — | ok |  |
| wisconsin | reuploading | iid | 15 | 0.6441 | 0.1338 | — | ok |  |
| wisconsin | iqp | iid | 14 | 0.6120 | 0.1706 | — | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | angle | alpha_0.1 | 2 | — | — | 0.6000 | insufficient |  |
| wisconsin | amplitude | alpha_0.1 | 5 | 0.5488 | 0.1235 | 0.0000 | ok |  |
| wisconsin | reuploading | alpha_0.1 | 0 | — | — | 1.0000 | absent |  |
| wisconsin | iqp | alpha_0.1 | 4 | 0.6735 | 0.1989 | 0.2000 | ok | seed=7 AUC replaced with 3-draw replication mean=0.848 (MN-09 bimodality) |
| wisconsin | angle | qty_skew | 5 | 0.5456 | 0.1134 | 0.0000 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 0.6388 | 0.1658 | 0.0000 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 0.6916 | 0.1600 | 0.0000 | ok |  |
| wisconsin | iqp | qty_skew | 5 | 0.7216 | 0.1861 | 0.0000 | ok | seed=7 AUC replaced with 3-draw replication mean=0.856 (MN-09) |
