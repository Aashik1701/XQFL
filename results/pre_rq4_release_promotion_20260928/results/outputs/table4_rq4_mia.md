# Table 4 — RQ4 Membership Inference: Direct-MIA AUC

| dataset | encoding | distribution | n | auc_mean | auc_std | collapse_rate | status | note |
|---|---|---|---|---|---|---|---|---|
| pima | amplitude | iid | 15 | 0.5637 | 0.1086 | 0.0000 | ok | RQ4 extension: cross-dataset check for the strongest encoding, n=15 — matches Wisconsin's n=15, same statistical footing. Stability check vs the n=5 preliminary read (0.579 +/- 0.130): shift = -0.015, well within the n=5 SE (0.058) — not a meaningful move. Pima (0.564) vs Wisconsin/amplitude (0.620) gap = 0.056, 1.40 SE — NOT resolved (tied, consistent with Wisconsin); vs random guess 2.27 SE — RESOLVED (genuine leakage). |
| wisconsin | angle | iid | 15 | 0.5327 | 0.0799 | — | ok |  |
| wisconsin | amplitude | iid | 15 | 0.6200 | 0.1118 | — | ok |  |
| wisconsin | reuploading | iid | 15 | 0.6652 | 0.1365 | — | ok |  |
| wisconsin | iqp | iid | 15 | 0.6469 | 0.1577 | — | ok |  |
| wisconsin | angle | alpha_0.1 | 1 | — | — | 0.8000 | insufficient |  |
| wisconsin | amplitude | alpha_0.1 | 5 | 0.5732 | 0.1667 | 0.0000 | ok |  |
| wisconsin | reuploading | alpha_0.1 | 4 | 0.6510 | 0.1993 | 0.2000 | ok |  |
| wisconsin | iqp | alpha_0.1 | 5 | 0.7008 | 0.1529 | 0.0000 | ok |  |
| wisconsin | angle | qty_skew | 5 | 0.5564 | 0.1181 | 0.0000 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 0.6428 | 0.1598 | 0.0000 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 0.7000 | 0.1693 | 0.0000 | ok |  |
| wisconsin | iqp | qty_skew | 5 | 0.6996 | 0.1899 | 0.0000 | ok |  |
