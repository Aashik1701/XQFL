# Table 4 — RQ4 Membership Inference: Direct-MIA AUC

| dataset | encoding | distribution | n | auc_mean | auc_std | collapse_rate | status | n_total | n_track_a | note |
|---|---|---|---|---|---|---|---|---|---|---|
| pima | amplitude | iid | 15 | 0.5011 | 0.0781 | 0.0667 | ok | 15 | 14 | release_repair_v1; mean five outer-fold raw AUCs; all IID targets; Track-A sensitivity reported separately |
| wisconsin | amplitude | alpha_0.1 | 5 | 0.4576 | 0.0740 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | amplitude | iid | 15 | 0.4865 | 0.0684 | 0.0000 | ok | 15 | 15 | release_repair_v1; mean five outer-fold raw AUCs; all IID targets; Track-A sensitivity reported separately |
| wisconsin | amplitude | qty_skew | 5 | 0.4864 | 0.1049 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | angle | alpha_0.1 | 1 | — | — | 0.8000 | insufficient | 5 | 1 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | angle | iid | 15 | 0.4888 | 0.0779 | 0.0667 | ok | 15 | 14 | release_repair_v1; mean five outer-fold raw AUCs; all IID targets; Track-A sensitivity reported separately |
| wisconsin | angle | qty_skew | 5 | 0.4628 | 0.0520 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | iqp | alpha_0.1 | 5 | 0.4672 | 0.0627 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | iqp | iid | 15 | 0.5044 | 0.0587 | 0.0000 | ok | 15 | 15 | release_repair_v1; mean five outer-fold raw AUCs; all IID targets; Track-A sensitivity reported separately |
| wisconsin | iqp | qty_skew | 5 | 0.5280 | 0.0690 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | reuploading | alpha_0.1 | 4 | 0.4960 | 0.0501 | 0.2000 | ok | 5 | 4 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
| wisconsin | reuploading | iid | 15 | 0.4795 | 0.0852 | 0.0000 | ok | 15 | 15 | release_repair_v1; mean five outer-fold raw AUCs; all IID targets; Track-A sensitivity reported separately |
| wisconsin | reuploading | qty_skew | 5 | 0.5200 | 0.1063 | 0.0000 | ok | 5 | 5 | release_repair_v1; mean five outer-fold raw AUCs; Track-A only; at least 3 seeds required |
