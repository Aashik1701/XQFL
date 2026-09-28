# Table 3 — RQ3 Local-to-Global Divergence: mean_{k,t} JSD(phi_k, phi_g) by Distribution (delta vs IID)

| dataset | encoding | distribution | n_total_seeds | n_track_a_seeds | n_track_b_seeds | collapse_rate | local_global_jsd_mean | local_global_jsd_std | status | note | delta_vs_iid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pima | angle | iid | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | amplitude | iid | 5 | 5 | 0 | 0.0000 | 0.3671 | 0.0424 | ok |  | 0.0000 |
| pima | reuploading | iid | 5 | 5 | 0 | 0.0000 | 0.3591 | 0.0859 | ok |  | 0.0000 |
| pima | iqp | iid | 5 | 3 | 2 | 0.4000 | 0.3638 | 0.0529 | ok |  | 0.0000 |
| pima | angle | alpha_0.5 | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | amplitude | alpha_0.5 | 5 | 3 | 2 | 0.4000 | 0.3249 | 0.0428 | ok |  | -0.0422 |
| pima | reuploading | alpha_0.5 | 5 | 5 | 0 | 0.0000 | 0.3968 | 0.0525 | ok |  | 0.0377 |
| pima | iqp | alpha_0.5 | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | angle | alpha_0.1 | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | amplitude | alpha_0.1 | 5 | 3 | 2 | 0.4000 | 0.2912 | 0.0509 | ok |  | -0.0759 |
| pima | reuploading | alpha_0.1 | 5 | 2 | 3 | 0.6000 | — | — | insufficient |  | — |
| pima | iqp | alpha_0.1 | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | angle | qty_skew | 5 | 0 | 5 | 1.0000 | — | — | insufficient |  | — |
| pima | amplitude | qty_skew | 5 | 5 | 0 | 0.0000 | 0.3553 | 0.0303 | ok |  | -0.0118 |
| pima | reuploading | qty_skew | 5 | 5 | 0 | 0.0000 | 0.3358 | 0.0464 | ok |  | -0.0233 |
| pima | iqp | qty_skew | 5 | 2 | 3 | 0.6000 | — | — | insufficient |  | — |
| wisconsin | angle | iid | 5 | 5 | 0 | 0.0000 | 0.4376 | 0.0874 | ok |  | 0.0000 |
| wisconsin | amplitude | iid | 5 | 5 | 0 | 0.0000 | 0.2887 | 0.0440 | ok |  | 0.0000 |
| wisconsin | reuploading | iid | 5 | 5 | 0 | 0.0000 | 0.3355 | 0.1179 | ok |  | 0.0000 |
| wisconsin | iqp | iid | 5 | 5 | 0 | 0.0000 | 0.3687 | 0.0273 | ok |  | 0.0000 |
| wisconsin | angle | alpha_0.5 | 5 | 3 | 2 | 0.4000 | 0.6190 | 0.1286 | ok |  | 0.1814 |
| wisconsin | amplitude | alpha_0.5 | 5 | 5 | 0 | 0.0000 | 0.4614 | 0.0787 | ok |  | 0.1727 |
| wisconsin | reuploading | alpha_0.5 | 5 | 5 | 0 | 0.0000 | 0.4057 | 0.0806 | ok |  | 0.0702 |
| wisconsin | iqp | alpha_0.5 | 5 | 5 | 0 | 0.0000 | 0.4054 | 0.0461 | ok |  | 0.0367 |
| wisconsin | angle | alpha_0.1 | 5 | 1 | 4 | 0.8000 | — | — | insufficient |  | — |
| wisconsin | amplitude | alpha_0.1 | 5 | 5 | 0 | 0.0000 | 0.4464 | 0.0727 | ok |  | 0.1577 |
| wisconsin | reuploading | alpha_0.1 | 5 | 3 | 2 | 0.4000 | 0.4509 | 0.0393 | ok |  | 0.1153 |
| wisconsin | iqp | alpha_0.1 | 5 | 4 | 1 | 0.2000 | 0.4744 | 0.1146 | ok |  | 0.1057 |
| wisconsin | angle | qty_skew | 5 | 5 | 0 | 0.0000 | 0.4578 | 0.1054 | ok |  | 0.0202 |
| wisconsin | amplitude | qty_skew | 5 | 5 | 0 | 0.0000 | 0.2917 | 0.0542 | ok |  | 0.0030 |
| wisconsin | reuploading | qty_skew | 5 | 5 | 0 | 0.0000 | 0.3045 | 0.0445 | ok |  | -0.0310 |
| wisconsin | iqp | qty_skew | 5 | 5 | 0 | 0.0000 | 0.3849 | 0.0767 | ok |  | 0.0162 |
