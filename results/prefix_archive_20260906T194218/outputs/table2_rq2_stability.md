# Table 2 — RQ2 Stability: Round-to-Round tau / cosine / L2

| dataset | encoding | distribution | n_final_track_a_seeds | n_total_seeds | tau_mean | tau_std | cos_mean | cos_std | l2_mean | l2_std | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pima | angle | iid | 0 | 5 | — | — | — | — | — | — | absent |  |
| pima | amplitude | iid | 5 | 5 | 0.8800 | 0.0337 | 0.9723 | 0.0301 | 0.0052 | 0.0004 | ok |  |
| pima | reuploading | iid | 5 | 5 | 0.8578 | 0.0298 | 0.9043 | 0.0362 | 0.0151 | 0.0020 | ok |  |
| pima | iqp | iid | 1 | 5 | — | — | — | — | — | — | insufficient |  |
| wisconsin | angle | iid | 15 | 15 | 0.9333 | 0.0403 | 0.9435 | 0.0613 | 0.0200 | 0.0016 | ok |  |
| wisconsin | amplitude | iid | 15 | 15 | 0.9393 | 0.0514 | 0.9643 | 0.0465 | 0.0251 | 0.0014 | ok |  |
| wisconsin | reuploading | iid | 15 | 15 | 0.8785 | 0.0489 | 0.9394 | 0.0602 | 0.0346 | 0.0032 | ok |  |
| wisconsin | iqp | iid | 14 | 15 | 0.8683 | 0.0757 | 0.9332 | 0.0572 | 0.0308 | 0.0040 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | angle | alpha_0.5 | 3 | 5 | 0.8029 | 0.0807 | 0.9276 | 0.1005 | 0.0338 | 0.0224 | ok |  |
| wisconsin | amplitude | alpha_0.5 | 5 | 5 | 0.9111 | 0.0994 | 0.9408 | 0.0659 | 0.0251 | 0.0062 | ok |  |
| wisconsin | reuploading | alpha_0.5 | 5 | 5 | 0.8400 | 0.0553 | 0.9372 | 0.0693 | 0.0289 | 0.0075 | ok |  |
| wisconsin | iqp | alpha_0.5 | 4 | 5 | 0.8222 | 0.1162 | 0.9404 | 0.0479 | 0.0314 | 0.0072 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | angle | alpha_0.1 | 0 | 5 | — | — | — | — | — | — | absent |  |
| wisconsin | amplitude | alpha_0.1 | 5 | 5 | 0.9200 | 0.0600 | 0.9434 | 0.0602 | 0.0176 | 0.0019 | ok |  |
| wisconsin | reuploading | alpha_0.1 | 4 | 5 | 0.8889 | 0.0257 | 0.9574 | 0.0553 | 0.0416 | 0.0019 | ok |  |
| wisconsin | iqp | alpha_0.1 | 1 | 5 | — | — | — | — | — | — | insufficient | IQP/seed=7 excluded (MN-09) |
| wisconsin | angle | qty_skew | 5 | 5 | 0.9378 | 0.0365 | 0.9407 | 0.0599 | 0.0200 | 0.0006 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 5 | 0.9156 | 0.0655 | 0.9399 | 0.0653 | 0.0260 | 0.0018 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 5 | 0.8800 | 0.1129 | 0.9637 | 0.0539 | 0.0355 | 0.0073 | ok |  |
| wisconsin | iqp | qty_skew | 4 | 5 | 0.7444 | 0.0588 | 0.9353 | 0.0554 | 0.0345 | 0.0042 | ok | IQP/seed=7 excluded (MN-09) |
