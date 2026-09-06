# Table 1 — RQ1 Fidelity: JSD_cent by Encoding x Distribution x Dataset

| dataset | encoding | distribution | n_total_draws | n_track_a | n_valid_jsd | collapse_rate | jsd_cent_mean | jsd_cent_std | status | note |
|---|---|---|---|---|---|---|---|---|---|---|
| pima | angle | iid | 5 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | alpha_0.5 | 15 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | alpha_0.1 | 15 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | qty_skew | 11 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | amplitude | iid | 5 | 5 | 5 | 0.0000 | 0.2074 | 0.1112 | ok |  |
| pima | amplitude | alpha_0.5 | 5 | 3 | 3 | 0.4000 | 0.3111 | 0.0313 | ok |  |
| pima | amplitude | alpha_0.1 | 15 | 6 | 6 | 0.6000 | 0.2239 | 0.0370 | ok |  |
| pima | amplitude | qty_skew | 5 | 5 | 5 | 0.0000 | 0.2367 | 0.1154 | ok |  |
| pima | reuploading | iid | 5 | 5 | 5 | 0.0000 | 0.2990 | 0.1805 | ok |  |
| pima | reuploading | alpha_0.5 | 13 | 6 | 6 | 0.5385 | 0.3428 | 0.0622 | ok |  |
| pima | reuploading | alpha_0.1 | 15 | 2 | 2 | 0.8667 | — | — | insufficient |  |
| pima | reuploading | qty_skew | 5 | 4 | 4 | 0.2000 | 0.3091 | 0.1213 | ok |  |
| pima | iqp | iid | 5 | 1 | 1 | 0.8000 | — | — | insufficient |  |
| pima | iqp | alpha_0.5 | 15 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | iqp | alpha_0.1 | 15 | 1 | 1 | 0.9333 | — | — | insufficient |  |
| pima | iqp | qty_skew | 11 | 4 | 4 | 0.6364 | 0.3285 | 0.1276 | ok |  |
| wisconsin | angle | iid | 5 | 5 | 5 | 0.0000 | 0.1318 | 0.0461 | ok |  |
| wisconsin | angle | alpha_0.5 | 22 | 13 | 13 | 0.4091 | 0.2101 | 0.0742 | ok |  |
| wisconsin | angle | alpha_0.1 | 12 | 1 | 1 | 0.9167 | — | — | insufficient |  |
| wisconsin | angle | qty_skew | 5 | 5 | 5 | 0.0000 | 0.1031 | 0.0488 | ok |  |
| wisconsin | amplitude | iid | 5 | 5 | 5 | 0.0000 | 0.1392 | 0.0280 | ok |  |
| wisconsin | amplitude | alpha_0.5 | 5 | 4 | 4 | 0.0000 | 0.1875 | 0.0612 | ok | seed(s) [123] excluded (RQ3 anomaly resolution) |
| wisconsin | amplitude | alpha_0.1 | 5 | 5 | 5 | 0.0000 | 0.1930 | 0.1108 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 5 | 5 | 0.0000 | 0.0884 | 0.0207 | ok |  |
| wisconsin | reuploading | iid | 5 | 5 | 5 | 0.0000 | 0.1430 | 0.0339 | ok |  |
| wisconsin | reuploading | alpha_0.5 | 15 | 12 | 12 | 0.0000 | 0.2026 | 0.0280 | ok | seed(s) [999] excluded (RQ3 anomaly resolution) |
| wisconsin | reuploading | alpha_0.1 | 15 | 9 | 9 | 0.4000 | 0.1850 | 0.0711 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 5 | 5 | 0.0000 | 0.1870 | 0.0558 | ok |  |
| wisconsin | iqp | iid | 5 | 4 | 4 | 0.0000 | 0.1637 | 0.0442 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | alpha_0.5 | 13 | 10 | 10 | 0.0769 | 0.1985 | 0.0829 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | alpha_0.1 | 15 | 3 | 3 | 0.6667 | 0.1901 | 0.0826 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | qty_skew | 5 | 4 | 4 | 0.0000 | 0.1440 | 0.0384 | ok | IQP/seed=7 excluded (MN-09) |
