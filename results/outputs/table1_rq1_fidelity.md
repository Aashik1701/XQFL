# Table 1 — RQ1 Fidelity: JSD_cent by Encoding x Distribution x Dataset

| dataset | encoding | distribution | n_total_draws | n_track_a | n_valid_jsd | collapse_rate | jsd_cent_mean | jsd_cent_std | status | note |
|---|---|---|---|---|---|---|---|---|---|---|
| pima | angle | iid | 5 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | alpha_0.5 | 7 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | alpha_0.1 | 15 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | angle | qty_skew | 7 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | amplitude | iid | 5 | 5 | 5 | 0.0000 | 0.2007 | 0.1010 | ok |  |
| pima | amplitude | alpha_0.5 | 11 | 6 | 6 | 0.4545 | 0.2899 | 0.0243 | ok |  |
| pima | amplitude | alpha_0.1 | 15 | 9 | 9 | 0.4000 | 0.2199 | 0.0392 | ok |  |
| pima | amplitude | qty_skew | 7 | 7 | 7 | 0.0000 | 0.2209 | 0.0740 | ok |  |
| pima | reuploading | iid | 5 | 5 | 5 | 0.0000 | 0.3349 | 0.0866 | ok |  |
| pima | reuploading | alpha_0.5 | 5 | 5 | 5 | 0.0000 | 0.3041 | 0.0705 | ok |  |
| pima | reuploading | alpha_0.1 | 15 | 7 | 7 | 0.5333 | 0.3526 | 0.0409 | ok |  |
| pima | reuploading | qty_skew | 5 | 5 | 5 | 0.0000 | 0.3034 | 0.1499 | ok |  |
| pima | iqp | iid | 5 | 3 | 3 | 0.4000 | 0.2575 | 0.0624 | ok |  |
| pima | iqp | alpha_0.5 | 5 | 0 | 0 | 1.0000 | — | — | insufficient |  |
| pima | iqp | alpha_0.1 | 15 | 2 | 2 | 0.8667 | — | — | insufficient |  |
| pima | iqp | qty_skew | 5 | 2 | 2 | 0.6000 | — | — | insufficient |  |
| wisconsin | angle | iid | 5 | 5 | 5 | 0.0000 | 0.1209 | 0.0241 | ok |  |
| wisconsin | angle | alpha_0.5 | 15 | 9 | 9 | 0.4000 | 0.1541 | 0.0219 | ok |  |
| wisconsin | angle | alpha_0.1 | 15 | 3 | 3 | 0.8000 | 0.1530 | 0.0163 | ok |  |
| wisconsin | angle | qty_skew | 7 | 7 | 7 | 0.0000 | 0.0818 | 0.0440 | ok |  |
| wisconsin | amplitude | iid | 5 | 5 | 5 | 0.0000 | 0.1516 | 0.0357 | ok |  |
| wisconsin | amplitude | alpha_0.5 | 5 | 4 | 4 | 0.0000 | 0.1869 | 0.0414 | ok | seed(s) [123] excluded (RQ3 anomaly resolution) |
| wisconsin | amplitude | alpha_0.1 | 5 | 5 | 5 | 0.0000 | 0.1701 | 0.0774 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 5 | 5 | 0.0000 | 0.1037 | 0.0218 | ok |  |
| wisconsin | reuploading | iid | 5 | 5 | 5 | 0.0000 | 0.1357 | 0.0500 | ok |  |
| wisconsin | reuploading | alpha_0.5 | 7 | 4 | 4 | 0.0000 | 0.2057 | 0.0789 | ok | seed(s) [999] excluded (RQ3 anomaly resolution) |
| wisconsin | reuploading | alpha_0.1 | 15 | 9 | 9 | 0.4000 | 0.2115 | 0.0570 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 5 | 5 | 0.0000 | 0.1739 | 0.0262 | ok |  |
| wisconsin | iqp | iid | 5 | 4 | 4 | 0.0000 | 0.1384 | 0.0665 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | alpha_0.5 | 9 | 8 | 8 | 0.0000 | 0.2211 | 0.0570 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | alpha_0.1 | 15 | 10 | 10 | 0.1333 | 0.2952 | 0.1129 | ok | IQP/seed=7 excluded (MN-09) |
| wisconsin | iqp | qty_skew | 5 | 4 | 4 | 0.0000 | 0.1272 | 0.0608 | ok | IQP/seed=7 excluded (MN-09) |
