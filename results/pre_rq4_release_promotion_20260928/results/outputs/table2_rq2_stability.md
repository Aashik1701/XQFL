# Table 2 — RQ2 Stability: Round-to-Round tau / cosine / L2

| dataset | encoding | distribution | n_final_track_a_seeds | n_total_seeds | tau_mean | tau_std | cos_mean | cos_std | l2_mean | l2_std | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pima | angle | iid | 0 | 5 | — | — | — | — | — | — | absent |  |
| pima | amplitude | iid | 5 | 5 | 0.8533 | 0.0404 | 0.9747 | 0.0283 | 0.0050 | 0.0004 | ok |  |
| pima | reuploading | iid | 5 | 5 | 0.8400 | 0.0596 | 0.9088 | 0.0358 | 0.0151 | 0.0013 | ok |  |
| pima | iqp | iid | 3 | 5 | 0.8000 | 0.0801 | 0.8779 | 0.0110 | 0.0168 | 0.0005 | ok |  |
| wisconsin | angle | iid | 15 | 15 | 0.9348 | 0.0456 | 0.9431 | 0.0617 | 0.0201 | 0.0008 | ok |  |
| wisconsin | amplitude | iid | 15 | 15 | 0.9363 | 0.0556 | 0.9644 | 0.0462 | 0.0247 | 0.0012 | ok |  |
| wisconsin | reuploading | iid | 15 | 15 | 0.8859 | 0.0489 | 0.9395 | 0.0600 | 0.0335 | 0.0025 | ok |  |
| wisconsin | iqp | iid | 15 | 15 | 0.8578 | 0.0915 | 0.9286 | 0.0698 | 0.0308 | 0.0035 | ok |  |
| wisconsin | angle | alpha_0.5 | 3 | 5 | 0.9407 | 0.0463 | 0.9384 | 0.0631 | 0.0178 | 0.0026 | ok |  |
| wisconsin | amplitude | alpha_0.5 | 5 | 5 | 0.9244 | 0.0659 | 0.9408 | 0.0658 | 0.0270 | 0.0024 | ok |  |
| wisconsin | reuploading | alpha_0.5 | 5 | 5 | 0.8044 | 0.1211 | 0.9576 | 0.0500 | 0.0334 | 0.0031 | ok |  |
| wisconsin | iqp | alpha_0.5 | 5 | 5 | 0.8311 | 0.0713 | 0.9248 | 0.0532 | 0.0306 | 0.0066 | ok |  |
| wisconsin | angle | alpha_0.1 | 1 | 5 | — | — | — | — | — | — | insufficient |  |
| wisconsin | amplitude | alpha_0.1 | 5 | 5 | 0.9067 | 0.0760 | 0.9409 | 0.0653 | 0.0228 | 0.0054 | ok |  |
| wisconsin | reuploading | alpha_0.1 | 3 | 5 | 0.9111 | 0.0385 | 0.9500 | 0.0640 | 0.0371 | 0.0039 | ok |  |
| wisconsin | iqp | alpha_0.1 | 4 | 5 | 0.8278 | 0.1124 | 0.9384 | 0.0516 | 0.0249 | 0.0066 | ok |  |
| wisconsin | angle | qty_skew | 5 | 5 | 0.9333 | 0.0351 | 0.9372 | 0.0639 | 0.0209 | 0.0012 | ok |  |
| wisconsin | amplitude | qty_skew | 5 | 5 | 0.9156 | 0.0655 | 0.9401 | 0.0655 | 0.0264 | 0.0020 | ok |  |
| wisconsin | reuploading | qty_skew | 5 | 5 | 0.8978 | 0.0337 | 0.9635 | 0.0533 | 0.0374 | 0.0040 | ok |  |
| wisconsin | iqp | qty_skew | 5 | 5 | 0.8044 | 0.1231 | 0.9212 | 0.0554 | 0.0333 | 0.0062 | ok |  |
