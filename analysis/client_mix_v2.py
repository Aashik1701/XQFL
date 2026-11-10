"""Client sizes and class mix after partitioning, primary setting (4 clients), all 30 seeds.

Reads the data and partition code only; trains nothing:
    PYTHONPATH=. xqfl-env/bin/python analysis/client_mix_v2.py
"""
import numpy as np
import pandas as pd

from v2 import config
from v2.data import make_data, partition_indices

rows = []
for dataset in config.DATASETS:
    for seed in config.SEEDS:
        data = make_data(dataset, seed)
        for regime in config.REGIMES:
            parts = partition_indices(data.y_train, 4, regime, seed)
            sizes = np.array([len(p) for p in parts])
            shares = np.array([data.y_train[p].mean() if len(p) else np.nan for p in parts])
            rows.append(dict(dataset=dataset, regime=regime, seed=seed, train_positive=data.y_train.mean(),
                             size_min=sizes.min(), size_max=sizes.max(), share_min=np.nanmin(shares),
                             share_max=np.nanmax(shares), sparse=int((sizes < config.SPARSE_CLIENT).sum()),
                             single_class=int(sum(len(np.unique(data.y_train[p])) < 2 for p in parts if len(p)))))
frame = pd.DataFrame(rows)
summary = frame.groupby(["dataset", "regime"], sort=False).agg(
    train_positive=("train_positive", "mean"), size_min=("size_min", "mean"), size_max=("size_max", "mean"),
    share_min=("share_min", "mean"), share_max=("share_max", "mean"),
    runs_with_sparse=("sparse", lambda s: int((s > 0).sum())),
    runs_with_single_class=("single_class", lambda s: int((s > 0).sum()))).reset_index()
summary.to_csv("results/protocol_v2/tables/client_mix.csv", index=False)
print(summary.round(3).to_string(index=False))
