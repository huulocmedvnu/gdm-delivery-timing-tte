"""10th percentile of birthweight by completed week (39-44) and sex among cohort live births (work/bwq.tsv from the
rust bwq mode). Writes work/bw_p10.tsv for the ps aggregation (env BWQ)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); W = os.path.join(ROOT, "work")
h = pd.read_csv(os.path.join(W, "bwq.tsv"), sep="\t", header=None, usecols=range(5), names=["tag", "ga", "sex", "bw", "n"])
rows = []
for (ga, sex), g in h.groupby(["ga", "sex"]):
    g = g.groupby("bw").n.sum().sort_index(); cdf = g.cumsum() / g.sum()
    rows.append((ga, sex, int(g.index[np.searchsorted(cdf.values, 0.10)])))
pd.DataFrame(rows).to_csv(os.path.join(W, "bw_p10.tsv"), sep="\t", header=False, index=False)
print(rows)
