"""Shared helpers copied from the ARRIVE project (iptw.py): design matrix, weighted quantile, E-value."""
import numpy as np, pandas as pd, statsmodels.api as sm


def design(p, keys):
    X = pd.get_dummies(p[[k for k in keys if k != "bi"]].astype(str), drop_first=True).astype(float)
    b = (p.bi - p.bi.mean()) / max(p.bi.std(), 1e-9)
    X["bi"], X["bi2"] = b, b ** 2
    return sm.add_constant(X, has_constant="add")


def evalue(rr):
    if rr is None or not np.isfinite(rr): return None
    rr = rr if rr >= 1 else 1 / rr
    return round(float(rr + np.sqrt(rr * (rr - 1))), 2)
