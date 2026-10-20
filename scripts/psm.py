"""Amendment a (post hoc): propensity score matched conventional comparison in the 39-week trial population, induced and
born in week 39 (course I) versus not (courses S and O, followed to the end of pregnancy). Logistic propensity on all
registered covariates, BMI linear and quadratic and BMI class; 1:1 nearest neighbour on the logit without replacement,
caliper 0.2 SD, exact within BMI class. Outcomes attributed at random within pattern and course (seed 11). Bayesian
bootstrap over matched pairs. Output outputs/psm.json. Usage: python psm.py [ndraw] [year0 year1]"""
import os, sys, json, time
import numpy as np, pandas as pd, statsmodels.api as sm
import gdm_ccw as g
from util import design

NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
Y0, Y1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (2019, 2024)
OUTS = ["death", "cesar", "macro", "nicu", "morb", "smm"]
rng = np.random.default_rng(11); t0 = time.time(); tm = {}

c, _ = g.build(g.load(39, Y0, Y1)); tm["build_s"] = round(time.time() - t0, 1)
c["treat"] = (c.course == 0).astype(int)
# propensity on patterns (all classes), weights = individuals
pat = c.groupby(g.KEYS + ["bmic", "treat"], sort=False).risk.sum().unstack("treat", fill_value=0).reset_index()
X = design(pat, g.KEYS); X = pd.concat([X, pd.get_dummies(pat.bmic, prefix="cls", drop_first=True).astype(float)], axis=1)
tot = pat[0] + pat[1]
fit = sm.GLM(pat[1] / tot, X, family=sm.families.Binomial(), var_weights=tot).fit()
pat["lp"] = X.values @ fit.params.values
c = c.merge(pat[g.KEYS + ["bmic", "lp"]], on=g.KEYS + ["bmic"], how="left")
tm["ps_s"] = round(time.time() - t0, 1)

# expand cells to individuals and attribute outcomes at random within cell
n = c.risk.values.astype(int); cell = np.repeat(np.arange(len(c)), n); N = len(cell)
start = np.repeat(np.cumsum(n) - n, n)
def rank(key):
    o = np.lexsort((key, cell)); r = np.empty(N, int); r[o] = np.arange(N) - start[o]; return r
r0 = rank(rng.random(N))
sb = c.sb.values[cell].astype(int); live = (r0 >= sb).astype(int)
death = ((r0 < sb) | ((r0 - sb) < c.nd.values[cell])).astype(int)
ind = pd.DataFrame(dict(cell=cell, treat=c.treat.values[cell], bmic=c.bmic.values[cell], lp=c.lp.values[cell], live=live, death=death))
for o in OUTS[1:]:
    rl = rank(np.where(live == 1, rng.random(N), 2.0))
    ind[o] = ((live == 1) & (rl < np.minimum(c[o].values[cell], c.live.values[cell]))).astype(int)
tm["expand_s"] = round(time.time() - t0, 1)

# greedy 1:1 nearest neighbour without replacement, caliper 0.2 SD of the logit, within BMI class
cal = 0.2 * ind.lp.std(); pairs = []
for k in g.CLS:
    T = ind.index[(ind.bmic == k) & (ind.treat == 1)].values; C = ind.index[(ind.bmic == k) & (ind.treat == 0)].values
    C = C[np.argsort(ind.lp.values[C], kind="stable")]; cl = ind.lp.values[C]; m = len(C)
    nxt = np.arange(m + 1); prv = np.arange(-1, m)          # union-find style pointers to available controls
    def fr(i):
        root = i
        while nxt[root] != root: root = nxt[root]
        while nxt[i] != root: nxt[i], i = root, nxt[i]
        return root
    def fl(i):
        root = i
        while root >= 0 and prv[root + 1] != root: root = prv[root + 1]
        while i >= 0 and prv[i + 1] != root: prv[i + 1], i = root, prv[i + 1]
        return root
    for t in rng.permutation(T):
        x = ind.lp.values[t]; p = np.searchsorted(cl, x)
        a = fr(p); b = fl(p - 1)
        da = abs(cl[a] - x) if a < m else np.inf; db = abs(cl[b] - x) if b >= 0 else np.inf
        j = a if da <= db else b; d = min(da, db)
        if d > cal: continue
        pairs.append((t, C[j], k)); nxt[j] = j + 1; prv[j + 1] = j - 1
P = pd.DataFrame(pairs, columns=["t", "c", "bmic"]); tm["match_s"] = round(time.time() - t0, 1)

# balance after matching (standardized differences of every covariate level and BMI)
cov = c[g.CAT + ["bi"]].iloc[ind.cell.values].reset_index(drop=True)
D = pd.get_dummies(cov[g.CAT].astype(str)).astype(float); D["bmi"] = cov.bi.values
def smd(a, b):
    xa, xb = D.values[a], D.values[b]; s = np.sqrt((xa.var(0) + xb.var(0)) / 2); s[s == 0] = 1
    return np.abs(xa.mean(0) - xb.mean(0)) / s
allT = ind.index[ind.treat == 1].values; allC = ind.index[ind.treat == 0].values
res = dict(n_treated=int(len(allT)), n_control=int(len(allC)), n_pairs=int(len(P)), pct_treated_matched=round(100 * len(P) / len(allT), 1),
           caliper=round(float(cal), 4), smd_max_before=round(float(smd(allT, allC).max()), 3), smd_max_after=round(float(smd(P.t.values, P.c.values).max()), 3))

# Bayesian bootstrap over matched pairs
def est(mask):
    tt, cc = P.t.values[mask], P.c.values[mask]; out = {}
    Y = {o: (ind[o].values[tt].astype(float), ind[o].values[cc].astype(float),
             np.ones(len(tt)) if o == "death" else ind.live.values[tt].astype(float),
             np.ones(len(cc)) if o == "death" else ind.live.values[cc].astype(float)) for o in OUTS}
    dr = {o: [] for o in OUTS}
    for b0 in range(0, NDRAW, 50):                     # one weight matrix per batch, shared by all outcomes
        W = rng.gamma(1.0, size=(min(50, NDRAW - b0), len(tt)))
        for o, (yT, yC, dT, dC) in Y.items(): dr[o].append(np.c_[(W @ yT) / (W @ dT), (W @ yC) / (W @ dC)])
    q = lambda v, p: round(float(np.nanquantile(v, p)), 3)
    for o, (yT, yC, dT, dC) in Y.items():
        d = np.vstack(dr[o]); rd = (d[:, 1] - d[:, 0]) * 1e4; rr = d[:, 1] / d[:, 0]; pt = (yT.sum() / dT.sum(), yC.sum() / dC.sum())
        out[o] = dict(risk_induced=round(pt[0] * 1e4, 3), risk_not=round(pt[1] * 1e4, 3), rd=round((pt[1] - pt[0]) * 1e4, 3), rd_lo=q(rd, .025), rd_hi=q(rd, .975),
                      rr=round(pt[1] / pt[0], 3) if pt[0] > 0 else None, rr_lo=q(rr, .025), rr_hi=q(rr, .975), events_induced=int(yT.sum()), events_not=int(yC.sum()))
    return out
res["all"] = est(np.ones(len(P), bool))
for k in g.CLS: res[k] = est(P.bmic.values == k); res[k]["n_pairs"] = int((P.bmic == k).sum())
tm["total_s"] = round(time.time() - t0, 1); res["timing"] = tm; res["ndraw"] = NDRAW; res["years"] = [Y0, Y1]
json.dump(res, open(os.path.join(g.OUT, "psm.json" if (Y0, Y1) == (2019, 2024) and NDRAW >= 1000 else "psm_test.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ["n_treated", "n_control", "n_pairs", "pct_treated_matched", "smd_max_before", "smd_max_after", "timing"]}), flush=True)
