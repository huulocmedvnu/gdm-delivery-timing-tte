"""Probabilistic bias analysis for an unmeasured binary confounder (amendment c, 2026-10-27, post hoc), 39-week trial,
perinatal death. Bias factor for a binary confounder U with prevalence p1 under delivery and p0 under expectant management
and risk ratio RR_U with perinatal death (Lash et al. 2014): B = [p1(RR_U - 1) + 1] / [p0(RR_U - 1) + 1], RR_adj = RR_obs / B.
Random error enters through the bootstrap replicates of RR_obs. Scenario A: medication-treated gestational diabetes, more
frequent among pregnancies delivered in week 39. Scenario B: an unmeasured risk factor more frequent among pregnancies
managed expectantly (the direction that would exaggerate the benefit of delivery). Writes outputs/pba.json."""
import os, json, numpy as np
import gdm_ccw as g

R = g.run(g.load(39), 1000, 7000, "primary_w39")          # same seed as the primary analysis, identical replicates
rng = np.random.default_rng(20261027); M = 200_000
SC = {"A_medication_treated": dict(p1=(0.30, 0.50), p0=(0.15, 0.30), rru=(1.5, 3.0)),
      "B_favours_delivery": dict(p1=(0.05, 0.15), p0=(0.10, 0.30), rru=(1.5, 3.0))}
out = {"scenarios": SC, "draws": M}
for k in ("all", "lt25", "ob35p"):
    dd = R["draws"]["death"][k]; rr_rep = dd[:, 0] / dd[:, 1]          # delivery / expectant
    a, b = R["pt"]["death"][k]; rr_pt = a / b
    out[k] = {"rr_observed": round(rr_pt, 3)}
    for name, s in SC.items():
        p1 = rng.uniform(*s["p1"], M); p0 = rng.uniform(*s["p0"], M); ru = rng.uniform(*s["rru"], M)
        bias = (p1 * (ru - 1) + 1) / (p0 * (ru - 1) + 1)
        sys_only = rr_pt / bias
        ok = np.where(np.isfinite(rr_rep))[0]; j = rng.choice(ok, M)
        tot = rr_rep[j] / bias
        rd = dd[j, 1] * (tot - 1) * g.S                              # delivery minus expectant per 10,000, expectant risk from the same replicate
        out[k][name] = dict(rr_sys_median=round(float(np.median(sys_only)), 3),
                            rr_median=round(float(np.median(tot)), 3), rr_lo=round(float(np.quantile(tot, .025)), 3), rr_hi=round(float(np.quantile(tot, .975)), 3),
                            rd_median=round(float(np.median(rd)), 2), rd_lo=round(float(np.quantile(rd, .025)), 2), rd_hi=round(float(np.quantile(rd, .975)), 2),
                            p_rr_ge_1=round(float(np.mean(tot >= 1)), 4))
json.dump(out, open(os.path.join(g.OUT, "pba.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("all", "lt25", "ob35p")}, indent=1))
