"""Checks that rust_ccw point GLM parameters match statsmodels on the GDM 39-week trial (both treatment models, every
BMI class). No outcome is computed."""
import json, os, numpy as np, statsmodels.api as sm
import gdm_ccw as g

d = g.load(39); c, _ = g.build(d); P = g.prep(c); out = os.path.join(g.W, "rust_ccw", "validate_w39"); g.export(c, P, out)
import subprocess; subprocess.run([g.RUST, "point", out], check=True, stdout=subprocess.DEVNULL)
pj = json.load(open(os.path.join(out, "point.json"))); T = c.risk.values.astype(float); worst = 0
for ki, k in enumerate(g.CLS):
    p = P[k]; t = T[p["idx"]]; cr = p["course"]; X = p["X"]
    n = [np.bincount(p["pid"][cr == a], weights=t[cr == a], minlength=p["npat"]) for a in (0, 1, 2)]
    tot = n[0] + n[1] + n[2]; m = tot > 0; non = n[1] + n[2]; m2 = non > 0
    f1 = sm.GLM(n[0][m] / tot[m], X[m], family=sm.families.Binomial(), var_weights=tot[m]).fit(maxiter=100)
    f2 = sm.GLM(n[2][m2] / non[m2], X[m2], family=sm.families.Binomial(), var_weights=non[m2]).fit(maxiter=100)
    r1, r2 = np.array(pj["params"][ki][0]), np.array(pj["params"][ki][1])
    e = max(np.abs(r1 - f1.params).max(), np.abs(r2 - f2.params).max()); worst = max(worst, e)
    print(k, "p =", X.shape[1], "max |rust - statsmodels| =", f"{e:.2e}", "converged", f1.converged and f2.converged, pj["converged"][ki])
print("WORST", f"{worst:.2e}", "PASS" if worst < 1e-6 else "FAIL")
