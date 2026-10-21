"""PLAN (2026-10-18): clone-censor-weight emulation of delivery within week w (A) versus expectant management through
week w (B) in singleton GDM pregnancies ongoing at w+0, w in 38, 39, 40 (primary 39), 2019-2024. Courses: I induced and
born in week w, S other end of pregnancy in week w, O ongoing at w+1. A: S weight 1, I weight 1 + P(O|X)/P(I|X), O
censored. B: I censored, S and O weight 1/(1 - P(I|X)). Point fit and Bayesian bootstrap (both treatment models refitted
in every draw) in rust_ccw. Input work/counts_gdm.tsv (rust vitals gdm mode)."""
import os, sys, json, time, subprocess
import numpy as np, pandas as pd
from util import design, evalue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs")
COLS = ["kind", "year", "ga", "bmic", "bi", "age6", "nullip", "race", "educ", "precare", "wic", "smoke", "height", "inftr",
        "induct", "payer", "married", "trial", "sga", "ghyp", "est", "n", "nicu", "cesar", "morb", "nd",
        "apg7", "aven1", "aven6", "surf", "anti", "seiz", "mtr", "plac", "rupt", "uhyst", "aicu", "smm", "macro"]
CAT = ["year", "age6", "nullip", "race", "educ", "precare", "wic", "smoke", "height", "inftr", "ghyp"]
KEYS = ["bi"] + CAT
# PLAN classes; BMI 40+ (48,324 at risk in the 39-week trial, under 50,000) merged with 35.0-39.9 by the registered rule
CLS = ["lt25", "over", "ob30", "ob35p"]
CMAP = {0: "lt25", 1: "lt25", 2: "over", 3: "ob30", 4: "ob35p", 5: "ob35p", 6: "ob35p"}
LIVE = ["cesar", "macro", "nicu", "morb", "plac", "smm", "mtr", "rupt", "uhyst", "aicu", "apg7", "aven1", "aven6", "seiz"]
OUTC = {"death": ("deaths", "risk"), "stillbirth": ("sb", "risk"), "neonatal": ("nd", "risk"), "sbS": ("sbS", "risk")}
for _v in LIVE: OUTC[_v] = (_v, "live")
S = 1e4
RUST = os.path.join(ROOT, "rust_ccw", "target", "release", "ccw")


def load(w, y0=2019, y1=2024):
    d = pd.read_csv(os.path.join(W, "counts_gdm.tsv"), sep="\t", names=COLS, dtype={"est": str})
    d = d[(d.year >= y0) & (d.year <= y1) & (d.ga >= w)].copy()
    d["bmic"] = d.bmic.map(CMAP)
    d["course"] = np.select([(d.ga == w) & (d.induct == 1) & (d.kind != "fet"), d.ga == w], [0, 1], 2)
    return d


def build(d):
    pop, nat, inf = d[d.kind != "inf"], d[d.kind == "nat"], d[d.kind == "inf"]
    g = lambda x, col: x.groupby(KEYS + ["bmic", "course"])[col].sum()
    fet = pop[pop.kind == "fet"]
    c = pd.concat([g(pop, "n").rename("risk"), g(fet, "n").rename("sb"), g(inf, "nd").rename("nd"), g(nat, "n").rename("live")]
                  + [g(nat, v) for v in LIVE], axis=1).fillna(0).reset_index()
    lost = int(c[c.risk == 0].nd.sum()); c = c[c.risk > 0].reset_index(drop=True)
    c["live"] = np.minimum(c.live, c.risk)
    c["deaths"] = np.minimum(c.sb + c.nd, c.risk)
    c["sbS"] = np.where(c.course == 1, c.sb, 0)
    return c, lost


def prep(c):
    P = {}
    for k in CLS:
        idx = np.where(c.bmic.values == k)[0]
        ck = c.iloc[idx]
        pid = ck.groupby(KEYS, sort=False).ngroup().values
        first = pd.Series(np.arange(len(ck))).groupby(pid).first().values
        pat = ck.iloc[first][KEYS].reset_index(drop=True)
        P[k] = dict(idx=idx, pid=pid, course=ck.course.values, X=design(pat, KEYS).values, npat=len(pat))
    return P


def export(c, P, outdir):
    os.makedirs(outdir, exist_ok=True)
    pid = np.zeros(len(c)); clsv = np.zeros(len(c))
    for i, k in enumerate(CLS): pid[P[k]["idx"]] = P[k]["pid"]; clsv[P[k]["idx"]] = i
    cols = [clsv, pid, c.course.values.astype(float), c.risk.values.astype(float), c.live.values.astype(float)] + [c[e].values.astype(float) for e, _ in OUTC.values()]
    np.concatenate(cols).astype("<f8").tofile(os.path.join(outdir, "cells.bin"))
    for i, k in enumerate(CLS): np.ascontiguousarray(P[k]["X"], dtype="<f8").tofile(os.path.join(outdir, f"X_{i}.bin"))
    json.dump(dict(ncells=len(c), outcomes=[dict(name=o, den=den) for o, (_, den) in OUTC.items()],
                   classes=[dict(name=k, npat=P[k]["npat"], p=P[k]["X"].shape[1]) for k in CLS]), open(os.path.join(outdir, "meta.json"), "w"))


def run(d, ndraw, seed, tag, threads=48, timing=None):
    t0 = time.time(); c, lost = build(d); P = prep(c); t1 = time.time()
    outdir = os.path.join(W, "rust_ccw", tag); export(c, P, outdir)
    subprocess.run([RUST, "point", outdir], check=True, stdout=subprocess.DEVNULL); t2 = time.time()
    subprocess.run([RUST, "draws", outdir, str(ndraw), str(seed), str(threads)], check=True, stdout=subprocess.DEVNULL); t3 = time.time()
    if timing is not None: timing.update(cells=len(c), build_s=round(t1 - t0, 1), point_s=round(t2 - t1, 1), draws_s=round(t3 - t2, 1), ndraw=ndraw)
    pj = json.load(open(os.path.join(outdir, "point.json")))
    arr = np.fromfile(os.path.join(outdir, "draws.bin"), dtype="<f8").reshape(ndraw, len(OUTC), len(CLS), 2)
    outs = list(OUTC)
    pt = {o: {k: tuple(pj["risks"][oi][ki]) for ki, k in enumerate(CLS)} for oi, o in enumerate(outs)}
    draws = {o: {k: arr[:, oi, ki, :] for ki, k in enumerate(CLS)} for oi, o in enumerate(outs)}
    # all classes combined: standardise class risks to the at-risk (or live-birth) distribution of the trial population
    for o, (e, den) in OUTC.items():
        sh = c.groupby("bmic")[den].sum().reindex(CLS).values; sh = sh / sh.sum()
        pt[o]["all"] = tuple(float(np.dot(sh, [pt[o][k][j] for k in CLS])) for j in (0, 1))
        draws[o]["all"] = np.tensordot(arr[:, outs.index(o), :, :], sh, axes=([1], [0]))
    info = {}
    for ki, k in enumerate(CLS):
        cr = P[k]["course"]; r = c.risk.values[P[k]["idx"]]
        info[k] = dict(n_I=int(r[cr == 0].sum()), n_S=int(r[cr == 1].sum()), n_O=int(r[cr == 2].sum()),
                       trunc_A=round(pj["trunc"][ki][0], 2), trunc_B=round(pj["trunc"][ki][1], 3), converged=pj["converged"][ki])
    return dict(c=c, info=info, lost=lost, pt=pt, draws=draws)


q = lambda v, p: round(float(np.nanquantile(v, p)), 3)


def summarise(R, outcomes=None):
    """Posterior summaries from the Bayesian bootstrap draws: medians, 95% credible intervals (2.5th and 97.5th percentiles)
    and posterior probabilities. rd is expectant minus delivery per 10,000 (positive = fewer events with delivery).
    The plug-in estimate from the full-data fit is kept as *_plugin for reference."""
    out = {}
    for o in (outcomes or R["pt"]):
        out[o] = {}
        for k, (a, b) in R["pt"][o].items():
            dd = R["draws"][o][k]; rd = (dd[:, 1] - dd[:, 0]) * S; rr = dd[:, 1] / np.where(dd[:, 0] > 0, dd[:, 0], np.nan)
            out[o][k] = dict(risk_A=q(dd[:, 0] * S, .5), risk_B=q(dd[:, 1] * S, .5), rd=q(rd, .5), rd_lo=q(rd, .025), rd_hi=q(rd, .975),
                             rr=q(rr, .5) if a > 0 else None, rr_lo=q(rr, .025), rr_hi=q(rr, .975), p_B_higher=round(float(np.mean(rd > 0)), 4),
                             p_fewer_ge={str(t): round(float(np.mean(rd >= t)), 4) for t in (1, 5, 10)},
                             p_more_ge={str(t): round(float(np.mean(rd <= -t)), 4) for t in (1, 5, 10)},
                             risk_A_plugin=round(a * S, 3), risk_B_plugin=round(b * S, 3), rd_plugin=round((b - a) * S, 3))
    return out


def balance(R, outdir):
    c = R["c"]; w = np.fromfile(os.path.join(outdir, "point_weights.bin"), dtype="<f8").reshape(2, -1); wA, wB = w[0], w[1]; rows = []
    for k in CLS:
        ck = c[c.bmic == k]; i = ck.index.values
        D = pd.get_dummies(ck[CAT].astype(str)).astype(float); D["bmi"] = ck.bi.values
        w0 = ck.risk.values; cr = ck.course.values
        for v in D.columns:
            x = D[v].values; m0 = np.average(x, weights=w0); sd = np.sqrt(np.average((x - m0) ** 2, weights=w0)) or 1.0
            mI = np.average(x[cr == 0], weights=w0[cr == 0]); mO = np.average(x[cr == 2], weights=w0[cr == 2])
            mA = np.average(x, weights=w0 * wA[i]); mB = np.average(x, weights=w0 * wB[i])
            rows.append(dict(bmic=k, var=v, raw_I_vs_O=(mI - mO) / sd, A_vs_population=(mA - m0) / sd, B_vs_population=(mB - m0) / sd))
    return pd.DataFrame(rows)


def trial(w, ndraw, seed, tag, d=None):
    d = load(w) if d is None else d
    R = run(d, ndraw, seed, tag); js = dict(week=w, n_at_risk=int(R["c"].risk.sum()), neonatal_deaths_unmatched=R["lost"], models=R["info"])
    js["deaths_by_course"] = {k: {cname: int(R["c"][(R["c"].bmic == k) & (R["c"].course == ci)][["sb", "nd"]].sum().sum())
                                  for ci, cname in enumerate("ISO")} for k in CLS}
    js["primary"] = summarise(R)
    rdd = {(o, k): (R["draws"][o][k][:, 1] - R["draws"][o][k][:, 0]) * S for o in R["draws"] for k in R["draws"][o]}
    js["modification_vs_lt25"] = {o: {k: dict(diff=round(float(np.median(rdd[(o, k)] - rdd[(o, "lt25")])), 3), lo=q(rdd[(o, k)] - rdd[(o, "lt25")], .025),
                                              hi=q(rdd[(o, k)] - rdd[(o, "lt25")], .975), p_larger=round(float((rdd[(o, k)] > rdd[(o, "lt25")]).mean()), 4))
                                      for k in CLS[1:]} for o in ["death", "cesar", "macro", "nicu", "morb"]}
    js["evalue_death"] = {k: dict(rr=v["rr"], e_point=evalue(v["rr"]), e_ci_limit=(evalue(v["rr_lo"]) if v["rr_lo"] > 1 else (evalue(v["rr_hi"]) if v["rr_hi"] < 1 else 1.0)))
                          for k, v in js["primary"]["death"].items()}
    js["qba"] = {}
    for x in (0.25, 0.5):
        js["qba"][str(x)] = {}
        for k in CLS + ["all"]:
            dA = R["draws"]["death"][k][:, 0] - x * R["draws"]["sbS"][k][:, 0]; dB = R["draws"]["death"][k][:, 1]
            rd = (dB - dA) * S; rr = dB / dA
            js["qba"][str(x)][k] = dict(rd=q(rd, .5), rd_lo=q(rd, .025), rd_hi=q(rd, .975), rr=q(rr, .5), rr_lo=q(rr, .025), rr_hi=q(rr, .975))
    bal = balance(R, os.path.join(W, "rust_ccw", tag)); bal.round(4).to_csv(os.path.join(OUT, f"balance_w{w}.tsv"), sep="\t", index=False)
    js["balance_max_abs"] = {k: {c_: round(float(bal[bal.bmic == k][c_].abs().max()), 3) for c_ in ["raw_I_vs_O", "A_vs_population", "B_vs_population"]} for k in CLS}
    return js, d


def sensitivity(w, d):
    sens = {}
    d1 = d.copy(); d1.loc[(d1.ga == w + 1) & (d1.induct == 1) & (d1.kind != "fet"), "course"] = 0
    variants = (("exclude_gestational_hypertension", d[d.ghyp != 1]), ("late_induction_to_I", d1),
                ("no_prelabor_cesarean_week_w", d[~((d.kind != "fet") & (d.ga == w) & (d.induct != 1) & (d.trial == 0))]),
                ("cohort_2016_2024", load(w, 2016, 2024)))
    for i, (name, dd) in enumerate(variants):
        R = run(dd, 400, 20000 + 1000 * i, f"w{w}_{name}")
        sens[name] = summarise(R, ["death", "cesar", "macro", "nicu"]); sens[name]["n_at_risk"] = int(R["c"].risk.sum())
    return sens


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "timing"
    if mode == "timing":
        # computation check only: cell counts and run times, no outcome is printed
        for w in (39, 38):
            tm = {}; run(load(w), 50, 1, f"timing_w{w}", timing=tm); print("week", w, tm, flush=True)
    elif mode == "main":
        os.makedirs(OUT, exist_ok=True); res = {}
        for w, seed in ((39, 7000), (40, 8000), (38, 9000)):
            t0 = time.time(); js, d = trial(w, 1000, seed, f"primary_w{w}")
            js["sensitivity"] = sensitivity(w, d) if w == 39 else {}
            res[f"w{w}"] = js; json.dump(res, open(os.path.join(OUT, "gdm_ccw.json"), "w"), indent=1)
            print("week", w, "done", round(time.time() - t0, 1), "s", flush=True)
