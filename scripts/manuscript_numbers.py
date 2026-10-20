"""Numbers and tables for the manuscript, all from outputs/ (gdm_ccw.json, psm.json, balance.tsv, flow_counts.tsv) and
work/counts_gdm.tsv (baseline characteristics). Writes outputs/manuscript_numbers.json. Differences and risk ratios are
reported as delivery within the week versus expectant management (negative difference: fewer events with delivery)."""
import os, json, subprocess, platform
import numpy as np, pandas as pd, statsmodels
import gdm_ccw as g

ROOT = g.ROOT; OUT = g.OUT
R = json.load(open(os.path.join(OUT, "gdm_ccw.json"))); PS = json.load(open(os.path.join(OUT, "psm.json")))
N = {}
c0 = lambda x: f"{int(round(x)):,}"
def f1(x): return f"{x:,.1f}".replace("-", "−")
def f2(x): return f"{x:.2f}".replace("-", "−")
def pct(a, b): return f"{100 * a / b:.1f}"
CLAB = {"lt25": "less than 25.0", "over": "25.0 to 29.9", "ob30": "30.0 to 34.9", "ob35p": "35.0 or higher", "all": "all"}
TLAB = {"lt25": "Less than 25.0", "over": "25.0–29.9", "ob30": "30.0–34.9", "ob35p": "35.0 or higher", "all": "All"}

def contrast(v):
    """CCW summary (B minus A) to delivery-minus-expectant difference, RR delivery/expectant, NND or NNH."""
    d, lo, hi = -v["rd"], -v["rd_hi"], -v["rd_lo"]
    inv = lambda z: 1 / z if z else float("inf")
    rr = inv(v["rr"]) if v["rr"] is not None else float("nan"); rrlo = inv(v["rr_hi"]); rrhi = inv(v["rr_lo"])
    clear = hi < 0 or lo > 0; nn = round(1e4 / abs(d)) if clear and d != 0 else None
    return dict(a=v["risk_A"], b=v["risk_B"], d=d, lo=lo, hi=hi, rr=rr, rrlo=rrlo, rrhi=rrhi, nn=nn, nnkind=("NND" if d < 0 else "NNH") if nn else "", p=1 - v["p_B_higher"])

def put(prefix, v):
    x = contrast(v)
    N[f"{prefix}_a"], N[f"{prefix}_b"] = f1(x["a"]), f1(x["b"]); N[f"{prefix}_d"] = f1(x["d"]); N[f"{prefix}_ci"] = f"{f1(x['lo'])} to {f1(x['hi'])}"
    N[f"{prefix}_absd"] = f1(abs(x["d"])); N[f"{prefix}_absci"] = f"{f1(min(abs(x['lo']), abs(x['hi'])))} to {f1(max(abs(x['lo']), abs(x['hi'])))}"
    N[f"{prefix}_rr"] = f2(x["rr"]); N[f"{prefix}_rrci"] = f"{f2(x['rrlo'])}–{f2(x['rrhi'])}"; N[f"{prefix}_nn"] = f"{x['nn']:,}" if x["nn"] else "not estimated"
    N[f"{prefix}_pfewer"] = f"{1 - x['p']:.2f}" if x["d"] < 0 else f"{x['p']:.2f}"
    return x

# flow
fl = dict(pd.read_csv(os.path.join(OUT, "flow_counts.tsv"), sep="\t", header=None).values)
N["flow_all"] = c0(sum(fl.values())); names = ["multiple", "ga", "bmi", "nogdm", "pdm_cht", "prevcs_pres", "anom"]
for i, nm in enumerate(names, 1): N[f"flow_ex_{nm}"] = c0(fl[i])
N["n_cohort"] = c0(fl[8])

# cohorts and courses
for w in (38, 39, 40):
    j = R[f"w{w}"]; N[f"n_w{w}"] = c0(j["n_at_risk"])
    N[f"n_I_w{w}"] = c0(sum(m["n_I"] for m in j["models"].values())); N[f"n_S_w{w}"] = c0(sum(m["n_S"] for m in j["models"].values())); N[f"n_O_w{w}"] = c0(sum(m["n_O"] for m in j["models"].values()))
    N[f"pct_I_w{w}"] = pct(sum(m["n_I"] for m in j["models"].values()), j["n_at_risk"])
    N[f"deaths_w{w}"] = c0(sum(sum(v.values()) for v in j["deaths_by_course"].values()))
d = g.load(39); pop = d[d.kind != "inf"]
N["live_w39"] = c0(pop[pop.kind == "nat"].n.sum()); N["sb_w39"] = c0(pop[pop.kind == "fet"].n.sum()); N["nd_w39"] = c0(d[d.kind == "inf"].nd.sum())
N["deaths_obs_w39"] = c0(pop[pop.kind == "fet"].n.sum() + d[d.kind == "inf"].nd.sum())
for k in g.CLS:
    N[f"pctI_{k}"] = pct(R["w39"]["models"][k]["n_I"], sum(R["w39"]["models"][k][x] for x in ("n_I", "n_S", "n_O")))
    N[f"n_{k}"] = c0(sum(R["w39"]["models"][k][x] for x in ("n_I", "n_S", "n_O")))
raw = pd.read_csv(os.path.join(g.W, "counts_gdm.tsv"), sep="\t", names=g.COLS, dtype={"est": str})
raw = raw[(raw.year >= 2019) & (raw.year <= 2024) & (raw.ga >= 39) & (raw.kind != "inf")]
N["n_bmi40"] = c0(raw[raw.bmic >= 5].n.sum())
N["trunc_max"] = f2(max(max(m["trunc_A"], m["trunc_B"]) for w in ("w38", "w39", "w40") for m in R[w]["models"].values()))
N["smd_max_w39"] = f"{max(max(v['A_vs_population'], v['B_vs_population']) for v in R['w39']['balance_max_abs'].values()):.3f}"
N["smd_max_all"] = f"{max(max(v['A_vs_population'], v['B_vs_population']) for w in ('w38', 'w39', 'w40') for v in R[w]['balance_max_abs'].values()):.3f}"
N["smd_raw_w39"] = f"{max(v['raw_I_vs_O'] for v in R['w39']['balance_max_abs'].values()):.2f}"
N["nd_unmatched"] = str(sum(R[w]["neonatal_deaths_unmatched"] for w in ("w38", "w39", "w40")))

# primary and secondary results
OUTS = ["death", "stillbirth", "neonatal", "cesar", "macro", "nicu", "morb", "aven6", "apg7", "seiz", "plac", "smm", "mtr", "rupt", "uhyst", "aicu"]
X = {}
for w in ("w38", "w39", "w40"):
    for o in OUTS:
        for k in g.CLS + ["all"]:
            X[(w, o, k)] = put(f"{w}_{o}_{k}", R[w]["primary"][o][k])
for k in g.CLS[1:]:
    m = R["w39"]["modification_vs_lt25"]["death"][k]; N[f"mod_{k}"] = f1(-m["diff"]); N[f"mod_{k}_ci"] = f"{f1(-m['hi'])} to {f1(-m['lo'])}"
    m = R["w40"]["modification_vs_lt25"]["death"][k]; N[f"mod40_{k}"] = f1(-m["diff"]); N[f"mod40_{k}_ci"] = f"{f1(-m['hi'])} to {f1(-m['lo'])}"
N["ratio_ob35_lt25"] = f1(X[("w39", "death", "ob35p")]["d"] / X[("w39", "death", "lt25")]["d"])
for w in ("w38", "w39", "w40"): N[f"eval_{w}"] = f2(R[w]["evalue_death"]["all"]["e_point"]); N[f"evallim_{w}"] = f2(R[w]["evalue_death"]["all"]["e_ci_limit"])
SENS = {"exclude_gestational_hypertension": "Gestational hypertension or preeclampsia excluded", "late_induction_to_I": "Induced births in week 40 counted as delivery in week 39",
        "no_prelabor_cesarean_week_w": "Prelabor cesarean births in week 39 excluded", "cohort_2016_2024": "Births 2016–2024"}
for s in SENS:
    for o in ("death", "cesar", "macro", "nicu"): put(f"sens_{s}_{o}", R["w39"]["sensitivity"][s][o]["all"])
    N[f"sens_{s}_n"] = c0(R["w39"]["sensitivity"][s]["n_at_risk"])
dd = [contrast(R["w39"]["sensitivity"][s]["death"]["all"])["d"] for s in SENS]
N["sens_death_range"] = f"{f1(min(dd, key=abs))} to {f1(max(dd, key=abs))}"
for x in ("0.25", "0.5"):
    q = R["w39"]["qba"][x]["all"]; N[f"qba_{x[2:]}"] = f1(-q["rd"]); N[f"qba_{x[2:]}_ci"] = f"{f1(-q['rd_hi'])} to {f1(-q['rd_lo'])}"
# propensity matched
N["psm_pairs"] = c0(PS["n_pairs"]); N["psm_pct"] = f"{PS['pct_treated_matched']:.1f}"; N["psm_smd"] = f"{PS['smd_max_after']:.3f}"; N["psm_smd_before"] = f"{PS['smd_max_before']:.2f}"
for k in ["all"] + g.CLS:
    for o in ("death", "cesar", "macro", "nicu", "morb", "smm"):
        v = PS[k][o]; d_, lo_, hi_ = -v["rd"], -v["rd_hi"], -v["rd_lo"]
        N[f"psm_{o}_{k}"] = f1(d_); N[f"psm_{o}_{k}_ci"] = f"{f1(lo_)} to {f1(hi_)}"; N[f"psm_{o}_{k}_rr"] = f2(1 / v["rr"]); N[f"psm_{o}_{k}_rrci"] = f"{f2(1 / v['rr_hi'])}–{f2(1 / v['rr_lo'])}"
        N[f"psm_{o}_{k}_ri"] = f1(v["risk_induced"]); N[f"psm_{o}_{k}_rn"] = f1(v["risk_not"])

# Table 1: baseline characteristics by observed course in week 39 (I induced and born in week 39, S other end of pregnancy in week 39, O ongoing at 40 0/7 weeks)
def courses(x):
    i = x[(x.ga == 39) & (x.induct == 1) & (x.kind == "nat")]; o = x[x.ga >= 40]; return [i, x[(x.ga == 39)].drop(i.index), o, x]
G1 = courses(pop)
def line(lab, groups, f):
    return f"| {lab} | " + " | ".join(f"{c0(f(x))} ({pct(f(x), x.n.sum())})" for x in groups) + " |"
def rows(var, labels, title):
    out = [f"| **{title}** | | | | |"]
    for code, lab in labels:
        out.append(line(lab, G1, lambda x: x[x[var].isin(code if isinstance(code, list) else [code])].n.sum()))
    return out
rawpop = raw  # 7 BMI classes for description
def rows_raw(title):
    out = [f"| **{title}** | | | | |"]
    for code, lab in [([0, 1], "Less than 25.0"), ([2], "25.0–29.9"), ([3], "30.0–34.9"), ([4], "35.0–39.9"), ([5, 6], "40.0 or higher")]:
        out.append(line(lab, courses(rawpop), lambda x: x[x.bmic.isin(code)].n.sum()))
    return out
T1 = ["| Characteristic | Labor induction with birth in week 39 | Birth without induction or stillbirth in week 39* | Continued to 40 0/7 weeks | All |", "|---|---|---|---|---|",
      "| Number | " + " | ".join(c0(x.n.sum()) for x in G1) + " |"]
T1 += rows_raw("Prepregnancy BMI, kg/m²")
T1 += rows("age6", [(0, "Younger than 20"), (1, "20–24"), (2, "25–29"), (3, "30–34"), (4, "35–39"), (5, "40 or older")], "Maternal age, y")
T1 += rows("nullip", [(1, "Nulliparous"), (0, "Parous"), (9, "Unknown")], "Parity")
T1 += rows("race", [(3, "American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, or more than one race"), (4, "Asian"), (2, "Black"), (7, "Hispanic"), (1, "White"), (8, "Unknown")], "Race and ethnicity†")
T1 += rows("educ", [(1, "High school or less"), (2, "Some college or associate degree"), (3, "Bachelor's degree or higher"), (9, "Unknown")], "Education")
T1 += rows("precare", [(1, "First trimester"), (2, "Second trimester"), (3, "Third trimester"), (4, "None"), (9, "Unknown")], "Prenatal care began")
T1 += rows("wic", [(1, "Yes")], "WIC participation")
T1 += rows("smoke", [(1, "Yes")], "Smoking during pregnancy")
T1 += rows("inftr", [(1, "Yes")], "Infertility treatment")
T1 += rows("ghyp", [(1, "Yes")], "Gestational hypertension or preeclampsia")
T1 += ["| **End of pregnancy** | | | | |", line("Stillbirth", G1, lambda x: x[x.kind != "nat"].n.sum()),
       line("Born in week 40", G1, lambda x: x[(x.ga == 40) & (x.kind == "nat")].n.sum()), line("Born at 41 weeks or later", G1, lambda x: x[(x.ga >= 41) & (x.kind == "nat")].n.sum())]
N["table1"] = "\n".join(T1)
N["pct_nullip"] = pct(pop[pop.nullip == 1].n.sum(), pop.n.sum()); N["pct_ob30plus"] = pct(raw[raw.bmic >= 3].n.sum(), raw.n.sum())
N["pct_ghyp"] = pct(pop[pop.ghyp == 1].n.sum(), pop.n.sum()); N["pct_race_unknown"] = pct(pop[pop.race.isin([8, 9])].n.sum(), pop.n.sum())

# Table 2: week 39 outcomes, all classes, plus perinatal death by BMI
ON = {"death": "Perinatal death", "stillbirth": "Stillbirth", "neonatal": "Neonatal death", "nicu": "NICU admission", "morb": "Composite neonatal morbidity†",
      "aven6": "Assisted ventilation longer than 6 hours", "apg7": "5-minute Apgar score less than 7", "seiz": "Neonatal seizures", "macro": "Birth weight 4,500 g or more",
      "cesar": "Cesarean delivery‡", "plac": "Third- or fourth-degree perineal laceration", "smm": "Severe maternal morbidity§", "mtr": "Maternal transfusion",
      "uhyst": "Unplanned hysterectomy", "rupt": "Uterine rupture", "aicu": "Maternal ICU admission"}
def trow(lab, x): return f"| {lab} | {f1(x['a'])} | {f1(x['b'])} | {f1(x['d'])} ({f1(x['lo'])} to {f1(x['hi'])}) | {f2(x['rr'])} ({f2(x['rrlo'])}–{f2(x['rrhi'])}) | {(x['nnkind'] + ' ' + format(x['nn'], ',')) if x['nn'] else '—'} |"
T2 = ["| Outcome | Delivery in week 39 | Expectant management | Difference (95% CrI) | Risk ratio (95% CrI) | NND or NNH |", "|---|---|---|---|---|---|",
      "| **Perinatal outcomes, per 10,000 pregnancies** | | | | | |"]
for o in ("death", "stillbirth", "neonatal"): T2.append(trow(ON[o], X[("w39", o, "all")]))
T2.append("| **Perinatal death by prepregnancy BMI, kg/m²** | | | | | |")
for k in g.CLS: T2.append(trow(TLAB[k], X[("w39", "death", k)]))
T2.append("| **Neonatal outcomes, per 10,000 live births** | | | | | |")
for o in ("nicu", "morb", "aven6", "apg7", "seiz", "macro"): T2.append(trow(ON[o], X[("w39", o, "all")]))
T2.append("| **Maternal outcomes, per 10,000 live births** | | | | | |")
for o in ("cesar", "plac", "smm", "mtr", "uhyst", "rupt", "aicu"): T2.append(trow(ON[o], X[("w39", o, "all")]))
N["table2"] = "\n".join(T2).replace("—", "NA")

# Supplemental tables: every outcome by trial and BMI class; sensitivity; propensity matched
def etable(w):
    L = ["| Outcome | BMI, kg/m² | Delivery | Expectant | Difference (95% CrI) | Risk ratio (95% CrI) |", "|---|---|---|---|---|---|"]
    for o in OUTS:
        for k in ["all"] + g.CLS:
            x = X[(w, o, k)]; L.append(f"| {ON[o].rstrip('†‡§') if k == 'all' else ''} | {TLAB[k]} | {f1(x['a'])} | {f1(x['b'])} | {f1(x['d'])} ({f1(x['lo'])} to {f1(x['hi'])}) | {f2(x['rr'])} ({f2(x['rrlo'])}–{f2(x['rrhi'])}) |")
    return "\n".join(L)
for w in ("w38", "w39", "w40"): N[f"etable_{w}"] = etable(w)
L = ["| Analysis | Pregnancies | Perinatal death | Birth weight 4,500 g or more | NICU admission | Cesarean delivery |", "|---|---|---|---|---|---|",
     f"| Primary analysis | {N['n_w39']} | {N['w39_death_all_d']} ({N['w39_death_all_ci']}) | {N['w39_macro_all_d']} ({N['w39_macro_all_ci']}) | {N['w39_nicu_all_d']} ({N['w39_nicu_all_ci']}) | {N['w39_cesar_all_d']} ({N['w39_cesar_all_ci']}) |"]
for s, lab in SENS.items():
    L.append(f"| {lab} | {N[f'sens_{s}_n']} | " + " | ".join(f"{N[f'sens_{s}_{o}_d']} ({N[f'sens_{s}_{o}_ci']})" for o in ("death", "macro", "nicu", "cesar")) + " |")
for x, lab in (("25", "25% of week-39 stillbirths preventable by delivery"), ("5", "50% of week-39 stillbirths preventable by delivery")):
    L.append(f"| {lab} | {N['n_w39']} | {N[f'qba_{x}']} ({N[f'qba_{x}_ci']}) | NA | NA | NA |")
N["etable_sens"] = "\n".join(L)
L = ["| Outcome | BMI, kg/m² | Induced in week 39 | Not induced | Difference (95% CrI) | Risk ratio (95% CrI) |", "|---|---|---|---|---|---|"]
PN = {"death": "Perinatal death", "cesar": "Cesarean delivery", "macro": "Birth weight 4,500 g or more", "nicu": "NICU admission", "morb": "Composite neonatal morbidity", "smm": "Severe maternal morbidity"}
for o, lab in PN.items():
    for k in ["all"] + g.CLS:
        L.append(f"| {lab if k == 'all' else ''} | {TLAB[k]} | {N[f'psm_{o}_{k}_ri']} | {N[f'psm_{o}_{k}_rn']} | {N[f'psm_{o}_{k}']} ({N[f'psm_{o}_{k}_ci']}) | {N[f'psm_{o}_{k}_rr']} ({N[f'psm_{o}_{k}_rrci']}) |")
N["etable_psm"] = "\n".join(L)
bal = []
for w in ("w38", "w39", "w40"):
    for k, v in R[w]["balance_max_abs"].items():
        bal.append(f"| {w[1:]} weeks | {TLAB[k]} | {v['raw_I_vs_O']:.3f} | {v['A_vs_population']:.3f} | {v['B_vs_population']:.3f} | {R[w]['models'][k]['trunc_A']:.2f} | {R[w]['models'][k]['trunc_B']:.2f} |")
N["etable_balance"] = "\n".join(["| Trial | BMI, kg/m² | Course I vs O, unweighted | Delivery vs population | Expectant vs population | Weight truncation, delivery | Weight truncation, expectant |", "|---|---|---|---|---|---|---|"] + bal)

# software versions
N["python_version"] = platform.python_version(); N["statsmodels_version"] = statsmodels.__version__
N["rust_version"] = subprocess.run([os.path.expanduser("~/.cargo/bin/rustc"), "--version"], capture_output=True, text=True).stdout.split()[1]
json.dump(N, open(os.path.join(OUT, "manuscript_numbers.json"), "w"), indent=1)
print(len(N), "numbers;", {k: N[k] for k in ["flow_all", "n_cohort", "w39_death_all_d", "w39_death_all_ci", "w39_death_all_rr", "w39_death_all_nn", "ratio_ob35_lt25", "sens_death_range", "python_version", "rust_version"]})
