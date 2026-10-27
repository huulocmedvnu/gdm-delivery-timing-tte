"""Numbers and tables for the PPE manuscript (amendment b, frequentist), all from outputs/gdm_ccw.json, flow_counts.tsv and
work/counts_gdm.tsv. Writes outputs/numbers_ppe.json. Contrasts are delivery within the week versus expectant management
(negative difference: fewer events with delivery). PPE style: CIs in round brackets separated by a comma, risk estimates
and CIs to two significant digits for ratios, differences per 10,000 to one decimal, percentages to one decimal."""
import os, json, subprocess, platform
import numpy as np, pandas as pd, statsmodels
import gdm_ccw as g

OUT = g.OUT
R = json.load(open(os.path.join(OUT, "gdm_ccw.json")))
N = {}
c0 = lambda x: f"{int(round(x)):,}"
def f1(x): return (f"{x:.2f}" if 0 < abs(x) < 0.05 else f"{x:,.1f}").replace("-", "−")
def f2(x): return f"{x:.2f}".replace("-", "−")
def pct(a, b): return f"{100 * a / b:.1f}"
def fp(t): return t if t.startswith("<") else "= " + t
def pv(p): return "<0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
TLAB = {"lt25": "<25.0", "over": "25.0 to 29.9", "ob30": "30.0 to 34.9", "ob35p": "≥35.0", "all": "All"}


def contrast(v):
    d, lo, hi = -v["rd"], -v["rd_hi"], -v["rd_lo"]
    inv = lambda z: 1 / z if z else float("inf")
    rr = inv(v["rr"]) if v["rr"] is not None else float("nan"); rrlo = inv(v["rr_hi"]); rrhi = inv(v["rr_lo"])
    nn = v["nnt"]
    return dict(a=v["risk_A"], b=v["risk_B"], d=d, lo=lo, hi=hi, rr=rr, rrlo=rrlo, rrhi=rrhi, nn=nn, nnkind=("NND" if d < 0 else "NNH") if nn else "")


def put(prefix, v):
    x = contrast(v)
    N[f"{prefix}_a"], N[f"{prefix}_b"] = f1(x["a"]), f1(x["b"]); N[f"{prefix}_d"] = f1(x["d"]); N[f"{prefix}_ci"] = f"{f1(x['lo'])}, {f1(x['hi'])}"
    N[f"{prefix}_rr"] = f2(x["rr"]); N[f"{prefix}_rrci"] = f"{f2(x['rrlo'])}, {f2(x['rrhi'])}"; N[f"{prefix}_nn"] = f"{x['nn']:,}" if x["nn"] else "not estimated"
    return x


fl = dict(pd.read_csv(os.path.join(OUT, "flow_counts.tsv"), sep="\t", header=None).values)
N["flow_all"] = c0(sum(fl.values())); names = ["multiple", "ga", "bmi", "nogdm", "pdm_cht", "prevcs_pres", "anom"]
for i, nm in enumerate(names, 1): N[f"flow_ex_{nm}"] = c0(fl[i])
N["n_cohort"] = c0(fl[8]); N["pct_bmi_missing"] = pct(fl[3], sum(fl.values()) - fl[1] - fl[2])

for w in (38, 39, 40):
    j = R[f"w{w}"]; N[f"n_w{w}"] = c0(j["n_at_risk"])
    for c in "ISO": N[f"n_{c}_w{w}"] = c0(sum(m[f"n_{c}"] for m in j["models"].values()))
    N[f"pct_I_w{w}"] = pct(sum(m["n_I"] for m in j["models"].values()), j["n_at_risk"])
d = g.load(39); pop = d[d.kind != "inf"]
N["sb_w39"] = c0(pop[pop.kind == "fet"].n.sum()); N["live_w39"] = c0(pop[pop.kind == "nat"].n.sum()); N["nd_w39"] = c0(d[d.kind == "inf"].nd.sum())
N["pct_death_obs_w39"] = f"{1e4 * (pop[pop.kind == 'fet'].n.sum() + d[d.kind == 'inf'].nd.sum()) / pop.n.sum():.1f}"
for k in g.CLS:
    tot = sum(R["w39"]["models"][k][x] for x in ("n_I", "n_S", "n_O"))
    N[f"pctI_{k}"] = pct(R["w39"]["models"][k]["n_I"], tot); N[f"n_{k}"] = c0(tot)
raw = pd.read_csv(os.path.join(g.W, "counts_gdm.tsv"), sep="\t", names=g.COLS, dtype={"est": str})
raw = raw[(raw.year >= 2019) & (raw.year <= 2024) & (raw.ga >= 39) & (raw.kind != "inf")]
N["n_bmi40"] = c0(raw[raw.bmic >= 5].n.sum())
N["trunc_max"] = f2(max(max(m["trunc_A"], m["trunc_B"]) for w in ("w38", "w39", "w40") for m in R[w]["models"].values()))
N["smd_max_w39"] = f"{max(max(v['A_vs_population'], v['B_vs_population']) for v in R['w39']['balance_max_abs'].values()):.3f}"
N["smd_raw_w39"] = f"{max(v['raw_I_vs_O'] for v in R['w39']['balance_max_abs'].values()):.2f}"
N["nd_unmatched"] = str(sum(R[w]["neonatal_deaths_unmatched"] for w in ("w38", "w39", "w40")))
# missing covariate values among the 39-week cohort
for v, codes in (("race", [8, 9]), ("educ", [9]), ("precare", [9]), ("nullip", [9]), ("smoke", [9]), ("wic", [9]), ("height", [9]), ("inftr", [9])):
    N[f"miss_{v}"] = pct(pop[pop[v].isin(codes)].n.sum(), pop.n.sum())

OUTS = ["death", "stillbirth", "neonatal", "cesar", "macro", "nicu", "morb", "aven6", "apg7", "seiz", "plac", "smm", "mtr", "rupt", "uhyst", "aicu"]
X = {}
for w in ("w38", "w39", "w40"):
    for o in OUTS:
        for k in g.CLS + ["all"]:
            X[(w, o, k)] = put(f"{w}_{o}_{k}", R[w]["primary"][o][k])

# effect modification by BMI (delivery vs expectant): ratio of RRs and difference in RDs vs BMI <25, homogeneity tests
MODO = {"death": "Perinatal death", "stillbirth": "Stillbirth", "nicu": "NICU admission", "morb": "Composite neonatal morbidity", "macro": "Birthweight ≥4500 g",
        "cesar": "Caesarean delivery", "smm": "Severe maternal morbidity"}
for w in ("w39",):
    for o in MODO:
        m = R[w]["modification"][o]
        for k in g.CLS[1:]:
            # stored as expectant/delivery; invert to delivery/expectant
            N[f"mod_{o}_{k}_rrr"] = f2(1 / m[k]["rrr"]); N[f"mod_{o}_{k}_rrrci"] = f"{f2(1 / m[k]['rrr_hi'])}, {f2(1 / m[k]['rrr_lo'])}"
            N[f"mod_{o}_{k}_drd"] = f1(-m[k]["drd"]); N[f"mod_{o}_{k}_drdci"] = f"{f1(-m[k]['drd_hi'])}, {f1(-m[k]['drd_lo'])}"
        N[f"mod_{o}_prr"] = pv(m["p_homog_rr"]); N[f"mod_{o}_prd"] = pv(m["p_homog_rd"])
for k, v in R["w39"]["evalue_death"].items(): N[f"eval_{k}"] = f2(v["e_point"]); N[f"evallim_{k}"] = f2(v["e_ci_limit"])
for w in ("w38", "w40"): N[f"eval_{w}"] = f2(R[w]["evalue_death"]["all"]["e_point"]); N[f"evallim_{w}"] = f2(R[w]["evalue_death"]["all"]["e_ci_limit"])

SENS = {"exclude_gestational_hypertension": "Gestational hypertension or pre-eclampsia excluded",
        "low_indication": "No gestational hypertension or pre-eclampsia, no infertility treatment, maternal age <35 years",
        "late_induction_to_I": "Induced births in week 40 counted as delivery in week 39",
        "no_prelabor_cesarean_week_w": "Prelabour caesarean births in week 39 excluded", "cohort_2016_2024": "Births 2016 to 2024"}
for s in SENS:
    for o in ("death", "stillbirth", "macro", "nicu", "cesar", "smm"): put(f"sens_{s}_{o}", R["w39"]["sensitivity"][s][o]["all"])
    N[f"sens_{s}_n"] = c0(R["w39"]["sensitivity"][s]["n_at_risk"])
dd = [contrast(R["w39"]["sensitivity"][s]["death"]["all"])["d"] for s in SENS]
N["sens_death_min"] = f1(max(dd)); N["sens_death_max"] = f1(min(dd))
for x in ("0.25", "0.5"):
    q = R["w39"]["qba"][x]["all"]; N[f"qba_{x[2:]}"] = f1(-q["rd"]); N[f"qba_{x[2:]}_ci"] = f"{f1(-q['rd_hi'])}, {f1(-q['rd_lo'])}"
    N[f"qba_{x[2:]}_rr"] = f2(1 / q["rr"]); N[f"qba_{x[2:]}_rrci"] = f"{f2(1 / q['rr_hi'])}, {f2(1 / q['rr_lo'])}"


# Table 1 by observed course in week 39
def courses(x):
    i = x[(x.ga == 39) & (x.induct == 1) & (x.kind == "nat")]; o = x[x.ga >= 40]; return [i, x[(x.ga == 39)].drop(i.index), o, x]
G1 = courses(pop)
def line(lab, groups, f): return f"| {lab} | " + " | ".join(f"{c0(f(x))} ({pct(f(x), x.n.sum())})" for x in groups) + " |"
def rows(var, labels, title):
    out = [f"| {title} | | | | |"]
    for code, lab in labels: out.append(line("  " + lab, G1, lambda x: x[x[var].isin(code if isinstance(code, list) else [code])].n.sum()))
    return out
T1 = ["| Characteristic | Induction with birth in week 39 | Birth without induction or stillbirth in week 39^a^ | Continued to 40 0/7 weeks | All |", "|---|---|---|---|---|",
      "| Number | " + " | ".join(c0(x.n.sum()) for x in G1) + " |", "| Prepregnancy BMI, kg/m² | | | | |"]
for code, lab in [([0, 1], "<25.0"), ([2], "25.0 to 29.9"), ([3], "30.0 to 34.9"), ([4], "35.0 to 39.9"), ([5, 6], "≥40.0")]:
    T1.append(line("  " + lab, courses(raw), lambda x: x[x.bmic.isin(code)].n.sum()))
T1 += rows("age6", [(0, "<20"), (1, "20 to 24"), (2, "25 to 29"), (3, "30 to 34"), (4, "35 to 39"), (5, "≥40")], "Maternal age, years")
T1 += rows("nullip", [(1, "Parity zero"), (0, "Parity one or more"), (9, "Unknown")], "Parity")
T1 += rows("race", [(3, "American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, or more than one race"), (4, "Asian"), (2, "Black"), (7, "Hispanic"), (1, "White"), (8, "Unknown")], "Race and ethnicity^b^")
T1 += rows("educ", [(1, "High school or less"), (2, "Some college or associate degree"), (3, "Bachelor's degree or higher"), (9, "Unknown")], "Education")
T1 += rows("precare", [(1, "First trimester"), (2, "Second trimester"), (3, "Third trimester"), (4, "None"), (9, "Unknown")], "Prenatal care began")
T1 += [line("WIC participation", G1, lambda x: x[x.wic == 1].n.sum()), line("Smoking during pregnancy", G1, lambda x: x[x.smoke == 1].n.sum()),
       line("Infertility treatment", G1, lambda x: x[x.inftr == 1].n.sum()), line("Gestational hypertension or pre-eclampsia", G1, lambda x: x[x.ghyp == 1].n.sum()),
       "| End of pregnancy | | | | |", line("  Stillbirth", G1, lambda x: x[x.kind != "nat"].n.sum()),
       line("  Born in week 40", G1, lambda x: x[(x.ga == 40) & (x.kind == "nat")].n.sum()), line("  Born at 41 weeks or later", G1, lambda x: x[(x.ga >= 41) & (x.kind == "nat")].n.sum())]
N["table1"] = "\n".join(T1)
N["pct_ob30plus"] = pct(raw[raw.bmic >= 3].n.sum(), raw.n.sum()); N["pct_ghyp"] = pct(pop[pop.ghyp == 1].n.sum(), pop.n.sum())

ON = {"death": "Perinatal death", "stillbirth": "Stillbirth", "neonatal": "Neonatal death", "nicu": "NICU admission", "morb": "Composite neonatal morbidity^b^",
      "aven6": "Assisted ventilation >6 hours", "apg7": "5-minute Apgar score <7", "seiz": "Neonatal seizures", "macro": "Birthweight ≥4500 g",
      "cesar": "Caesarean delivery^c^", "plac": "Third- or fourth-degree perineal laceration", "smm": "Severe maternal morbidity^d^", "mtr": "Maternal transfusion",
      "uhyst": "Unplanned hysterectomy", "rupt": "Uterine rupture", "aicu": "Maternal ICU admission"}
def trow(lab, x):
    return f"| {lab} | {f1(x['a'])} | {f1(x['b'])} | {f1(x['d'])} ({f1(x['lo'])}, {f1(x['hi'])}) | {f2(x['rr'])} ({f2(x['rrlo'])}, {f2(x['rrhi'])}) | {(x['nnkind'] + ' ' + format(x['nn'], ',')) if x['nn'] else 'NA'} |"
T2 = ["| Outcome | Delivery in week 39 | Expectant management | Risk difference (95% CI) | Risk ratio (95% CI) | NND or NNH^a^ |", "|---|---|---|---|---|---|",
      "| Perinatal outcomes, per 10,000 pregnancies | | | | | |"]
for o in ("death", "stillbirth", "neonatal"): T2.append(trow(ON[o], X[("w39", o, "all")]))
T2.append("| Perinatal death by prepregnancy BMI, kg/m² | | | | | |")
for k in g.CLS: T2.append(trow("  " + TLAB[k], X[("w39", "death", k)]))
T2.append("| Neonatal outcomes, per 10,000 live births | | | | | |")
for o in ("nicu", "morb", "aven6", "apg7", "seiz", "macro"): T2.append(trow(ON[o], X[("w39", o, "all")]))
T2.append("| Maternal outcomes, per 10,000 live births | | | | | |")
for o in ("cesar", "plac", "smm", "mtr", "uhyst", "rupt", "aicu"): T2.append(trow(ON[o], X[("w39", o, "all")]))
N["table2"] = "\n".join(T2)

# Table 3: BMI modification on both scales (39-week trial)
T3 = ["| Outcome | BMI, kg/m² | Risk ratio (95% CI) | Ratio of risk ratios (95% CI) | Risk difference per 10,000 (95% CI) | Difference in risk differences (95% CI) |", "|---|---|---|---|---|---|"]
for o, lab in MODO.items():
    for k in g.CLS:
        x = X[("w39", o, k)]
        rrr = "1.00 (Reference)" if k == "lt25" else f"{N[f'mod_{o}_{k}_rrr']} ({N[f'mod_{o}_{k}_rrrci']})"
        drd = "0.00 (Reference)" if k == "lt25" else f"{N[f'mod_{o}_{k}_drd']} ({N[f'mod_{o}_{k}_drdci']})"
        T3.append(f"| {lab if k == 'lt25' else ''} | {TLAB[k]} | {f2(x['rr'])} ({f2(x['rrlo'])}, {f2(x['rrhi'])}) | {rrr} | {f1(x['d'])} ({f1(x['lo'])}, {f1(x['hi'])}) | {drd} |")
    T3.append(f"| | Homogeneity^a^ | | P {fp(N[f'mod_{o}_prr'])} | | P {fp(N[f'mod_{o}_prd'])} |")
N["table3"] = "\n".join(T3)


def etable(w):
    L = ["| Outcome | BMI, kg/m² | Delivery | Expectant | Risk difference (95% CI) | Risk ratio (95% CI) |", "|---|---|---|---|---|---|"]
    for o in OUTS:
        for k in ["all"] + g.CLS:
            x = X[(w, o, k)]
            L.append(f"| {ON[o].split('^')[0] if k == 'all' else ''} | {TLAB[k]} | {f1(x['a'])} | {f1(x['b'])} | {f1(x['d'])} ({f1(x['lo'])}, {f1(x['hi'])}) | {f2(x['rr'])} ({f2(x['rrlo'])}, {f2(x['rrhi'])}) |")
    return "\n".join(L)
for w in ("w38", "w39", "w40"): N[f"etable_{w}"] = etable(w)
L = ["| Analysis | Pregnancies | Perinatal death | Stillbirth | Birthweight ≥4500 g | NICU admission | Caesarean delivery | Severe maternal morbidity |", "|---|---|---|---|---|---|---|---|",
     f"| Primary analysis | {N['n_w39']} | " + " | ".join(f"{N[f'w39_{o}_all_d']} ({N[f'w39_{o}_all_ci']})" for o in ("death", "stillbirth", "macro", "nicu", "cesar", "smm")) + " |"]
for s, lab in SENS.items():
    L.append(f"| {lab} | {N[f'sens_{s}_n']} | " + " | ".join(f"{N[f'sens_{s}_{o}_d']} ({N[f'sens_{s}_{o}_ci']})" for o in ("death", "stillbirth", "macro", "nicu", "cesar", "smm")) + " |")
for x, lab in (("25", "25% of week-39 stillbirths assumed preventable by delivery"), ("5", "50% of week-39 stillbirths assumed preventable by delivery")):
    L.append(f"| {lab} | {N['n_w39']} | {N[f'qba_{x}']} ({N[f'qba_{x}_ci']}) | NA | NA | NA | NA | NA |")
N["etable_sens"] = "\n".join(L)
L = ["| Trial | BMI, kg/m² | Induced vs continued, unweighted | Delivery clone vs cohort, weighted | Expectant clone vs cohort, weighted | Weight truncation, delivery | Weight truncation, expectant |", "|---|---|---|---|---|---|---|"]
for w in ("w38", "w39", "w40"):
    for k, v in R[w]["balance_max_abs"].items():
        L.append(f"| {w[1:]} weeks | {TLAB[k]} | {v['raw_I_vs_O']:.3f} | {v['A_vs_population']:.3f} | {v['B_vs_population']:.3f} | {R[w]['models'][k]['trunc_A']:.2f} | {R[w]['models'][k]['trunc_B']:.2f} |")
N["etable_balance"] = "\n".join(L)
L = ["| BMI, kg/m² | Risk ratio (95% CI) | E-value, point estimate | E-value, confidence limit |", "|---|---|---|---|"]
for k in ["all"] + g.CLS:
    x = X[("w39", "death", k)]; L.append(f"| {TLAB[k]} | {f2(x['rr'])} ({f2(x['rrlo'])}, {f2(x['rrhi'])}) | {N[f'eval_{k}']} | {N[f'evallim_{k}']} |")
N["etable_evalue"] = "\n".join(L)

N["python_version"] = platform.python_version(); N["statsmodels_version"] = statsmodels.__version__
N["rust_version"] = subprocess.run([os.path.expanduser("~/.cargo/bin/rustc"), "--version"], capture_output=True, text=True).stdout.split()[1]
json.dump(N, open(os.path.join(OUT, "numbers_ppe.json"), "w"), indent=1, ensure_ascii=False)
print(len(N), "numbers")
