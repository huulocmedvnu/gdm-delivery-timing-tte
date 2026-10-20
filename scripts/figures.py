"""Figures. Fig 3: the three trials (38, 39, 40 weeks) for key benefits and harms. Fig 2: benefit and harm of delivery in week 39 versus expectant management, absolute risk differences per
10,000 by BMI class, with numbers needed to deliver (NND) and to harm (NNH). Also writes outputs/balance.tsv (trials at 38, 39 and 40 weeks).
Input outputs/gdm_ccw.json. Large type so panels stay legible when the journal system downsamples figures."""
import os, json
import numpy as np, pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(OUT, "gdm_ccw.json")))
CLS = ["all", "lt25", "over", "ob30", "ob35p"]
LAB = {"all": "<b>All</b>", "lt25": "<25.0", "over": "25.0–29.9", "ob30": "30.0–34.9", "ob35p": "≥35.0"}
NEO, MAT = "#1f4e79", "#c0504d"
PANELS = [("death", "Perinatal death", NEO), ("stillbirth", "Stillbirth", NEO), ("nicu", "NICU admission", NEO),
          ("morb", "Neonatal morbidity", NEO), ("macro", "Birthweight ≥4,500 g", NEO),
          ("smm", "Severe maternal morbidity", MAT), ("mtr", "Maternal transfusion", MAT), ("aicu", "Maternal ICU admission", MAT)]


def save(f, name, w, h, size=18):
    f.update_layout(template="plotly_white", width=w, height=h, font=dict(size=size, family="Arial"))
    f.write_html(os.path.join(FIG, f"{name}.html"), include_plotlyjs="cdn"); f.write_image(os.path.join(FIG, f"{name}.pdf"))
    png = os.path.join(FIG, f"{name}.png"); f.write_image(png, scale=4.5)
    im = Image.open(png).convert("RGB"); dpi = max(300, round(im.width / (17 / 2.54)))
    im.save(os.path.join(FIG, f"{name}.tif"), compression="tiff_lzw", dpi=(dpi, dpi))


def table(w):
    rows = []
    for o, v in R[w]["primary"].items():
        if o == "sbS": continue
        for k in CLS:
            x = v[k]; d, lo, hi = -x["rd"], -x["rd_hi"], -x["rd_lo"]          # delivery minus expectant
            n = round(1e4 / abs(d)) if abs(d) > 0 and (hi < 0 or lo > 0) else None
            rows.append(dict(trial=w, outcome=o, bmi=k, risk_delivery=x["risk_A"], risk_expectant=x["risk_B"], diff=round(d, 2),
                             lo=round(lo, 2), hi=round(hi, 2), p_fewer_with_delivery=x["p_B_higher"],
                             nnd=n if (n and d < 0) else None, nnh=n if (n and d > 0) else None))
    return pd.DataFrame(rows)


T = pd.concat([table(w) for w in ("w39", "w38", "w40")]); T.to_csv(os.path.join(OUT, "balance.tsv"), sep="\t", index=False)
t = T[T.trial == "w39"]
f = make_subplots(rows=2, cols=4, subplot_titles=[p[1] for p in PANELS], horizontal_spacing=0.06, vertical_spacing=0.2, shared_yaxes=True)
for i, (o, title, col) in enumerate(PANELS):
    r, c = i // 4 + 1, i % 4 + 1
    s = t[t.outcome == o].set_index("bmi").loc[CLS]; y = [LAB[k] for k in CLS]
    f.add_trace(go.Scatter(x=s["diff"], y=y, mode="markers", marker=dict(color=col, size=[16] + [12] * 4, symbol=["diamond"] + ["circle"] * 4),
                           error_x=dict(type="data", symmetric=False, array=s.hi - s["diff"], arrayminus=s["diff"] - s.lo, color=col, thickness=2.5, width=6),
                           showlegend=False), row=r, col=c)
    f.add_vline(x=0, line=dict(color="#555", width=1.5, dash="dot"), row=r, col=c)
    if o in ("death", "smm"):
        for k, yy in zip(CLS, y):
            n = s.loc[k, "nnd"] if o == "death" else s.loc[k, "nnh"]
            if pd.notna(n): f.add_annotation(x=s.loc[k, "hi"], y=yy, text=("NND " if o == "death" else "NNH ") + f"{int(n):,}", xanchor="left", xshift=8,
                                              showarrow=False, font=dict(size=15, color=col), row=r, col=c)
f.update_yaxes(autorange="reversed", tickfont=dict(size=17))
f.update_xaxes(zeroline=False, title_text="per 10,000", title_font=dict(size=15), tickfont=dict(size=15))
f.add_annotation(x=0.5, y=-0.13, xref="paper", yref="paper", showarrow=False, font=dict(size=16),
                 text="Difference, delivery in week 39 minus expectant management (95% credible interval). Left of zero: fewer events with delivery")
f.add_annotation(x=-0.075, y=0.5, xref="paper", yref="paper", textangle=-90, showarrow=False, text="Prepregnancy BMI, kg/m²", font=dict(size=17))
for a in f.layout.annotations:
    if a.text in [p[1] for p in PANELS]: a.font = dict(size=19)
f.update_layout(margin=dict(l=150, r=40, t=60, b=120))
save(f, "Figure_2", 1600, 900)
print(t[t.outcome.isin(["death", "smm"])][["outcome", "bmi", "diff", "lo", "hi", "nnd", "nnh"]].to_string(index=False))

# Figure 3: the three trials side by side (delivery within week 38, 39 or 40 versus expectant management), all classes
# combined and the two extreme BMI classes, to show where the balance turns
P3 = [("death", "Perinatal death"), ("neonatal", "Neonatal death"), ("nicu", "NICU admission"), ("morb", "Neonatal morbidity"),
      ("aven6", "Ventilation >6 hours"), ("macro", "Birthweight ≥4,500 g"), ("smm", "Severe maternal morbidity"), ("plac", "3rd or 4th degree laceration")]
GRP = [("all", "All", "#222222", "diamond", 15, 0.0), ("lt25", "BMI <25.0", "#5b9bd5", "circle", 11, -0.22), ("ob35p", "BMI ≥35.0", "#c0504d", "circle", 11, 0.22)]
WK = {"w38": 0, "w39": 1, "w40": 2}
f = make_subplots(rows=2, cols=4, subplot_titles=[p[1] for p in P3], horizontal_spacing=0.06, vertical_spacing=0.2, shared_yaxes=True)
for i, (o, title) in enumerate(P3):
    r, c = i // 4 + 1, i % 4 + 1
    for k, lab, col, sym, sz, off in GRP:
        s = T[(T.outcome == o) & (T.bmi == k)].set_index("trial").loc[list(WK)]
        f.add_trace(go.Scatter(x=s["diff"], y=[WK[w] + off for w in WK], mode="markers", name=lab, legendgroup=lab, showlegend=(i == 0),
                               marker=dict(color=col, size=sz, symbol=sym),
                               error_x=dict(type="data", symmetric=False, array=s.hi - s["diff"], arrayminus=s["diff"] - s.lo, color=col, thickness=2.2, width=4)),
                      row=r, col=c)
    f.add_vline(x=0, line=dict(color="#555", width=1.5, dash="dot"), row=r, col=c)
    if c > 1: pass
f.update_yaxes(tickvals=[0, 1, 2], ticktext=["Week 38", "Week 39", "Week 40"], autorange="reversed", range=[2.5, -0.5], tickfont=dict(size=17))
f.update_xaxes(zeroline=False, title_text="per 10,000", title_font=dict(size=15), tickfont=dict(size=15))
for a in f.layout.annotations:
    if a.text in [p[1] for p in P3]: a.font = dict(size=19)
f.add_annotation(x=0.5, y=-0.13, xref="paper", yref="paper", showarrow=False, font=dict(size=16),
                 text="Difference, delivery within the week minus expectant management (95% credible interval). Left of zero: fewer events with delivery")
f.add_annotation(x=-0.07, y=0.5, xref="paper", yref="paper", textangle=-90, showarrow=False, text="Trial (time zero)", font=dict(size=17))
f.update_layout(margin=dict(l=150, r=40, t=100, b=120), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12, font=dict(size=17)))
save(f, "Figure_3", 1600, 920)
print(T[(T.bmi == "all") & T.outcome.isin(["death", "neonatal", "nicu", "morb", "aven6", "smm", "plac"])][["trial", "outcome", "diff", "lo", "hi", "nnd", "nnh"]].to_string(index=False))

# Figure 1: flow (A) and clone-censor-weight design (B), large type so it stays legible when downsampled
N = json.load(open(os.path.join(OUT, "manuscript_numbers.json")))
f = go.Figure(); f.update_xaxes(visible=False, range=[0, 1]); f.update_yaxes(visible=False, range=[0, 1])
def box(x0, y0, x1, y1, text, fill="white", size=17):
    f.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="#444", width=1.5), fillcolor=fill)
    f.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text, showarrow=False, font=dict(size=size), align="center")
def arrow(x0, y0, x1, y1):
    f.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1.6, arrowcolor="#444", text="")
box(0.05, 0.93, 0.47, 0.995, "Live births and fetal deaths at 20 weeks or later,<br>US residents, 2019–2024<br>n = " + N["flow_all"])
exc = [(0.893, "Multiple gestation: " + N["flow_ex_multiple"]), (0.851, "Gestation below 39 weeks or unknown: " + N["flow_ex_ga"]),
       (0.809, "Prepregnancy BMI unknown: " + N["flow_ex_bmi"]), (0.767, "No gestational diabetes: " + N["flow_ex_nogdm"]),
       (0.725, "Prepregnancy diabetes or chronic hypertension: " + N["flow_ex_pdm_cht"]),
       (0.677, "Previous cesarean, or noncephalic or unknown<br>presentation: " + N["flow_ex_prevcs_pres"]), (0.629, "Congenital anomaly: " + N["flow_ex_anom"])]
for y, t in exc:
    h = 0.044 if "<br>" in t else 0.034
    box(0.36, y - h / 2, 0.99, y + h / 2, t, fill="#f4f4f4", size=16); arrow(0.23, y, 0.36, y)
f.add_shape(type="line", x0=0.23, y0=0.93, x1=0.23, y1=0.598, line=dict(color="#444", width=1.6)); arrow(0.23, 0.605, 0.23, 0.592)
box(0.05, 0.512, 0.47, 0.592, "Singleton pregnancies with gestational diabetes<br>ongoing at 39 0/7 weeks<br>n = " + N["n_cohort"] + "<br>(" + N["live_w39"] + " live births, " + N["sb_w39"] + " stillbirths)", fill="#eef1ff")
box(0.53, 0.512, 0.99, 0.592, "Same criteria at other time zeros:<br>ongoing at 38 0/7 weeks, n = " + N["n_w38"] + "<br>ongoing at 40 0/7 weeks, n = " + N["n_w40"], fill="white", size=16)
f.add_annotation(x=0.5, y=0.472, text="<b>Courses observed in week 39 and their use in each emulated strategy</b>", showarrow=False, font=dict(size=17))
f.add_annotation(x=0.21, y=0.428, text="<b>Observed course</b>", showarrow=False, font=dict(size=17))
f.add_annotation(x=0.575, y=0.428, text="<b>Delivery in week 39</b>", showarrow=False, font=dict(size=17))
f.add_annotation(x=0.86, y=0.428, text="<b>Expectant management</b>", showarrow=False, font=dict(size=17))
cy = [0.338, 0.208, 0.078]; CC = ["#636efa", "#00cc96", "#ef553b"]
crs = ["I: induced and born in week 39<br>n = " + N["n_I_w39"], "S: stillbirth, spontaneous or<br>noninduced birth in week 39<br>n = " + N["n_S_w39"], "O: ongoing at 40 0/7 weeks<br>n = " + N["n_O_w39"]]
for t, cc, y in zip(crs, CC, cy):
    box(0.02, y - 0.055, 0.40, y + 0.055, t, size=16); f.add_shape(type="rect", x0=0.02, y0=y - 0.055, x1=0.035, y1=y + 0.055, fillcolor=cc, line=dict(width=0))
cells = [("Counts, weight<br>1 + P(O)/P(I)", "Censored at induction"), ("Counts, weight 1<br>(includes all week-39<br>stillbirths)", "Counts, weight 1/[1 − P(I)]<br>(includes all week-39<br>stillbirths)"),
         ("Censored at 40 0/7 weeks", "Counts, weight 1/[1 − P(I)]")]
for (a_, b_), y in zip(cells, cy):
    box(0.43, y - 0.055, 0.72, y + 0.055, a_, fill="#eef1ff" if "Counts" in a_ else "#f4f4f4", size=16)
    box(0.735, y - 0.055, 0.99, y + 0.055, b_, fill="#eef1ff" if "Counts" in b_ else "#f4f4f4", size=16)
f.add_annotation(x=0.5, y=0.0, text="P(I) and P(O): probabilities of courses I and O given covariates, estimated within each BMI class. Trials at 38 and 40 weeks follow the same scheme.",
                 showarrow=False, font=dict(size=14, color="#333"), yanchor="bottom")
f.add_annotation(x=0.0, y=1.0, text="<b>A</b>", showarrow=False, font=dict(size=24), xanchor="left", yanchor="top")
f.add_annotation(x=0.0, y=0.495, text="<b>B</b>", showarrow=False, font=dict(size=24), xanchor="left", yanchor="top")
f.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
save(f, "Figure_1", 1000, 1200, size=17)
