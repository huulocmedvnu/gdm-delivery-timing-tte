"""Figures for the manuscript, drawn with plotly and exported as PDF, PNG and LZW TIFF (at least 300 dpi at 17 cm).
Figure 1: study flow (A) and clone-censor-weight design (B).
Figure 2: delivery in week 39 compared with expectant management by BMI class. A, weighted risk of perinatal death under
each strategy. B to G, differences per 10,000 for stillbirth and secondary outcomes.
Figure 3: the trials at 38, 39 and 40 weeks, differences by week for all classes combined and the two extreme BMI classes.
Also writes outputs/balance.tsv. Input outputs/gdm_ccw.json and outputs/manuscript_numbers.json."""
import os, json
import pandas as pd, plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image
pio.kaleido.scope.mathjax = None   # otherwise kaleido can print a MathJax loading message into PDF exports

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, FIG = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(OUT, "gdm_ccw.json")))
N = json.load(open(os.path.join(OUT, "manuscript_numbers.json")))

# Palette (validated categorical slots 1 to 3 plus neutral inks). Delivery is blue throughout, expectant management grey.
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f3f2ee"
DELIV, EXPECT = "#2a78d6", "#8a8984"
G_ALL, G_LT25, G_OB35 = INK, "#1baf7a", "#eb6834"
FONT = "Arial"
CLS = ["all", "lt25", "over", "ob30", "ob35p"]
LAB = {"all": "<b>All classes</b>", "lt25": "Less than 25.0", "over": "25.0–29.9", "ob30": "30.0–34.9", "ob35p": "35.0 or higher"}


def save(f, name, w, h):
    f.update_layout(width=w, height=h, paper_bgcolor="white", plot_bgcolor="white", font=dict(family=FONT, size=17, color=INK))
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
num = lambda v: f"{v:.1f}".replace("-", "−")


def axis_style(f, **kw):
    f.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False, ticks="", tickfont=dict(size=15, color=INK2), **kw)
    f.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")


# ---------------------------------------------------------------- Figure 2
# Drawn at 1,100 px logical width so that text stays at 8 pt or more when printed at 6.5 inches,
# also after Editorial Manager downsamples the image to about 1,170 px.
t = T[T.trial == "w39"]
P2 = [("stillbirth", "Stillbirth", "per 10,000 pregnancies"), ("nicu", "NICU admission", "per 10,000 live births"),
      ("morb", "Neonatal morbidity", "per 10,000 live births"), ("macro", "Birth weight ≥4,500 g", "per 10,000 live births"),
      ("smm", "Severe maternal morbidity", "per 10,000 live births"), ("mtr", "Maternal transfusion", "per 10,000 live births")]
f = make_subplots(rows=3, cols=3, specs=[[{"colspan": 3}, None, None], [{}, {}, {}], [{}, {}, {}]],
                  row_heights=[0.36, 0.32, 0.32], horizontal_spacing=0.035, vertical_spacing=0.115)
y = list(range(len(CLS))); YR = [len(CLS) - 0.45, -0.55]
f.layout.xaxis.domain = [0, 0.56]
s = t[t.outcome == "death"].set_index("bmi").loc[CLS]
f.add_shape(type="rect", xref="paper", yref="y", x0=-0.19, x1=1.0, y0=-0.45, y1=0.45, fillcolor=BAND, line_width=0, layer="below")
for i, k in enumerate(CLS):
    f.add_trace(go.Scatter(x=[s.loc[k, "risk_delivery"], s.loc[k, "risk_expectant"]], y=[i, i], mode="lines", line=dict(color="#c9c8c2", width=6),
                           showlegend=False, hoverinfo="skip"), row=1, col=1)
f.add_trace(go.Scatter(x=s.risk_expectant, y=y, mode="markers", name="Expectant management", marker=dict(color="white", size=19, line=dict(color=EXPECT, width=3))), row=1, col=1)
f.add_trace(go.Scatter(x=s.risk_delivery, y=y, mode="markers", name="Delivery in week 39", marker=dict(color=DELIV, size=19, line=dict(color="white", width=2))), row=1, col=1)
for i, k in enumerate(CLS):
    f.add_annotation(x=s.loc[k, "risk_expectant"], y=i, text=num(s.loc[k, "risk_expectant"]), xshift=14, xanchor="left", showarrow=False, font=dict(size=18, color=INK2), row=1, col=1)
    f.add_annotation(x=s.loc[k, "risk_delivery"], y=i, text=num(s.loc[k, "risk_delivery"]), xshift=-14, xanchor="right", showarrow=False, font=dict(size=18, color=DELIV), row=1, col=1)
    f.add_annotation(xref="paper", yref="y", x=0.865, y=i, text=f"{num(s.loc[k, 'diff'])} ({num(s.loc[k, 'lo'])} to {num(s.loc[k, 'hi'])})", xanchor="right", showarrow=False, font=dict(size=18))
    f.add_annotation(xref="paper", yref="y", x=0.995, y=i, text=f"{int(s.loc[k, 'nnd']):,}", xanchor="right", showarrow=False, font=dict(size=18))
f.add_annotation(xref="paper", yref="y domain", x=0.865, y=1.0, yanchor="bottom", yshift=4, text="<b>Difference per 10,000<br>(95% CrI)</b>", xanchor="right", showarrow=False, font=dict(size=17, color=INK2))
f.add_annotation(xref="paper", yref="y domain", x=0.995, y=1.0, yanchor="bottom", yshift=4, text="<b>NND</b>", xanchor="right", showarrow=False, font=dict(size=17, color=INK2))
f.update_xaxes(range=[0, 30], dtick=10, title_text="Perinatal death per 10,000 pregnancies", row=1, col=1)
f.update_yaxes(tickvals=y, ticktext=[LAB[k] for k in CLS], range=YR, tickfont=dict(size=18), row=1, col=1)
for j, (o, title, unit) in enumerate(P2):
    r, c = j // 3 + 2, j % 3 + 1; n = j + 2
    s = t[t.outcome == o].set_index("bmi").loc[CLS]
    excl = (s.hi < 0) | (s.lo > 0)
    f.add_shape(type="rect", xref=f"x{n} domain", yref=f"y{n}", x0=0, x1=1, y0=-0.45, y1=0.45, fillcolor=BAND, line_width=0, layer="below")
    f.add_shape(type="line", xref=f"x{n}", yref=f"y{n} domain", x0=0, x1=0, y0=0, y1=1, line=dict(color=INK2, width=1.5))
    f.add_trace(go.Scatter(x=s["diff"], y=y, mode="markers", showlegend=False,
                           marker=dict(color=[INK if e else "white" for e in excl], size=[19] + [14] * 4, symbol=["diamond"] + ["circle"] * 4,
                                       line=dict(color=INK, width=2.2)),
                           error_x=dict(type="data", symmetric=False, array=s.hi - s["diff"], arrayminus=s["diff"] - s.lo, color=INK, thickness=2.4, width=0)),
                  row=r, col=c)
    lo_, hi_ = min(s.lo.min(), 0), max(s.hi.max(), 0); pad = 0.1 * (hi_ - lo_)
    f.update_xaxes(range=[lo_ - pad, hi_ + pad], title_text=unit, title_font=dict(size=17, color=INK2), nticks=4, row=r, col=c)
    f.update_yaxes(tickvals=y, ticktext=[LAB[k] for k in CLS], showticklabels=(c == 1), range=YR, tickfont=dict(size=18), row=r, col=c)
    f.add_annotation(xref=f"x{n} domain", yref=f"y{n} domain", x=0, y=1.02, yanchor="bottom", xanchor="left", text=f"<b>{'BCDEFG'[j]}</b>  {title}",
                     showarrow=False, font=dict(size=20))
axis_style(f)
f.update_xaxes(tickfont=dict(size=18, color=INK2))
f.update_xaxes(title_font=dict(size=18, color=INK2), row=1, col=1)
f.add_annotation(xref="x domain", yref="y domain", x=0, y=1.0, yanchor="bottom", xanchor="left", yshift=44, text="<b>A</b>  Perinatal death with each strategy",
                 showarrow=False, font=dict(size=21))
f.add_annotation(xref="paper", yref="paper", x=-0.19, y=-0.088, xanchor="left", showarrow=False, font=dict(size=17, color=INK2), align="left",
                 text="B–G: difference, delivery in week 39 minus expectant management, with 95% credible interval.<br>"
                      "Left of the line, fewer events with delivery. Filled marks: interval excludes zero.")
f.update_layout(margin=dict(l=215, r=20, t=120, b=150), legend=dict(orientation="h", xref="paper", x=0.0, xanchor="left", y=1.0, yanchor="bottom", font=dict(size=18),
                bgcolor="rgba(0,0,0,0)"))
f.layout.legend.y = f.layout.yaxis.domain[1] + 0.006; f.layout.legend.x = 0.0
f.add_annotation(xref="paper", yref="y domain", x=-0.19, y=1.0, yanchor="bottom", yshift=4, xanchor="left", showarrow=False, text="<b>BMI, kg/m²</b>", font=dict(size=17, color=INK2))
for ya in ("y2 domain", "y5 domain"):
    f.add_annotation(xref="paper", yref=ya, x=-0.19, y=1.0, yanchor="bottom", yshift=4, xanchor="left", showarrow=False, text="<b>BMI, kg/m²</b>", font=dict(size=17, color=INK2))
save(f, "Figure_2", 1100, 1350)

# ---------------------------------------------------------------- Figure 3 (same 1,100 px logical width, 4 rows by 2 columns)
P3 = [("death", "Perinatal death"), ("neonatal", "Neonatal death"), ("nicu", "NICU admission"), ("morb", "Composite neonatal morbidity"),
      ("aven6", "Ventilation longer than 6 hours"), ("macro", "Birth weight 4,500 g or more"), ("smm", "Severe maternal morbidity"),
      ("plac", "Third- or fourth-degree laceration")]
GRP = [("lt25", "BMI less than 25.0", G_LT25, "circle", -0.2), ("all", "All classes", G_ALL, "diamond", 0.0), ("ob35p", "BMI 35.0 or higher", G_OB35, "circle", 0.2)]
WK = [("w38", 38), ("w39", 39), ("w40", 40)]
f = make_subplots(rows=4, cols=2, horizontal_spacing=0.11, vertical_spacing=0.075)
for i, (o, title) in enumerate(P3):
    r, c = i // 2 + 1, i % 2 + 1
    f.add_hline(y=0, line=dict(color=INK2, width=1.5), row=r, col=c)
    for k, lab, col, sym, off in GRP:
        s = T[(T.outcome == o) & (T.bmi == k)].set_index("trial").loc[[w for w, _ in WK]]
        excl = (s.hi < 0) | (s.lo > 0); xs = [g + off for _, g in WK]
        f.add_trace(go.Scatter(x=xs, y=s["diff"], mode="lines", line=dict(color=col, width=1.8, dash="dot"), showlegend=False, hoverinfo="skip",
                               opacity=0.7), row=r, col=c)
        f.add_trace(go.Scatter(x=xs, y=s["diff"], mode="markers", name=lab, legendgroup=lab, showlegend=(i == 0),
                               marker=dict(color=[col if e else "white" for e in excl], size=17 if k == "all" else 14, symbol=sym, line=dict(color=col, width=2.4)),
                               error_y=dict(type="data", symmetric=False, array=s.hi - s["diff"], arrayminus=s["diff"] - s.lo, color=col, thickness=2.4, width=0)),
                      row=r, col=c)
    f.add_annotation(xref="x domain" if i == 0 else f"x{i + 1} domain", yref="y domain" if i == 0 else f"y{i + 1} domain", x=0, y=1.03,
                     text=f"<b>{'ABCDEFGH'[i]}</b>  {title}", xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=20))
f.update_xaxes(tickvals=[38, 39, 40], ticktext=["Week 38", "Week 39", "Week 40"], range=[37.5, 40.5], showgrid=False, zeroline=False, showline=True,
               linecolor=GRID, ticks="", tickfont=dict(size=18, color=INK2))
f.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, ticks="", tickfont=dict(size=18, color=INK2), nticks=5,
               title_text="per 10,000", title_font=dict(size=17, color=INK2))
f.add_annotation(xref="paper", yref="paper", x=0.5, y=-0.068, showarrow=False, font=dict(size=17, color=INK2), align="center",
                 text="Week of time zero: delivery within that week compared with expectant management. Below the line, fewer events with delivery.<br>"
                      "Differences per 10,000 pregnancies (A, B) or live births (C–H), 95% credible intervals. Filled marks: interval excludes zero.")
f.update_layout(margin=dict(l=95, r=25, t=95, b=135), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.035, yanchor="bottom", font=dict(size=19)))
save(f, "Figure_3", 1100, 1350)

# ---------------------------------------------------------------- Figure 1
f = go.Figure(); f.update_xaxes(visible=False, range=[0, 1]); f.update_yaxes(visible=False, range=[0, 1])
def box(x0, y0, x1, y1, text, fill="white", size=16, line=INK2, bold=False, color=INK):
    f.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=line, width=1.4), fillcolor=fill)
    f.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"<b>{text}</b>" if bold else text, showarrow=False, font=dict(size=size, color=color), align="center")
def arrow(x0, y0, x1, y1):
    f.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.5, arrowcolor=INK2, text="")
box(0.04, 0.93, 0.46, 0.995, "Live births and fetal deaths at 20 weeks or later,<br>US residents, 2019–2024<br><b>n = " + N["flow_all"] + "</b>")
exc = [(0.893, "Multiple gestation: " + N["flow_ex_multiple"]), (0.851, "Gestation below 39 weeks or unknown: " + N["flow_ex_ga"]),
       (0.809, "Prepregnancy BMI unknown: " + N["flow_ex_bmi"]), (0.767, "No gestational diabetes: " + N["flow_ex_nogdm"]),
       (0.725, "Prepregnancy diabetes or chronic hypertension: " + N["flow_ex_pdm_cht"]),
       (0.677, "Previous cesarean, or noncephalic or unknown<br>presentation: " + N["flow_ex_prevcs_pres"]), (0.629, "Congenital anomaly: " + N["flow_ex_anom"])]
for yy, tx in exc:
    h = 0.044 if "<br>" in tx else 0.034
    box(0.36, yy - h / 2, 0.99, yy + h / 2, tx, fill=BAND, size=15, line="#c9c8c2"); arrow(0.22, yy, 0.36, yy)
f.add_shape(type="line", x0=0.22, y0=0.93, x1=0.22, y1=0.598, line=dict(color=INK2, width=1.5)); arrow(0.22, 0.605, 0.22, 0.592)
box(0.04, 0.508, 0.46, 0.592, "<b>Singleton pregnancies with gestational diabetes<br>ongoing at 39 0/7 weeks<br>n = " + N["n_cohort"] + "</b><br>(" + N["live_w39"] + " live births, " + N["sb_w39"] + " stillbirths)",
    fill="#eaf2fc", line=DELIV)
box(0.53, 0.508, 0.99, 0.592, "Same criteria at the other time zeros<br>Ongoing at 38 0/7 weeks: n = " + N["n_w38"] + "<br>Ongoing at 40 0/7 weeks: n = " + N["n_w40"], size=15)
f.add_annotation(x=0.5, y=0.462, text="<b>Observed course in week 39 and its use in each emulated strategy</b>", showarrow=False, font=dict(size=17))
for x, tx in [(0.21, "Observed course"), (0.575, "Delivery in week 39"), (0.862, "Expectant management")]:
    f.add_annotation(x=x, y=0.418, text=f"<b>{tx}</b>", showarrow=False, font=dict(size=16, color=DELIV if "Delivery" in tx else INK2 if "Expectant" in tx else INK))
f.add_shape(type="line", x0=0.43, x1=0.72, y0=0.398, y1=0.398, line=dict(color=DELIV, width=3))
f.add_shape(type="line", x0=0.735, x1=0.99, y0=0.398, y1=0.398, line=dict(color=EXPECT, width=3))
cy = [0.335, 0.215, 0.095]; CC = [DELIV, G_LT25, EXPECT]
crs = ["<b>I</b>  Labor induction with birth<br>in week 39<br>n = " + N["n_I_w39"], "<b>S</b>  Birth without induction<br>or stillbirth in week 39<br>n = " + N["n_S_w39"],
       "<b>O</b>  Continued to 40 0/7 weeks<br>n = " + N["n_O_w39"]]
for tx, cc, yy in zip(crs, CC, cy):
    box(0.02, yy - 0.05, 0.40, yy + 0.05, tx, size=15); f.add_shape(type="rect", x0=0.02, y0=yy - 0.05, x1=0.034, y1=yy + 0.05, fillcolor=cc, line=dict(width=0))
cells = [("✓ Counted<br>weight 1 + P(O)/P(I)", "✕ Censored at induction"),
         ("✓ Counted, weight 1<br>includes all week-39 stillbirths", "✓ Counted, weight 1/[1 − P(I)]<br>includes all week-39 stillbirths"),
         ("✕ Censored at 40 0/7 weeks", "✓ Counted, weight 1/[1 − P(I)]")]
for (a_, b_), yy in zip(cells, cy):
    for (x0, x1, tx, col) in [(0.43, 0.72, a_, DELIV), (0.735, 0.99, b_, INK2)]:
        on = tx.startswith("✓")
        box(x0, yy - 0.05, x1, yy + 0.05, tx, fill="#eaf2fc" if (on and col == DELIV) else "#efeee9" if on else "white", size=15,
            line=col if on else "#c9c8c2", color=INK if on else MUTED)
f.add_annotation(x=0.5, y=0.0, text="P(I) and P(O): probabilities of courses I and O given covariates, estimated within each BMI class.<br>The trials at 38 and 40 weeks follow the same scheme.",
                 showarrow=False, font=dict(size=15, color=INK2), yanchor="bottom")
f.add_annotation(x=0.0, y=1.0, text="<b>A</b>", showarrow=False, font=dict(size=24), xanchor="left", yanchor="top")
f.add_annotation(x=0.0, y=0.485, text="<b>B</b>", showarrow=False, font=dict(size=24), xanchor="left", yanchor="top")
f.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
save(f, "Figure_1", 1000, 1200)
print(T[(T.bmi == "all") & (T.trial == "w39")][["outcome", "diff", "lo", "hi", "nnd", "nnh"]].to_string(index=False))
