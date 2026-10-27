"""eFigure 1: directed acyclic graph for the 39-week trial (PPE). Measured covariates adjusted for, and unmeasured factors
behind confounding by indication."""
import os, plotly.graph_objects as go, plotly.io as pio
pio.kaleido.scope.mathjax = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FIG = os.path.join(ROOT, "figures")
INK, INK2, DELIV, MUT = "#0b0b0b", "#52514e", "#2a78d6", "#8a8984"
f = go.Figure(); f.update_xaxes(visible=False, range=[0, 1]); f.update_yaxes(visible=False, range=[0, 1])
def box(x0, y0, x1, y1, text, line=INK2, fill="white", dash="solid", color=INK):
    f.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=line, width=1.6, dash=dash), fillcolor=fill)
    f.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text, showarrow=False, font=dict(size=16, color=color), align="center")
def arrow(x0, y0, x1, y1, color=INK2, dash=False):
    f.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.2,
                     arrowwidth=1.8, arrowcolor=color, text="")
box(0.02, 0.70, 0.98, 0.97, "<b>Measured baseline covariates (adjusted)</b><br>Prepregnancy BMI class and BMI, maternal age, parity, race and ethnicity,<br>"
    "education, start of prenatal care, WIC participation, smoking, height,<br>infertility treatment, gestational hypertension or pre-eclampsia, year")
box(0.06, 0.36, 0.40, 0.52, "<b>Delivery in week 39</b><br>(versus expectant management)", line=DELIV, fill="#eaf2fc")
box(0.60, 0.36, 0.94, 0.52, "<b>Perinatal death</b><br>(stillbirth or neonatal death)")
box(0.02, 0.03, 0.98, 0.24, "<b>Unmeasured (not recorded on certificates)</b><br>Treatment with nutrition or medication, glycaemic control,<br>"
    "estimated fetal weight, cervical status, indication for induction", line=MUT, dash="dash", color=INK2)
arrow(0.40, 0.44, 0.60, 0.44, color=DELIV)
arrow(0.23, 0.70, 0.23, 0.52); arrow(0.77, 0.70, 0.77, 0.52)
arrow(0.23, 0.24, 0.23, 0.36, color=MUT); arrow(0.77, 0.24, 0.77, 0.36, color=MUT)
f.update_layout(width=1000, height=700, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Arial", size=16, color=INK))
f.write_image(os.path.join(FIG, "eFigure_1.png"), scale=3); f.write_image(os.path.join(FIG, "eFigure_1.pdf"))
print("ok")
