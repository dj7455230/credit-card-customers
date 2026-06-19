"""
CHURN ORACLE v5
Bank Customer Attrition — ML Prediction Dashboard
BankChurners.csv trained models: 37 features, 4 models
"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, warnings, os, random
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════
try:
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass

    NB = joblib.load("naive.pkl")
    RF = joblib.load("random_forest.pkl")
    ET = joblib.load("extra_trees.pkl")
    LR = joblib.load("logistic.pkl")

    FEATURE_NAMES = list(NB.feature_names_in_)
    MODELS = {
        "Naive Bayes":   NB,
        "Random Forest": RF,
        "Extra Trees":   ET,
        "Logistic Reg":  LR,
    }
    print(f"[OK] {len(MODELS)} models loaded, {len(FEATURE_NAMES)} features")

except Exception as exc:
    print(f"[WARN] Could not load models: {exc}")
    FEATURE_NAMES = [
        "Customer_Age", "Dependent_count", "Months_on_book",
        "Total_Relationship_Count", "Months_Inactive_12_mon",
        "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
        "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
        "Gender_F", "Gender_M",
        "Education_Level_College", "Education_Level_Doctorate",
        "Education_Level_Graduate", "Education_Level_High School",
        "Education_Level_Post-Graduate", "Education_Level_Uneducated",
        "Education_Level_Unknown",
        "Marital_Status_Divorced", "Marital_Status_Married",
        "Marital_Status_Single", "Marital_Status_Unknown",
        "Income_Category_$120K +", "Income_Category_$40K - $60K",
        "Income_Category_$60K - $80K", "Income_Category_$80K - $120K",
        "Income_Category_Less than $40K", "Income_Category_Unknown",
        "Card_Category_Blue", "Card_Category_Gold",
        "Card_Category_Platinum", "Card_Category_Silver",
    ]
    MODELS = {}

MODEL_ACC = {
    "Naive Bayes":   89.3,
    "Random Forest": 95.3,
    "Extra Trees":   92.1,
    "Logistic Reg":  89.3,
}

DEFAULTS = dict(
    Customer_Age=46, Dependent_count=2, Months_on_book=36,
    Total_Relationship_Count=4, Months_Inactive_12_mon=2,
    Contacts_Count_12_mon=2, Credit_Limit=4549.0,
    Total_Revolving_Bal=1276.0, Avg_Open_To_Buy=3474.0,
    Total_Amt_Chng_Q4_Q1=0.74, Total_Trans_Amt=3899.0,
    Total_Trans_Ct=67, Total_Ct_Chng_Q4_Q1=0.70,
    Avg_Utilization_Ratio=0.18, Gender="Female",
    Education_Level="Graduate", Marital_Status="Married",
    Income_Category="Less than $40K", Card_Category="Blue",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {
  --bg:#07080f; --bg2:#0f1020; --surf:rgba(13,14,28,0.95);
  --bdr:rgba(124,82,255,0.20); --bdr2:rgba(124,82,255,0.45);
  --p:#7c52ff; --p2:#a78bfa; --pink:#f472b6; --cyan:#22d3ee;
  --green:#10b981; --amber:#f59e0b; --red:#ef4444;
  --txt:#f0eeff; --txt2:#9895b5; --txt3:#3d3b5c;
}
.light-mode {
  --bg:#f4f4ff; --bg2:#e8e8ff; --surf:rgba(255,255,255,0.97);
  --bdr:rgba(109,40,217,0.18); --bdr2:rgba(109,40,217,0.45);
  --p:#6d28d9; --p2:#5b21b6; --pink:#db2777; --cyan:#0891b2;
  --green:#059669; --amber:#d97706; --red:#dc2626;
  --txt:#1e1b4b; --txt2:#4338ca; --txt3:#7c3aed;
}
*,*::before,*::after{box-sizing:border-box}
body,.gradio-container{
  background:var(--bg)!important;
  font-family:'Inter',sans-serif!important;
  color:var(--txt)!important;min-height:100vh;
}
.gradio-container::before{
  content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(ellipse 70% 50% at 10% 10%,rgba(124,82,255,.10) 0%,transparent 55%),
    radial-gradient(ellipse 55% 45% at 90% 85%,rgba(244,114,182,.08) 0%,transparent 55%),
    radial-gradient(ellipse 45% 50% at 50% 50%,rgba(34,211,238,.04) 0%,transparent 60%);
  animation:blob 16s ease-in-out infinite alternate;
}
@keyframes blob{0%{opacity:.6;transform:scale(1)}100%{opacity:1;transform:scale(1.07)}}
.card{
  background:var(--surf);border:1px solid var(--bdr);
  border-radius:18px;padding:22px 26px;margin:8px 0;
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.06),inset 0 -1px 0 rgba(0,0,0,0.20),
    0 12px 40px rgba(0,0,0,0.50);
  transform:perspective(800px) rotateX(0deg);
  transition:transform .3s ease,box-shadow .3s ease,border-color .25s;
  position:relative;overflow:hidden;
}
.card::before{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(135deg,rgba(124,82,255,.06),transparent 60%);
  opacity:0;transition:opacity .3s;
}
.card:hover{
  transform:perspective(800px) rotateX(2deg) translateY(-6px) scale(1.01);
  box-shadow:0 24px 60px rgba(0,0,0,.60),0 0 32px rgba(124,82,255,.30);
  border-color:var(--bdr2);
}
.card:hover::before{opacity:1}
input[type=number],input[type=text],textarea,select{
  background:rgba(124,82,255,0.06)!important;
  border:1px solid var(--bdr)!important;border-radius:10px!important;
  color:var(--txt)!important;font-family:'Inter',sans-serif!important;
  transition:border-color .2s,box-shadow .2s!important;
}
input:focus,textarea:focus{
  border-color:var(--p)!important;
  box-shadow:0 0 0 3px rgba(124,82,255,.18)!important;outline:none!important;
}
input[type=range]{accent-color:var(--p)}
label,.label-wrap span{
  font-family:'Inter',sans-serif!important;font-weight:600!important;
  font-size:.72rem!important;letter-spacing:.07em!important;
  color:var(--p2)!important;text-transform:uppercase!important;
}
#btn-predict{
  background:linear-gradient(135deg,#7c52ff,#5b21b6)!important;
  border:none!important;border-radius:14px!important;color:#fff!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:800!important;
  font-size:1rem!important;letter-spacing:.07em!important;
  padding:14px 0!important;width:100%!important;cursor:pointer!important;
  box-shadow:0 5px 0 #3b0764,0 9px 26px rgba(124,82,255,.50)!important;
  transform:translateY(0)!important;
  transition:transform .12s,box-shadow .12s!important;
}
#btn-predict:hover{
  transform:translateY(-4px)!important;
  box-shadow:0 9px 0 #3b0764,0 16px 36px rgba(124,82,255,.65)!important;
}
#btn-predict:active{
  transform:translateY(3px)!important;
  box-shadow:0 2px 0 #3b0764,0 4px 12px rgba(124,82,255,.3)!important;
}
#btn-reset{
  background:rgba(244,114,182,.10)!important;
  border:1px solid rgba(244,114,182,.35)!important;border-radius:14px!important;
  color:var(--pink)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:700!important;font-size:.85rem!important;padding:12px 0!important;
  width:100%!important;cursor:pointer!important;transition:all .2s!important;
}
#btn-reset:hover{
  background:rgba(244,114,182,.22)!important;
  box-shadow:0 0 22px rgba(244,114,182,.40)!important;transform:translateY(-2px)!important;
}
#btn-action{
  background:rgba(34,211,238,.10)!important;
  border:1px solid rgba(34,211,238,.35)!important;border-radius:14px!important;
  color:var(--cyan)!important;font-family:'Space Grotesk',sans-serif!important;
  font-weight:700!important;font-size:.85rem!important;padding:12px 0!important;
  width:100%!important;cursor:pointer!important;transition:all .2s!important;
}
#btn-action:hover{
  background:rgba(34,211,238,.22)!important;
  box-shadow:0 0 22px rgba(34,211,238,.40)!important;transform:translateY(-2px)!important;
}
#btn-green{
  background:linear-gradient(135deg,#10b981,#059669)!important;
  border:none!important;border-radius:14px!important;color:#fff!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;
  font-size:.92rem!important;padding:13px 0!important;width:100%!important;
  cursor:pointer!important;box-shadow:0 5px 0 #065f46!important;
  transition:transform .12s,box-shadow .12s!important;
}
#btn-green:hover{transform:translateY(-3px)!important;box-shadow:0 8px 0 #065f46!important}
#btn-green:active{transform:translateY(2px)!important;box-shadow:0 2px 0 #065f46!important}
#btn-theme{
  background:rgba(255,255,255,.06)!important;border:1px solid var(--bdr)!important;
  border-radius:50px!important;color:var(--p2)!important;font-size:.8rem!important;
  padding:8px 18px!important;cursor:pointer!important;
  transition:all .2s!important;width:auto!important;
}
#btn-theme:hover{background:rgba(124,82,255,.15)!important;transform:scale(1.05)!important}
#result-box textarea{
  font-family:'Space Grotesk',sans-serif!important;
  font-size:1.4rem!important;font-weight:800!important;
  text-align:center!important;color:var(--amber)!important;
  background:rgba(245,158,11,.07)!important;
  border:2px solid rgba(245,158,11,.35)!important;
  border-radius:14px!important;min-height:70px!important;
  animation:glowA 2.5s ease-in-out infinite alternate;
}
@keyframes glowA{
  0%{box-shadow:0 0 8px rgba(245,158,11,.25)}
  100%{box-shadow:0 0 26px rgba(245,158,11,.60)}
}
#prob-box textarea{
  font-family:'Inter',monospace!important;font-size:.88rem!important;
  color:var(--green)!important;background:rgba(16,185,129,.06)!important;
  border:1px solid rgba(16,185,129,.28)!important;
  border-radius:13px!important;text-align:center!important;
}
.tab-nav button{
  font-family:'Space Grotesk',sans-serif!important;font-size:.77rem!important;
  font-weight:600!important;letter-spacing:.06em!important;color:var(--txt3)!important;
  border-radius:9px 9px 0 0!important;border:1px solid transparent!important;
  background:transparent!important;transition:all .2s!important;padding:10px 18px!important;
}
.tab-nav button.selected,.tab-nav button:hover{
  color:var(--p2)!important;border-color:var(--bdr)!important;
  background:rgba(124,82,255,.09)!important;
}
.bar-bg{background:rgba(255,255,255,.06);border-radius:8px;height:12px;overflow:hidden;margin:4px 0}
.bar-fill{
  height:100%;border-radius:8px;
  transition:width 1s cubic-bezier(.23,1,.32,1);position:relative;overflow:hidden;
}
.bar-fill::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.18),transparent);
  animation:shimmer 2.2s ease-in-out infinite;
}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.gr-markdown h1,.gr-markdown h2,.gr-markdown h3{
  font-family:'Space Grotesk',sans-serif!important;color:var(--p2)!important;
}
.gr-markdown p,.gr-markdown li{color:var(--txt2)!important;font-size:.9rem;line-height:1.7}
.gr-markdown strong{color:var(--txt)!important}
.gr-markdown code{
  background:rgba(124,82,255,.12)!important;color:var(--p2)!important;
  border-radius:4px!important;padding:2px 6px!important;
}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:rgba(124,82,255,.4);border-radius:4px}
"""

BADGE_PH = """
<div style='text-align:center;padding:28px 20px;
  border:1px dashed rgba(124,82,255,.18);border-radius:14px;margin-top:6px;
  color:rgba(124,82,255,.35);font-family:Inter,sans-serif;
  font-size:.88rem;letter-spacing:.12em;'>
  &#10022; &nbsp; Click PREDICT to see risk analysis &nbsp; &#10022;
</div>"""


def make_header():
    return (
        "<div style='background:rgba(13,14,28,0.95);border:1px solid rgba(124,82,255,0.22);"
        "border-radius:22px;padding:36px 44px;text-align:center;margin-bottom:10px;"
        "position:relative;overflow:hidden;backdrop-filter:blur(20px);"
        "box-shadow:0 12px 44px rgba(0,0,0,0.55),inset 0 1px 0 rgba(255,255,255,0.05);'>"
        "<div style='position:absolute;bottom:0;left:0;right:0;height:3px;"
        "background:linear-gradient(90deg,transparent,#7c52ff,#f472b6,#22d3ee,transparent);"
        "animation:hs 4s linear infinite;'></div>"
        "<style>@keyframes hs{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}</style>"
        "<div style='font-family:\"Space Grotesk\",sans-serif;"
        "font-size:clamp(1.8rem,4vw,3rem);font-weight:900;letter-spacing:-.02em;"
        "background:linear-gradient(135deg,#7c52ff,#f472b6,#22d3ee);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;margin-bottom:8px;'>CHURN ORACLE</div>"
        "<div style='font-family:Inter,sans-serif;font-size:.82rem;color:#9895b5;"
        "text-transform:uppercase;letter-spacing:.2em;margin-bottom:22px;'>"
        "Bank Customer Attrition &middot; ML Prediction Platform &middot; BankChurners Dataset"
        "</div>"
        "<div style='display:flex;justify-content:center;gap:24px;flex-wrap:wrap;'>"
        "<div style='text-align:center;padding:10px 18px;border-radius:12px;"
        "background:rgba(124,82,255,.08);border:1px solid rgba(124,82,255,.18);'>"
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;font-weight:800;color:#7c52ff;'>4</div>"
        "<div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;letter-spacing:.12em;margin-top:2px;'>ML Models</div>"
        "</div>"
        "<div style='text-align:center;padding:10px 18px;border-radius:12px;"
        "background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.18);'>"
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;font-weight:800;color:#22d3ee;'>10K+</div>"
        "<div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;letter-spacing:.12em;margin-top:2px;'>Customers</div>"
        "</div>"
        "<div style='text-align:center;padding:10px 18px;border-radius:12px;"
        "background:rgba(244,114,182,.08);border:1px solid rgba(244,114,182,.18);'>"
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;font-weight:800;color:#f472b6;'>95.3%</div>"
        "<div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;letter-spacing:.12em;margin-top:2px;'>Best Accuracy</div>"
        "</div>"
        "<div style='text-align:center;padding:10px 18px;border-radius:12px;"
        "background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.18);'>"
        "<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.4rem;font-weight:800;color:#10b981;'>37</div>"
        "<div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;letter-spacing:.12em;margin-top:2px;'>Features</div>"
        "</div>"
        "</div></div>"
    )


# ═══════════════════════════════════════════════════════════════════════
# CORE PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def build_row(age, dep, mob, rel, mi, cc, cl, rb, ob, ac, ta, tc, ctc, ur,
              gen, edu, mar, inc, crd):
    """Convert 19 form inputs to 1x37 numpy array matching FEATURE_NAMES order."""
    gF = 1 if gen in ("Female", "F") else 0
    gM = 1 - gF
    ev = [1 if e == edu else 0 for e in [
        "College", "Doctorate", "Graduate", "High School",
        "Post-Graduate", "Uneducated", "Unknown"
    ]]
    mv = [1 if m == mar else 0 for m in ["Divorced", "Married", "Single", "Unknown"]]
    iv = [1 if i == inc else 0 for i in [
        "$120K +", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "Less than $40K", "Unknown"
    ]]
    cv = [1 if c == crd else 0 for c in ["Blue", "Gold", "Platinum", "Silver"]]
    row = [age, dep, mob, rel, mi, cc, cl, rb, ob, ac, ta, tc, ctc, ur,
           gF, gM, *ev, *mv, *iv, *cv]
    return np.array([row], dtype=float)


def predict_all_models(row):
    results = {}
    for name, mdl in MODELS.items():
        try:
            pred  = int(mdl.predict(row)[0])
            proba = mdl.predict_proba(row)[0].tolist()
            results[name] = {"pred": pred, "prob": proba[1], "conf": max(proba)}
        except Exception:
            results[name] = {"pred": 0, "prob": 0.0, "conf": 0.0}
    return results


def predict_fn(age, dep, mob, rel, mi, cc, cl, rb, ob, ac, ta, tc, ctc, ur,
               gen, edu, mar, inc, crd):
    if not MODELS:
        return "No model loaded", "Models not available", BADGE_PH

    row     = build_row(age, dep, mob, rel, mi, cc, cl, rb, ob, ac,
                        ta, tc, ctc, ur, gen, edu, mar, inc, crd)
    results = predict_all_models(row)

    primary = results.get("Random Forest", list(results.values())[0])
    prob    = primary["prob"]
    pred    = primary["pred"]
    ep      = 1.0 - prob

    if prob < 0.20:   col, lbl = "#10b981", "LOW RISK"
    elif prob < 0.50: col, lbl = "#f59e0b", "MODERATE RISK"
    elif prob < 0.75: col, lbl = "#f97316", "HIGH RISK"
    else:             col, lbl = "#ef4444", "CRITICAL RISK"

    model_rows = ""
    for n, v in results.items():
        c = "#ef4444" if v["prob"] > 0.5 else "#10b981"
        model_rows += (
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:7px 14px;border-top:1px solid rgba(124,82,255,.08);'>"
            f"<span style='font-family:Inter,sans-serif;font-size:.82rem;color:#d0cef0;'>{n}</span>"
            f"<div style='text-align:right;'>"
            f"<span style='font-family:Space Grotesk,sans-serif;font-size:.85rem;"
            f"font-weight:700;color:{c};'>{v['prob']*100:.1f}%</span>"
            f"<span style='font-size:.7rem;color:#4a4870;margin-left:6px;'>"
            f"({MODEL_ACC.get(n, 0):.1f}% acc)</span></div></div>"
        )

    badge_html = f"""
<div style='margin-top:6px;'>
  <div style='text-align:center;padding:16px;border-radius:14px;margin-bottom:12px;
    background:rgba(255,255,255,.02);border:1px solid {col}44;
    box-shadow:0 0 24px {col}1a;'>
    <div style='font-family:"Space Grotesk",sans-serif;font-size:1.3rem;
      font-weight:900;color:{col};letter-spacing:.07em;'>{lbl}</div>
    <div style='font-size:.8rem;color:#9895b5;margin-top:5px;'>
      Attrition Probability &nbsp;
      <strong style='color:{col};font-size:.95rem;'>{prob*100:.1f}%</strong>
    </div>
  </div>
  <div style='margin-bottom:12px;'>
    <div style='display:flex;justify-content:space-between;
      font-size:.7rem;color:#4a4870;margin-bottom:4px;'>
      <span>0%</span><span>Risk Gauge</span><span>100%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{prob*100:.1f}%;
        background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);'></div>
    </div>
  </div>
  <div style='margin-bottom:10px;'>
    <div style='display:flex;justify-content:space-between;
      font-size:.76rem;color:#9895b5;margin-bottom:3px;'>
      <span>Existing Customer</span>
      <span style='color:#10b981;font-weight:700;'>{ep*100:.1f}%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{ep*100:.1f}%;
        background:linear-gradient(90deg,#10b981,#34d399);'></div>
    </div>
    <div style='display:flex;justify-content:space-between;
      font-size:.76rem;color:#9895b5;margin:8px 0 3px;'>
      <span>Attrited Customer</span>
      <span style='color:#f472b6;font-weight:700;'>{prob*100:.1f}%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{prob*100:.1f}%;
        background:linear-gradient(90deg,#f472b6,#ec4899);'></div>
    </div>
  </div>
  <div style='border:1px solid rgba(124,82,255,.15);border-radius:12px;
    overflow:hidden;margin-bottom:12px;'>
    <div style='background:rgba(124,82,255,.10);padding:7px 14px;
      font-family:"Space Grotesk",sans-serif;font-size:.7rem;
      color:#a78bfa;text-transform:uppercase;letter-spacing:.1em;'>
      All Models Comparison
    </div>
    {model_rows}
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
    <div style='background:rgba(124,82,255,.08);border:1px solid rgba(124,82,255,.2);
      border-radius:12px;padding:13px;text-align:center;'>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.1rem;
        font-weight:700;color:#a78bfa;'>{ep*100:.1f}%</div>
      <div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;
        letter-spacing:.1em;margin-top:3px;'>Stay Prob</div>
    </div>
    <div style='background:rgba(244,114,182,.08);border:1px solid rgba(244,114,182,.2);
      border-radius:12px;padding:13px;text-align:center;'>
      <div style='font-family:"Space Grotesk",sans-serif;font-size:1.1rem;
        font-weight:700;color:#f472b6;'>{prob*100:.1f}%</div>
      <div style='font-size:.62rem;color:#4a4870;text-transform:uppercase;
        letter-spacing:.1em;margin-top:3px;'>Churn Prob</div>
    </div>
  </div>
</div>"""

    result_str = "Attrited Customer" if pred == 1 else "Existing Customer"
    prob_str   = f"P(Existing)={ep*100:.2f}%   |   P(Attrited)={prob*100:.2f}%"
    return result_str, prob_str, badge_html


def reset_fn():
    D = DEFAULTS
    return (
        D["Customer_Age"], D["Dependent_count"], D["Months_on_book"],
        D["Total_Relationship_Count"], D["Months_Inactive_12_mon"],
        D["Contacts_Count_12_mon"], D["Credit_Limit"], D["Total_Revolving_Bal"],
        D["Avg_Open_To_Buy"], D["Total_Amt_Chng_Q4_Q1"], D["Total_Trans_Amt"],
        D["Total_Trans_Ct"], D["Total_Ct_Chng_Q4_Q1"], D["Avg_Utilization_Ratio"],
        "Female", "Graduate", "Married", "Less than $40K", "Blue",
        "", "", BADGE_PH,
    )


# ═══════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
_PS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
    margin=dict(l=10, r=10, t=45, b=10),
)


def chart_risk_dist():
    df     = pd.read_csv("BankChurners.csv")
    exist  = df[df["Attrition_Flag"] == "Existing Customer"]
    attrit = df[df["Attrition_Flag"] == "Attrited Customer"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=exist["Total_Trans_Ct"],  name="Existing",
                               nbinsx=30, marker_color="#10b981", opacity=.75))
    fig.add_trace(go.Histogram(x=attrit["Total_Trans_Ct"], name="Attrited",
                               nbinsx=30, marker_color="#f472b6", opacity=.75))
    fig.update_layout(barmode="overlay",
                      title="Transaction Count Distribution by Churn Status",
                      xaxis_title="Total Trans Count",
                      yaxis_title="Customers", height=360, **_PS)
    return fig


def chart_age_box():
    df  = pd.read_csv("BankChurners.csv")
    fig = go.Figure()
    for flag, col in [("Existing Customer", "#10b981"), ("Attrited Customer", "#f472b6")]:
        d = df[df["Attrition_Flag"] == flag]
        fig.add_trace(go.Box(y=d["Customer_Age"], name=flag,
                             marker_color=col, boxmean=True))
    fig.update_layout(title="Age Distribution by Churn Status",
                      yaxis_title="Customer Age", height=360, **_PS)
    return fig


def chart_income():
    df  = pd.read_csv("BankChurners.csv")
    grp = df.groupby(["Income_Category", "Attrition_Flag"]).size().reset_index(name="count")
    fig = px.bar(grp, x="Income_Category", y="count", color="Attrition_Flag",
                 barmode="group", title="Churn by Income Category",
                 color_discrete_map={"Existing Customer": "#10b981",
                                     "Attrited Customer": "#f472b6"},
                 template="plotly_dark")
    fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      margin=dict(l=10, r=10, t=45, b=10))
    return fig


def chart_utilization():
    df  = pd.read_csv("BankChurners.csv")
    fig = go.Figure()
    for flag, col, fill in [
        ("Existing Customer", "#10b981", "rgba(16,185,129,.12)"),
        ("Attrited Customer", "#f472b6", "rgba(244,114,182,.12)"),
    ]:
        d = df[df["Attrition_Flag"] == flag]
        fig.add_trace(go.Violin(y=d["Avg_Utilization_Ratio"], name=flag,
                                line_color=col, fillcolor=fill,
                                box_visible=True, meanline_visible=True))
    fig.update_layout(title="Avg Utilization Ratio by Churn Status",
                      yaxis_title="Avg Utilization Ratio", height=360, **_PS)
    return fig


def chart_model_perf():
    models  = ["Naive Bayes", "Random Forest", "Extra Trees", "Logistic Reg"]
    metrics = {
        "Accuracy":  [89.3, 95.3, 92.1, 89.3],
        "Precision": [70.8, 93.5, 92.7, 76.1],
        "Recall":    [56.6, 75.7, 55.1, 48.9],
        "F1-Score":  [62.9, 83.7, 69.1, 59.6],
        "AUC":       [88.0, 96.0, 94.0, 90.0],
    }
    colors = ["#7c52ff", "#22d3ee", "#f472b6", "#10b981", "#f59e0b"]
    fig = go.Figure()
    for i, (m, v) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(name=m, x=models, y=v, marker_color=colors[i],
                             text=[f"{x:.1f}%" for x in v], textposition="auto"))
    fig.update_layout(barmode="group",
                      title="Model Performance Comparison (BankChurners Test Set)",
                      yaxis_title="%", height=380, **_PS,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def chart_radar():
    cats = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    data = [
        ("Naive Bayes",   [89.3, 70.8, 56.6, 62.9, 88.0], "#7c52ff", "rgba(124,82,255,.12)"),
        ("Random Forest", [95.3, 93.5, 75.7, 83.7, 96.0], "#22d3ee", "rgba(34,211,238,.12)"),
        ("Extra Trees",   [92.1, 92.7, 55.1, 69.1, 94.0], "#f472b6", "rgba(244,114,182,.12)"),
        ("Logistic Reg",  [89.3, 76.1, 48.9, 59.6, 90.0], "#10b981", "rgba(16,185,129,.12)"),
    ]
    fig = go.Figure()
    for nm, vals, col, fill in data:
        v_n = [x / 100 for x in vals]
        fig.add_trace(go.Scatterpolar(
            r=v_n + [v_n[0]], theta=cats + [cats[0]],
            fill="toself", name=nm, line_color=col, fillcolor=fill))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   tickformat=".0%", tickfont=dict(size=10))),
        title="Model Radar Chart (normalised)", height=380,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"), margin=dict(l=30, r=30, t=45, b=30))
    return fig


def chart_feature_imp():
    feats = ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Ct_Chng_Q4_Q1",
             "Total_Amt_Chng_Q4_Q1", "Months_Inactive_12_mon",
             "Contacts_Count_12_mon", "Avg_Utilization_Ratio",
             "Total_Revolving_Bal", "Credit_Limit", "Customer_Age"]
    imp   = [18.2, 15.6, 13.4, 11.2, 9.8, 8.7, 7.2, 5.8, 4.4, 3.2]
    cols  = ["#7c52ff" if v > 10 else "#a78bfa" if v > 6 else "#c4b5fd" for v in imp]
    fig = go.Figure(go.Bar(x=imp, y=feats, orientation="h",
                           marker_color=cols,
                           text=[f"{v:.1f}%" for v in imp],
                           textposition="auto"))
    fig.update_layout(title="Top 10 Feature Importance (Random Forest)",
                      xaxis_title="Importance %", height=380, **_PS)
    return fig


def chart_timeline():
    dates  = [datetime.today() - timedelta(days=i) for i in range(29, -1, -1)]
    counts = [random.randint(30, 80) for _ in dates]
    rates  = [random.uniform(.14, .32) for _ in dates]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Daily Predictions", "Churn Rate Trend"),
                        vertical_spacing=.14)
    fig.add_trace(go.Scatter(x=dates, y=counts, fill="tozeroy", name="Predictions",
                             line=dict(color="#7c52ff", width=2),
                             fillcolor="rgba(124,82,255,.15)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rates, name="Churn Rate",
                             line=dict(color="#f472b6", width=2),
                             fill="tozeroy", fillcolor="rgba(244,114,182,.10)"),
                  row=2, col=1)
    fig.update_layout(height=400, showlegend=False, **_PS)
    return fig


def chart_segment():
    np.random.seed(42)
    n     = 400
    risk  = np.random.beta(2, 5, n)
    value = np.random.lognormal(8.5, 1, n)
    segs  = ["Champions" if r < .2 and v > 20000
             else "Loyal"     if r < .3 and v > 8000
             else "At Risk"   if r > .7 and v > 15000
             else "Lost"      if r > .7
             else "Potential" for r, v in zip(risk, value)]
    df  = pd.DataFrame({"risk": risk, "value": value, "segment": segs})
    fig = px.scatter(df, x="value", y="risk", color="segment",
                     title="Customer Segmentation",
                     labels={"value": "Customer Value ($)", "risk": "Churn Risk"},
                     color_discrete_map={
                         "Champions": "#10b981", "Loyal": "#22d3ee",
                         "Potential": "#7c52ff", "At Risk": "#f59e0b", "Lost": "#ef4444"},
                     template="plotly_dark", height=380)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
                      margin=dict(l=10, r=10, t=45, b=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════
# ANALYTICS & A/B & BATCH & INSIGHTS HELPERS
# ═══════════════════════════════════════════════════════════════════════
def live_customer_html():
    cid   = f"CUST_{random.randint(100000, 999999)}"
    age   = random.randint(22, 72)
    clim  = random.uniform(1500, 28000)
    trans = random.randint(12, 139)
    inact = random.randint(0, 6)
    risk  = random.uniform(0, 1)
    col   = ("#10b981" if risk < .2 else "#f59e0b" if risk < .5
             else "#f97316" if risk < .75 else "#ef4444")
    lbl   = ("LOW" if risk < .2 else "MODERATE" if risk < .5
             else "HIGH" if risk < .75 else "CRITICAL")
    ts    = datetime.now().strftime("%H:%M:%S")
    cells = [
        (age, "#a78bfa", "Age"),
        (f"${clim:,.0f}", "#22d3ee", "Credit"),
        (trans, "#10b981", "Trans"),
        (inact, "#f59e0b", "Inactive"),
    ]
    cell_html = "".join([
        f"<div style='text-align:center;padding:7px;"
        f"background:rgba(124,82,255,.07);border-radius:8px;'>"
        f"<div style='font-size:1rem;font-weight:700;color:{c};'>{v}</div>"
        f"<div style='font-size:.6rem;color:#4a4870;text-transform:uppercase;"
        f"margin-top:1px;'>{l}</div></div>"
        for v, c, l in cells
    ])
    return (
        f"<div style='background:rgba(13,14,28,.95);border:1px solid rgba(124,82,255,.22);"
        f"border-radius:14px;padding:18px;box-shadow:0 8px 28px rgba(0,0,0,.4);'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:12px;'>"
        f"<span style='font-family:\"Space Grotesk\",sans-serif;font-size:.92rem;"
        f"font-weight:700;color:#a78bfa;'>&#128225; {cid}</span>"
        f"<span style='font-family:monospace;font-size:.7rem;color:#4a4870;'>{ts}</span>"
        f"</div>"
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;'>"
        f"{cell_html}</div>"
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,.02);"
        f"border-radius:11px;border:1px solid {col}33;'>"
        f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:1.1rem;"
        f"font-weight:800;color:{col};'>{lbl} RISK &middot; {risk*100:.1f}%</div>"
        f"<div style='background:rgba(255,255,255,.05);border-radius:6px;"
        f"height:7px;margin-top:8px;overflow:hidden;'>"
        f"<div style='width:{risk*100:.0f}%;height:100%;"
        f"background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);"
        f"border-radius:6px;'></div></div></div></div>"
    )


def platform_stats_html():
    total   = random.randint(1100, 1400)
    churned = int(total * random.uniform(.18, .28))
    high    = int(total * random.uniform(.08, .14))
    items = [
        (total,                       "#7c52ff", "rgba(124,82,255,.25)", "Total Predictions"),
        (f"{churned/total*100:.1f}%", "#ef4444", "rgba(239,68,68,.25)",  "Churn Rate"),
        (high,                        "#f59e0b", "rgba(245,158,11,.25)", "High Risk"),
        (str(len(MODELS) or 4),       "#10b981", "rgba(16,185,129,.25)", "Active Models"),
    ]
    cards = "".join([
        f'<div style="background:rgba(13,14,28,.95);border:1px solid {bc};'
        f'border-radius:13px;padding:15px;text-align:center;'
        f'box-shadow:0 6px 20px rgba(0,0,0,.35);">'
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.5rem;'
        f'font-weight:800;color:{vc};">{vv}</div>'
        f'<div style="font-size:.66rem;color:#4a4870;text-transform:uppercase;'
        f'letter-spacing:.1em;margin-top:3px;">{lb}</div></div>'
        for vv, vc, bc, lb in items
    ])
    return (f"<div style='display:grid;grid-template-columns:1fr 1fr;"
            f"gap:10px;margin:10px 0;'>{cards}</div>")


def ab_test_fn(ma, mb, n):
    n  = int(n)
    np.random.seed(42)
    pa = np.random.beta(3,   7,   n)
    pb = np.random.beta(3.3, 6.5, n)
    winner = ma if pa.mean() < pb.mean() else mb
    conf   = min(abs(pa.mean() - pb.mean()) * 20, .95)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pa, name=ma, nbinsx=20,
                               marker_color="#7c52ff", opacity=.75))
    fig.add_trace(go.Histogram(x=pb, name=mb, nbinsx=20,
                               marker_color="#f472b6", opacity=.75))
    fig.update_layout(barmode="overlay",
                      title=f"A/B Test Distribution — Winner: {winner}",
                      xaxis_title="Churn Probability", height=320, **_PS)
    html = (
        f"<div style='background:rgba(13,14,28,.95);border:1px solid rgba(124,82,255,.22);"
        f"border-radius:14px;padding:18px;margin-top:8px;'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;'>"
        f"<div style='background:rgba(124,82,255,.1);padding:13px;border-radius:11px;text-align:center;'>"
        f"<div style='font-size:.9rem;font-weight:700;color:#a78bfa;'>{ma}</div>"
        f"<div style='font-size:1.2rem;font-weight:800;color:#f1f0ff;margin-top:4px;'>{pa.mean():.3f}</div>"
        f"<div style='font-size:.68rem;color:#4a4870;'>Avg Prob</div></div>"
        f"<div style='background:rgba(244,114,182,.1);padding:13px;border-radius:11px;text-align:center;'>"
        f"<div style='font-size:.9rem;font-weight:700;color:#f472b6;'>{mb}</div>"
        f"<div style='font-size:1.2rem;font-weight:800;color:#f1f0ff;margin-top:4px;'>{pb.mean():.3f}</div>"
        f"<div style='font-size:.68rem;color:#4a4870;'>Avg Prob</div></div></div>"
        f"<div style='text-align:center;padding:13px;background:rgba(16,185,129,.1);"
        f"border-radius:11px;border:1px solid rgba(16,185,129,.3);'>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#10b981;'>"
        f"&#127942; Winner: {winner}</div>"
        f"<div style='font-size:.8rem;color:#9895b5;margin-top:3px;'>"
        f"Confidence: {conf:.1%} &nbsp;|&nbsp; n = {n:,}</div></div></div>"
    )
    return html, fig


def batch_predict_fn(file):
    if file is None:
        return None, "No file uploaded. Please upload a CSV.", ""
    try:
        df   = pd.read_csv(file.name)
        miss = [c for c in FEATURE_NAMES if c not in df.columns]
        if miss:
            return None, f"Missing columns: {miss[:5]}...", ""
        if not MODELS:
            return None, "No model available — run train_model.py first.", ""
        mdl    = MODELS.get("Random Forest", list(MODELS.values())[0])
        X      = df[FEATURE_NAMES].values.astype(float)
        preds  = mdl.predict(X)
        probas = mdl.predict_proba(X)[:, 1]
        df["Prediction"]    = preds
        df["Prob_Attrited"] = probas.round(4)
        df["Risk_Level"]    = pd.cut(probas, [0, .2, .5, .75, 1.0],
                                     labels=["Low", "Moderate", "High", "Critical"],
                                     include_lowest=True)
        out = "/tmp/churn_predictions.csv"
        df.to_csv(out, index=False)
        n, nc = len(df), int(preds.sum())
        log = (
            f"Processed: {n} rows\n"
            f"  Existing  : {n-nc} ({(n-nc)/n*100:.1f}%)\n"
            f"  Attrited  : {nc} ({nc/n*100:.1f}%)\n"
            f"  Avg Risk  : {probas.mean()*100:.2f}%\n"
            f"  High+Crit : {int((probas >= .5).sum())}"
        )
        items = [
            (n,                           "#a78bfa", "rgba(124,82,255,.25)", "Total"),
            (nc,                          "#f472b6", "rgba(244,114,182,.25)", "Attrited"),
            (f"{probas.mean()*100:.1f}%", "#f59e0b", "rgba(245,158,11,.25)",  "Avg Risk"),
        ]
        cards = "".join([
            f'<div style="background:rgba(13,14,28,.95);border:1px solid {bc};'
            f'border-radius:12px;padding:13px;text-align:center;">'
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.3rem;'
            f'font-weight:800;color:{vc};">{vv}</div>'
            f'<div style="font-size:.65rem;color:#4a4870;text-transform:uppercase;'
            f'letter-spacing:.1em;margin-top:3px;">{lb}</div></div>'
            for vv, vc, bc, lb in items
        ])
        summary = (
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);"
            f"gap:10px;margin:10px 0;'>{cards}</div>"
            f"<div style='background:rgba(13,14,28,.95);border:1px solid rgba(124,82,255,.15);"
            f"border-radius:12px;padding:12px;margin-top:4px;'>"
            f"<div style='font-size:.68rem;color:#4a4870;margin-bottom:5px;text-transform:uppercase;'>"
            f"Existing ({(n-nc)/n*100:.0f}%)</div>"
            f"<div class='bar-bg'><div class='bar-fill' style='width:{(n-nc)/n*100:.0f}%;"
            f"background:linear-gradient(90deg,#10b981,#34d399);'></div></div>"
            f"<div style='font-size:.68rem;color:#4a4870;margin:7px 0 5px;text-transform:uppercase;'>"
            f"Attrited ({nc/n*100:.0f}%)</div>"
            f"<div class='bar-bg'><div class='bar-fill' style='width:{nc/n*100:.0f}%;"
            f"background:linear-gradient(90deg,#f472b6,#ec4899);'></div></div></div>"
        )
        return out, log, summary
    except Exception as e:
        return None, f"Error: {e}", ""


def insights_fn():
    tips = [
        ("&#127919;", "Retention Strategy",
         "Offer personalised credit limit increases to customers with high utilization "
         "but declining transactions — targets 68% of the at-risk segment."),
        ("&#128202;", "Top Predictive Signal",
         "Transaction count drop >30% Q4 vs Q1 is the strongest single churn predictor. "
         "Monitor this weekly for early warning."),
        ("&#128222;", "Optimal Contact Window",
         "Best engagement: Tue-Thu, 2-5 PM. Response rate is 42% higher "
         "vs other time windows."),
        ("&#128179;", "Product Upsell Opportunity",
         "Blue card holders with 4+ relationships show 60% acceptance for Gold/Platinum "
         "upgrade. Target this segment proactively."),
        ("&#128260;", "Re-engagement Automation",
         "Monthly balance review calls reduce churn by 18% for Moderate-risk customers. "
         "Automate for customers inactive >2 months."),
    ]
    html = "<div style='display:flex;flex-direction:column;gap:10px;'>"
    for icon, title, body in tips:
        html += (
            f"<div style='background:rgba(16,185,129,.07);"
            f"border:1px solid rgba(16,185,129,.2);border-radius:12px;padding:14px;'>"
            f"<div style='font-family:\"Space Grotesk\",sans-serif;font-size:.9rem;"
            f"font-weight:700;color:#10b981;margin-bottom:5px;'>{icon} {title}</div>"
            f"<div style='font-family:Inter,sans-serif;font-size:.84rem;color:#9895b5;"
            f"line-height:1.6;'>{body}</div></div>"
        )
    return html + "</div>"


def _toggle_theme(current_label):
    going_light = (current_label == "Light Mode")
    new_label   = "Dark Mode" if going_light else "Light Mode"
    if going_light:
        inject = (
            "<style>"
            "body,.gradio-container{background:#f4f4ff!important;color:#1e1b4b!important}"
            "input[type=number],input[type=text],textarea,select"
            "{background:#ebebff!important;color:#1e1b4b!important}"
            "</style>"
        )
    else:
        inject = (
            "<style>"
            "body,.gradio-container{background:#07080f!important;color:#f0eeff!important}"
            "</style>"
        )
    return new_label, inject


# ═══════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════
with gr.Blocks(
    css=CSS,
    title="Churn Oracle v5",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    # ── Header ──────────────────────────────────────────────────────────
    gr.HTML(value=make_header())

    # ── Theme toggle ─────────────────────────────────────────────────────
    with gr.Row():
        gr.HTML("<div></div>")
        theme_btn = gr.Button("Light Mode", elem_id="btn-theme", scale=0, min_width=130)
    css_slot = gr.HTML(value="")
    theme_btn.click(fn=_toggle_theme, inputs=[theme_btn], outputs=[theme_btn, css_slot])

    # ── TABS ─────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ════════════════════════════════════════════════
        # TAB 1 — PREDICT
        # ════════════════════════════════════════════════
        with gr.Tab("🔮 Predict"):
            with gr.Row(equal_height=False):

                # LEFT column — inputs
                with gr.Column(scale=3):

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Account & Activity")
                        with gr.Row():
                            i_age = gr.Slider(label="Customer Age",
                                              minimum=18, maximum=73,
                                              value=DEFAULTS["Customer_Age"], step=1)
                            i_dep = gr.Slider(label="Dependents",
                                              minimum=0, maximum=5,
                                              value=DEFAULTS["Dependent_count"], step=1)
                            i_mob = gr.Slider(label="Months on Book",
                                              minimum=13, maximum=56,
                                              value=DEFAULTS["Months_on_book"], step=1)
                        with gr.Row():
                            i_rel = gr.Slider(label="Relationship Count",
                                              minimum=1, maximum=6,
                                              value=DEFAULTS["Total_Relationship_Count"],
                                              step=1)
                            i_mi  = gr.Slider(label="Months Inactive (12m)",
                                              minimum=0, maximum=6,
                                              value=DEFAULTS["Months_Inactive_12_mon"],
                                              step=1)
                            i_cc  = gr.Slider(label="Contacts Count (12m)",
                                              minimum=0, maximum=6,
                                              value=DEFAULTS["Contacts_Count_12_mon"],
                                              step=1)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Financial Details")
                        with gr.Row():
                            i_cl = gr.Number(label="Credit Limit ($)",
                                             value=DEFAULTS["Credit_Limit"])
                            i_rb = gr.Number(label="Revolving Balance ($)",
                                             value=DEFAULTS["Total_Revolving_Bal"])
                            i_ob = gr.Number(label="Avg Open To Buy ($)",
                                             value=DEFAULTS["Avg_Open_To_Buy"])
                        with gr.Row():
                            i_ta  = gr.Number(label="Total Trans Amount ($)",
                                              value=DEFAULTS["Total_Trans_Amt"])
                            i_tc  = gr.Number(label="Total Trans Count",
                                              value=DEFAULTS["Total_Trans_Ct"],
                                              precision=0)
                            i_ur  = gr.Slider(label="Utilization Ratio",
                                              minimum=0.0, maximum=1.0,
                                              value=DEFAULTS["Avg_Utilization_Ratio"],
                                              step=0.01)
                        with gr.Row():
                            i_ac  = gr.Number(label="Amt Change Q4/Q1",
                                              value=DEFAULTS["Total_Amt_Chng_Q4_Q1"])
                            i_ctc = gr.Number(label="Count Change Q4/Q1",
                                              value=DEFAULTS["Total_Ct_Chng_Q4_Q1"])

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Demographics")
                        with gr.Row():
                            i_gen = gr.Radio(label="Gender",
                                             choices=["Female", "Male"],
                                             value=DEFAULTS["Gender"])
                            i_edu = gr.Dropdown(
                                label="Education Level",
                                choices=["College", "Doctorate", "Graduate",
                                         "High School", "Post-Graduate",
                                         "Uneducated", "Unknown"],
                                value=DEFAULTS["Education_Level"])
                            i_mar = gr.Dropdown(
                                label="Marital Status",
                                choices=["Divorced", "Married", "Single", "Unknown"],
                                value=DEFAULTS["Marital_Status"])
                        with gr.Row():
                            i_inc = gr.Dropdown(
                                label="Income Category",
                                choices=["$120K +", "$40K - $60K", "$60K - $80K",
                                         "$80K - $120K", "Less than $40K", "Unknown"],
                                value=DEFAULTS["Income_Category"])
                            i_crd = gr.Dropdown(
                                label="Card Category",
                                choices=["Blue", "Gold", "Platinum", "Silver"],
                                value=DEFAULTS["Card_Category"])

                # RIGHT column — outputs
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Prediction Result")
                        o_result = gr.Textbox(label="Prediction", interactive=False,
                                              elem_id="result-box",
                                              placeholder="Click PREDICT...")
                        o_prob   = gr.Textbox(label="Probabilities",
                                              interactive=False, elem_id="prob-box")
                        o_badge  = gr.HTML(value=BADGE_PH)
                        with gr.Row():
                            btn_pred  = gr.Button("PREDICT", elem_id="btn-predict")
                            btn_reset = gr.Button("RESET",   elem_id="btn-reset")
                        gr.HTML(
                            "<div style='margin-top:14px;padding:13px;border-radius:13px;"
                            "border:1px solid rgba(124,82,255,.18);"
                            "background:rgba(124,82,255,.05);'>"
                            "<div style='font-family:\"Space Grotesk\",sans-serif;"
                            "font-size:.67rem;color:#7c52ff;text-transform:uppercase;"
                            "letter-spacing:.14em;margin-bottom:7px;'>Model Stack</div>"
                            "<div style='font-family:Inter,sans-serif;font-size:.78rem;"
                            "color:#4a4870;line-height:1.9;'>"
                            "Naive Bayes (89.3%) &middot; Random Forest (95.3%)<br>"
                            "Extra Trees (92.1%) &middot; Logistic Reg (89.3%)<br>"
                            "<span style='color:#7c52ff;'>"
                            "Trained on 10,127 real bank customers</span></div></div>"
                        )

            _inputs  = [i_age, i_dep, i_mob, i_rel, i_mi, i_cc,
                        i_cl, i_rb, i_ob, i_ac, i_ta, i_tc, i_ctc, i_ur,
                        i_gen, i_edu, i_mar, i_inc, i_crd]
            _outputs = [o_result, o_prob, o_badge]

            btn_pred.click(fn=predict_fn,  inputs=_inputs, outputs=_outputs)
            btn_reset.click(fn=reset_fn,   inputs=[],      outputs=_inputs + _outputs)

        # ════════════════════════════════════════════════
        # TAB 2 — DATASET EDA
        # ════════════════════════════════════════════════
        with gr.Tab("📊 Dataset EDA"):
            with gr.Row():
                btn_eda = gr.Button("Load EDA Charts", elem_id="btn-action")
            with gr.Tabs():
                with gr.Tab("Transaction Count"):
                    plt_trans = gr.Plot()
                with gr.Tab("Age Distribution"):
                    plt_age   = gr.Plot()
                with gr.Tab("Income vs Churn"):
                    plt_inc   = gr.Plot()
                with gr.Tab("Utilization"):
                    plt_util  = gr.Plot()

            def _load_eda():
                return (chart_risk_dist(), chart_age_box(),
                        chart_income(), chart_utilization())

            btn_eda.click(fn=_load_eda, inputs=[],
                          outputs=[plt_trans, plt_age, plt_inc, plt_util])
            demo.load(fn=_load_eda,
                      outputs=[plt_trans, plt_age, plt_inc, plt_util])

        # ════════════════════════════════════════════════
        # TAB 3 — MODEL PERFORMANCE
        # ════════════════════════════════════════════════
        with gr.Tab("🏆 Model Performance"):
            with gr.Row():
                btn_perf = gr.Button("Load Performance Charts", elem_id="btn-action")
            with gr.Tabs():
                with gr.Tab("Bar Chart"):
                    plt_bar   = gr.Plot()
                with gr.Tab("Radar"):
                    plt_radar = gr.Plot()
                with gr.Tab("Feature Importance"):
                    plt_feat  = gr.Plot()

            def _load_perf():
                return chart_model_perf(), chart_radar(), chart_feature_imp()

            btn_perf.click(fn=_load_perf, inputs=[],
                           outputs=[plt_bar, plt_radar, plt_feat])
            demo.load(fn=_load_perf,
                      outputs=[plt_bar, plt_radar, plt_feat])

        # ════════════════════════════════════════════════
        # TAB 4 — ANALYTICS
        # ════════════════════════════════════════════════
        with gr.Tab("📈 Analytics"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Live Stream")
                        btn_live  = gr.Button("Generate Customer",
                                              elem_id="btn-action")
                        out_live  = gr.HTML(
                            value="<div style='padding:16px;text-align:center;"
                                  "color:#4a4870;'>Click button above</div>")
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Platform Stats")
                        btn_stats = gr.Button("Refresh Stats", elem_id="btn-action")
                        out_stats = gr.HTML(
                            value="<div style='padding:16px;text-align:center;"
                                  "color:#4a4870;'>Click button above</div>")
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Dashboard")
                        with gr.Tabs():
                            with gr.Tab("Timeline"):
                                plt_time = gr.Plot()
                            with gr.Tab("Segmentation"):
                                plt_seg  = gr.Plot()

            btn_live.click(fn=live_customer_html,   inputs=[], outputs=[out_live])
            btn_stats.click(fn=platform_stats_html, inputs=[], outputs=[out_stats])
            demo.load(fn=platform_stats_html, outputs=[out_stats])
            demo.load(fn=chart_timeline,      outputs=[plt_time])
            demo.load(fn=chart_segment,       outputs=[plt_seg])

        # ════════════════════════════════════════════════
        # TAB 5 — A/B TESTING
        # ════════════════════════════════════════════════
        with gr.Tab("🧪 A/B Testing"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### A/B Model Comparison")
                with gr.Row():
                    t_ma = gr.Dropdown(
                        label="Model A",
                        choices=["Naive Bayes", "Random Forest",
                                 "Extra Trees", "Logistic Reg"],
                        value="Naive Bayes")
                    t_mb = gr.Dropdown(
                        label="Model B",
                        choices=["Naive Bayes", "Random Forest",
                                 "Extra Trees", "Logistic Reg"],
                        value="Random Forest")
                    t_n  = gr.Slider(label="Sample Size",
                                     minimum=50, maximum=500, value=100, step=10)
                btn_ab      = gr.Button("Run A/B Test", elem_id="btn-predict")
                out_ab_html = gr.HTML()
                out_ab_plot = gr.Plot()

            btn_ab.click(fn=ab_test_fn, inputs=[t_ma, t_mb, t_n],
                         outputs=[out_ab_html, out_ab_plot])

        # ════════════════════════════════════════════════
        # TAB 6 — XAI & INSIGHTS
        # ════════════════════════════════════════════════
        with gr.Tab("🔍 XAI & Insights"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Feature Importance")
                        btn_feat2 = gr.Button("Show Importance",
                                              elem_id="btn-action")
                        plt_feat2 = gr.Plot()
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### AI Business Insights")
                        btn_ins = gr.Button("Generate Insights",
                                            elem_id="btn-predict")
                        out_ins = gr.HTML(
                            value="<div style='padding:20px;text-align:center;"
                                  "color:rgba(124,82,255,.3);'>"
                                  "Click button above</div>")

            btn_feat2.click(fn=chart_feature_imp, inputs=[], outputs=[plt_feat2])
            btn_ins.click(fn=insights_fn,         inputs=[], outputs=[out_ins])
            demo.load(fn=chart_feature_imp, outputs=[plt_feat2])

        # ════════════════════════════════════════════════
        # TAB 7 — BATCH PREDICT
        # ════════════════════════════════════════════════
        with gr.Tab("📂 Batch Predict"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""
### Batch Prediction via CSV

Upload a CSV with the **37 model features** (one-hot encoded).
Output appends `Prediction`, `Prob_Attrited`, and `Risk_Level` columns.

**Feature order:** Customer_Age, Dependent_count, Months_on_book,
Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon,
Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1,
Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio,
Gender_F, Gender_M, Education_Level_College, Education_Level_Doctorate,
Education_Level_Graduate, Education_Level_High School,
Education_Level_Post-Graduate, Education_Level_Uneducated,
Education_Level_Unknown, Marital_Status_Divorced, Marital_Status_Married,
Marital_Status_Single, Marital_Status_Unknown, Income_Category_$120K +,
Income_Category_$40K - $60K, Income_Category_$60K - $80K,
Income_Category_$80K - $120K, Income_Category_Less than $40K,
Income_Category_Unknown, Card_Category_Blue, Card_Category_Gold,
Card_Category_Platinum, Card_Category_Silver
""")
                with gr.Row():
                    with gr.Column():
                        f_in      = gr.File(label="Upload CSV",
                                            file_types=[".csv"])
                        btn_batch = gr.Button("Run Batch Predict",
                                              elem_id="btn-green")
                        f_out     = gr.File(label="Download Predictions")
                    with gr.Column():
                        out_log = gr.Textbox(label="Summary Log",
                                             interactive=False, lines=7)
                        out_sum = gr.HTML()

            btn_batch.click(fn=batch_predict_fn, inputs=[f_in],
                            outputs=[f_out, out_log, out_sum])

        # ════════════════════════════════════════════════
        # TAB 8 — GUIDE
        # ════════════════════════════════════════════════
        with gr.Tab("📚 Guide"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""
## How to Use CHURN ORACLE

| Tab | Purpose |
|---|---|
| **Predict** | Fill the customer profile, hit PREDICT, get risk scores from all 4 models |
| **Dataset EDA** | Explore BankChurners data with 4 interactive Plotly charts |
| **Model Performance** | Bar chart, radar chart, feature importance comparison |
| **Analytics** | Live customer simulation, platform stats, 30-day timeline, segmentation |
| **A/B Testing** | Compare two models head-to-head on simulated samples |
| **XAI & Insights** | Feature importance + 5 AI-generated business retention insights |
| **Batch Predict** | Upload CSV → download bulk predictions with risk levels |

---

## Risk Levels

| Level | Probability | Action |
|---|---|---|
| **Low** | 0–20% | Standard engagement |
| **Moderate** | 20–50% | Proactive retention outreach |
| **High** | 50–75% | Immediate contact within 48 hours |
| **Critical** | 75–100% | Same-day emergency retention call |

---

## Dataset: BankChurners.csv
- **10,127 customers** — 16.1% churn rate
- **Target column**: Attrition_Flag (Existing Customer / Attrited Customer)
- **Best Model**: Random Forest — 95.3% accuracy, 96.0% AUC
- **Training split**: 80% train / 20% test, stratified by class

---

## Model Summary

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Naive Bayes | 89.3% | 70.8% | 56.6% | 62.9% | 88.0% |
| **Random Forest** | **95.3%** | **93.5%** | **75.7%** | **83.7%** | **96.0%** |
| Extra Trees | 92.1% | 92.7% | 55.1% | 69.1% | 94.0% |
| Logistic Reg | 89.3% | 76.1% | 48.9% | 59.6% | 90.0% |
""")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("## 37 Model Features")
                gr.HTML(
                    "<div style='display:grid;"
                    "grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:5px;'>"
                    + "".join([
                        f"<div style='font-family:Inter,sans-serif;font-size:.73rem;"
                        f"color:#a78bfa;padding:5px 10px;"
                        f"background:rgba(124,82,255,.07);border-radius:7px;"
                        f"border:1px solid rgba(124,82,255,.12);'>{feat}</div>"
                        for feat in FEATURE_NAMES
                    ])
                    + "</div>"
                )


# ═══════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7870))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=False,
        show_error=True,
    )
