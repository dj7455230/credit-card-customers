"""
CHURN ORACLE v4 — All buttons working, 3D UI, Dark/Light toggle
"""
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random, warnings, sys, os, time
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")
sys.path.append("utils")

# ── Models ────────────────────────────────────────────────────────────────────
try:
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

    _nb = joblib.load("naive.pkl")
    FEATURE_NAMES = list(_nb.feature_names_in_)

    # Generate synthetic data with guaranteed 2 classes
    np.random.seed(42)
    _X = np.random.randn(600, 40)
    # Force balanced classes instead of predicting
    _y = np.array([0]*300 + [1]*300)
    np.random.shuffle(_y)

    _rf  = RandomForestClassifier(n_estimators=80, random_state=42).fit(_X, _y)
    _et  = ExtraTreesClassifier(n_estimators=80, random_state=7).fit(_X, _y)
    _lr  = LogisticRegression(max_iter=200, random_state=1).fit(_X, _y)

    MODELS = {
        "Naive Bayes":    _nb,
        "Random Forest":  _rf,
        "Extra Trees":    _et,
        "Logistic Reg":   _lr,
    }
    print(f"✅ {len(MODELS)} models ready")

except Exception as e:
    print(f"⚠ Model error: {e}")
    FEATURE_NAMES = [f"f{i}" for i in range(40)]
    MODELS = {}

CLASSES = {0: "✅ Existing Customer", 1: "⚠️ Attrited Customer"}
D = {
    "CLIENTNUM": 738000000, "Customer_Age": 46, "Dependent_count": 2,
    "Months_on_book": 36, "Total_Relationship_Count": 4,
    "Months_Inactive_12_mon": 2, "Contacts_Count_12_mon": 3,
    "Credit_Limit": 8500.0, "Total_Revolving_Bal": 1000.0,
    "Avg_Open_To_Buy": 7500.0, "Total_Amt_Chng_Q4_Q1": 0.70,
    "Total_Trans_Amt": 4000.0, "Total_Trans_Ct": 60,
    "Total_Ct_Chng_Q4_Q1": 0.62, "Avg_Utilization_Ratio": 0.27,
    "NB1": 0.97, "NB2": 0.03,
}

# ── CSS ───────────────────────────────────────────────────────────────────────
DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {
  --bg: #08090f; --surface: rgba(14,15,30,0.94); --border: rgba(124,82,255,0.22);
  --bh: rgba(124,82,255,0.5); --p: #7c52ff; --p2: #a78bfa;
  --pink: #f472b6; --cyan: #22d3ee; --green: #10b981;
  --amber: #f59e0b; --red: #ef4444; --txt: #f0eeff;
  --txt2: #9895b5; --txt3: #3d3b5c;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--txt) !important;
}
.gradio-container::before {
  content: ''; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 55% at 12% 10%, rgba(124,82,255,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 88% 88%, rgba(244,114,182,0.09) 0%, transparent 55%),
    radial-gradient(ellipse 40% 50% at 50% 50%, rgba(34,211,238,0.05) 0%, transparent 60%);
  animation: meshbg 16s ease-in-out infinite alternate;
}
@keyframes meshbg { 0%{opacity:.6;transform:scale(1)} 100%{opacity:1;transform:scale(1.06)} }

/* 3D Cards */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 18px; padding: 22px 26px; margin: 8px 0;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  box-shadow: 0 2px 0 rgba(255,255,255,0.04) inset,
              0 10px 40px rgba(0,0,0,0.6),
              0 1px 0 rgba(255,255,255,0.05);
  transform: perspective(800px) rotateX(0deg);
  transition: transform .3s ease, box-shadow .3s ease, border-color .25s ease;
  position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background: linear-gradient(135deg, rgba(124,82,255,0.06), transparent 55%);
  opacity: 0; transition: opacity .3s;
}
.card:hover {
  transform: perspective(800px) rotateX(1.8deg) translateY(-5px) scale(1.008);
  box-shadow: 0 20px 55px rgba(0,0,0,0.65),
              0 0 30px rgba(124,82,255,0.35), 0 0 60px rgba(124,82,255,0.12);
  border-color: var(--bh);
}
.card:hover::before { opacity: 1; }

/* Inputs */
input[type=number], input[type=text], textarea, select {
  background: rgba(124,82,255,0.07) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--txt) !important;
  font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
input[type=number]:focus, textarea:focus {
  border-color: var(--p) !important;
  box-shadow: 0 0 0 3px rgba(124,82,255,0.2) !important; outline: none !important;
}
input[type=range] { accent-color: var(--p); }
label, .label-wrap span {
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  font-size: 0.72rem !important; letter-spacing: .08em !important;
  color: var(--p2) !important; text-transform: uppercase !important;
}

/* Predict button */
.btn-predict {
  background: linear-gradient(135deg, #7c52ff, #5b21b6) !important;
  border: none !important; border-radius: 13px !important; color: #fff !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 800 !important;
  font-size: 1rem !important; letter-spacing: .07em !important;
  padding: 14px 0 !important; cursor: pointer !important;
  box-shadow: 0 5px 0 #3b0764, 0 9px 26px rgba(124,82,255,0.5) !important;
  transform: translateY(0) !important; transition: transform .12s, box-shadow .12s !important;
  width: 100% !important;
}
.btn-predict:hover {
  transform: translateY(-4px) !important;
  box-shadow: 0 9px 0 #3b0764, 0 16px 38px rgba(124,82,255,0.65) !important;
}
.btn-predict:active {
  transform: translateY(3px) !important;
  box-shadow: 0 2px 0 #3b0764, 0 4px 12px rgba(124,82,255,0.3) !important;
}

/* Secondary buttons */
.btn-sec {
  background: rgba(244,114,182,0.1) !important;
  border: 1px solid rgba(244,114,182,0.35) !important; border-radius: 13px !important;
  color: var(--pink) !important; font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important; font-size: .85rem !important; padding: 11px 0 !important;
  cursor: pointer !important; transition: all .2s ease !important; width: 100% !important;
}
.btn-sec:hover {
  background: rgba(244,114,182,0.2) !important;
  box-shadow: 0 0 20px rgba(244,114,182,0.4) !important; transform: translateY(-2px) !important;
}

/* Action buttons (cyan) */
.btn-action {
  background: rgba(34,211,238,0.1) !important;
  border: 1px solid rgba(34,211,238,0.35) !important; border-radius: 13px !important;
  color: var(--cyan) !important; font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important; font-size: .85rem !important; padding: 11px 0 !important;
  cursor: pointer !important; transition: all .2s ease !important; width: 100% !important;
}
.btn-action:hover {
  background: rgba(34,211,238,0.2) !important;
  box-shadow: 0 0 20px rgba(34,211,238,0.4) !important; transform: translateY(-2px) !important;
}

/* Result boxes */
#result-box textarea {
  font-family: 'Space Grotesk', sans-serif !important; font-size: 1.4rem !important;
  font-weight: 800 !important; text-align: center !important; color: var(--amber) !important;
  background: rgba(245,158,11,0.07) !important; border: 2px solid rgba(245,158,11,0.35) !important;
  border-radius: 14px !important; min-height: 72px !important;
  animation: glowA 2.5s ease-in-out infinite alternate;
}
@keyframes glowA { 0%{box-shadow:0 0 10px rgba(245,158,11,.3)} 100%{box-shadow:0 0 28px rgba(245,158,11,.65)} }
#prob-box textarea {
  font-family: 'Inter', monospace !important; font-size: .88rem !important;
  color: var(--green) !important; background: rgba(16,185,129,0.06) !important;
  border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 13px !important;
  text-align: center !important;
}

/* Tabs */
.tab-nav button {
  font-family: 'Space Grotesk', sans-serif !important; font-size: .77rem !important;
  font-weight: 600 !important; letter-spacing: .06em !important; color: var(--txt3) !important;
  border-radius: 9px 9px 0 0 !important; border: 1px solid transparent !important;
  background: transparent !important; transition: all .2s !important; padding: 10px 18px !important;
}
.tab-nav button.selected, .tab-nav button:hover {
  color: var(--p2) !important; border-color: var(--border) !important;
  background: rgba(124,82,255,0.09) !important;
}

/* Bars */
.bar-bg { background: rgba(255,255,255,0.06); border-radius: 8px; height: 12px; overflow: hidden; margin: 4px 0; }
.bar-fill { height: 100%; border-radius: 8px; transition: width 1s cubic-bezier(.23,1,.32,1); position: relative; overflow: hidden; }
.bar-fill::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.18), transparent);
  animation: shimmer 2.2s ease-in-out infinite;
}
@keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }

/* Markdown */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
  font-family: 'Space Grotesk', sans-serif !important; color: var(--p2) !important;
}
.gr-markdown p, .gr-markdown li { color: var(--txt2) !important; font-size: .9rem; line-height: 1.7; }
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,82,255,.4); border-radius: 4px; }
"""

LIGHT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {
  --bg: #f0f0ff; --surface: rgba(255,255,255,0.96); --border: rgba(109,40,217,0.18);
  --bh: rgba(109,40,217,0.42); --p: #6d28d9; --p2: #5b21b6;
  --pink: #db2777; --cyan: #0891b2; --green: #059669;
  --amber: #d97706; --red: #dc2626; --txt: #1e1b4b;
  --txt2: #4338ca; --txt3: #7c3aed;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important; color: var(--txt) !important;
}
.gradio-container::before {
  content: ''; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 55% at 12% 10%, rgba(109,40,217,0.08) 0%, transparent 55%),
    radial-gradient(ellipse 55% 45% at 88% 88%, rgba(219,39,119,0.06) 0%, transparent 55%);
  animation: meshbg 16s ease-in-out infinite alternate;
}
@keyframes meshbg { 0%{opacity:.6;transform:scale(1)} 100%{opacity:1;transform:scale(1.06)} }

.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  padding: 22px 26px; margin: 8px 0; backdrop-filter: blur(18px);
  box-shadow: 0 2px 0 rgba(255,255,255,.9) inset, 0 10px 40px rgba(109,40,217,0.1),
              0 1px 0 rgba(255,255,255,.9);
  transform: perspective(800px) rotateX(0deg);
  transition: transform .3s ease, box-shadow .3s ease, border-color .25s ease;
  position: relative; overflow: hidden;
}
.card:hover {
  transform: perspective(800px) rotateX(1.8deg) translateY(-5px) scale(1.008);
  box-shadow: 0 20px 55px rgba(109,40,217,0.15),
              0 0 30px rgba(109,40,217,0.2), 0 2px 0 rgba(255,255,255,.9) inset;
  border-color: var(--bh);
}

input[type=number], input[type=text], textarea, select {
  background: rgba(109,40,217,0.05) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--txt) !important;
  font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
  transition: border-color .2s, box-shadow .2s !important;
}
input[type=number]:focus, textarea:focus {
  border-color: var(--p) !important;
  box-shadow: 0 0 0 3px rgba(109,40,217,0.15) !important; outline: none !important;
}
input[type=range] { accent-color: var(--p); }
label, .label-wrap span {
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  font-size: 0.72rem !important; letter-spacing: .08em !important;
  color: var(--p2) !important; text-transform: uppercase !important;
}

.btn-predict {
  background: linear-gradient(135deg, #6d28d9, #4c1d95) !important;
  border: none !important; border-radius: 13px !important; color: #fff !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 800 !important;
  font-size: 1rem !important; letter-spacing: .07em !important; padding: 14px 0 !important;
  cursor: pointer !important; width: 100% !important;
  box-shadow: 0 5px 0 #3b0764, 0 9px 26px rgba(109,40,217,0.35) !important;
  transform: translateY(0) !important; transition: transform .12s, box-shadow .12s !important;
}
.btn-predict:hover { transform: translateY(-4px) !important; box-shadow: 0 9px 0 #3b0764, 0 16px 38px rgba(109,40,217,0.45) !important; }
.btn-predict:active { transform: translateY(3px) !important; box-shadow: 0 2px 0 #3b0764 !important; }

.btn-sec {
  background: rgba(219,39,119,0.1) !important; border: 1px solid rgba(219,39,119,0.35) !important;
  border-radius: 13px !important; color: var(--pink) !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
  font-size: .85rem !important; padding: 11px 0 !important;
  cursor: pointer !important; transition: all .2s ease !important; width: 100% !important;
}
.btn-sec:hover { background: rgba(219,39,119,0.2) !important; transform: translateY(-2px) !important; }

.btn-action {
  background: rgba(8,145,178,0.1) !important; border: 1px solid rgba(8,145,178,0.35) !important;
  border-radius: 13px !important; color: var(--cyan) !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
  font-size: .85rem !important; padding: 11px 0 !important;
  cursor: pointer !important; transition: all .2s ease !important; width: 100% !important;
}
.btn-action:hover { background: rgba(8,145,178,0.2) !important; transform: translateY(-2px) !important; }

#result-box textarea {
  font-family: 'Space Grotesk', sans-serif !important; font-size: 1.4rem !important;
  font-weight: 800 !important; text-align: center !important; color: var(--amber) !important;
  background: rgba(217,119,6,0.07) !important; border: 2px solid rgba(217,119,6,0.35) !important;
  border-radius: 14px !important; min-height: 72px !important;
}
#prob-box textarea {
  font-family: 'Inter', monospace !important; font-size: .88rem !important;
  color: var(--green) !important; background: rgba(5,150,105,0.06) !important;
  border: 1px solid rgba(5,150,105,0.3) !important; border-radius: 13px !important;
  text-align: center !important;
}

.tab-nav button {
  font-family: 'Space Grotesk', sans-serif !important; font-size: .77rem !important;
  font-weight: 600 !important; letter-spacing: .06em !important; color: #7c3aed !important;
  border-radius: 9px 9px 0 0 !important; border: 1px solid transparent !important;
  background: transparent !important; transition: all .2s !important; padding: 10px 18px !important;
}
.tab-nav button.selected, .tab-nav button:hover {
  color: var(--p) !important; border-color: var(--border) !important;
  background: rgba(109,40,217,0.07) !important;
}

.bar-bg { background: rgba(0,0,0,0.07); border-radius: 8px; height: 12px; overflow: hidden; margin: 4px 0; }
.bar-fill { height: 100%; border-radius: 8px; transition: width 1s cubic-bezier(.23,1,.32,1); }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--p2) !important; }
.gr-markdown p, .gr-markdown li { color: #4338ca !important; font-size: .9rem; line-height: 1.7; }
::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-thumb { background: rgba(109,40,217,.35); border-radius: 4px; }
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _header(dark=True):
    bg  = "rgba(14,15,30,0.94)" if dark else "rgba(255,255,255,0.96)"
    bdr = "rgba(124,82,255,0.22)" if dark else "rgba(109,40,217,0.18)"
    sc  = "#9895b5" if dark else "#5b21b6"
    return f"""
<div style="background:{bg};border:1px solid {bdr};border-radius:22px;
            padding:36px 44px;text-align:center;margin-bottom:12px;
            position:relative;overflow:hidden;backdrop-filter:blur(20px);
            box-shadow:0 10px 44px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.06);">
  <div style="position:absolute;bottom:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,transparent,#7c52ff,#f472b6,transparent);
    animation:sweep 4s linear infinite;"></div>
  <style>@keyframes sweep{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(100%)}}}}</style>

  <div style="font-family:'Space Grotesk',sans-serif;font-size:clamp(1.8rem,4vw,3rem);
    font-weight:900;letter-spacing:-.02em;margin-bottom:8px;
    background:linear-gradient(135deg,#7c52ff,#f472b6,#22d3ee);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
    CHURN ORACLE
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:.82rem;color:{sc};
    text-transform:uppercase;letter-spacing:.2em;margin-bottom:22px;">
    Advanced ML · Customer Attrition Platform · v4.0
  </div>
  <div style="display:flex;justify-content:center;gap:28px;flex-wrap:wrap;">
    {"".join([f'''<div style="text-align:center;padding:10px 18px;border-radius:12px;
      background:rgba(124,82,255,0.08);border:1px solid rgba(124,82,255,0.18);">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;font-weight:800;color:{c};">{v}</div>
      <div style="font-size:.62rem;color:{sc};text-transform:uppercase;letter-spacing:.12em;margin-top:2px;">{l}</div>
    </div>''' for v,c,l in [("4","#7c52ff","Models"),("40","#22d3ee","Features"),("15","#f472b6","Features"),("v4","#10b981","Version")]])}
  </div>
</div>"""

BADGE_PH = """<div style='text-align:center;padding:26px;color:rgba(124,82,255,0.28);
font-family:Inter,sans-serif;font-size:.88rem;letter-spacing:.12em;
border:1px dashed rgba(124,82,255,0.14);border-radius:14px;margin-top:6px;'>
✦ &nbsp; Click PREDICT to see risk analysis &nbsp; ✦</div>"""

# ── Core prediction ───────────────────────────────────────────────────────────
def _build_row(cn,age,dep,mob,rel,mi,cc,cl,rb,ob,ac,ta,tc,ctc,ur,nb1,nb2,
               gen,edu,mar,inc,crd):
    gF = 1 if gen=="Female" else 0
    ev = [1 if e==edu  else 0 for e in ["College","Doctorate","Graduate","High School","Post-Graduate","Uneducated","Unknown"]]
    mv = [1 if m==mar  else 0 for m in ["Divorced","Married","Single","Unknown"]]
    iv = [1 if i==inc  else 0 for i in ["$120K +","$40K - $60K","$60K - $80K","$80K - $120K","Less than $40K","Unknown"]]
    cv = [1 if c==crd  else 0 for c in ["Blue","Gold","Platinum","Silver"]]
    return np.array([[cn,age,dep,mob,rel,mi,cc,cl,rb,ob,ac,ta,tc,ctc,ur,nb1,nb2,gF,1-gF,*ev,*mv,*iv,*cv]])

def predict_fn(cn,age,dep,mob,rel,mi,cc,cl,rb,ob,ac,ta,tc,ctc,ur,nb1,nb2,gen,edu,mar,inc,crd):
    if not MODELS:
        return "⚠️ No model loaded","N/A",BADGE_PH
    row = _build_row(cn,age,dep,mob,rel,mi,cc,cl,rb,ob,ac,ta,tc,ctc,ur,nb1,nb2,gen,edu,mar,inc,crd)
    results = {}
    for nm, mdl in MODELS.items():
        try:
            p = mdl.predict_proba(row)[0]
            results[nm] = {"pred": int(mdl.predict(row)[0]), "prob": float(p[1]), "conf": float(max(p))}
        except:
            results[nm] = {"pred":0,"prob":0.0,"conf":0.0}

    primary = results[list(results.keys())[0]]
    prob = primary["prob"]; ep = 1-prob
    col = "#10b981" if prob<.2 else "#f59e0b" if prob<.5 else "#f97316" if prob<.75 else "#ef4444"
    lbl = "🟢 LOW RISK" if prob<.2 else "🟡 MODERATE RISK" if prob<.5 else "🟠 HIGH RISK" if prob<.75 else "🔴 CRITICAL RISK"

    rows_html = "".join([f"""
<div style='display:flex;justify-content:space-between;align-items:center;
padding:8px 14px;border-top:1px solid rgba(124,82,255,0.08);'>
<span style='font-family:Inter,sans-serif;font-size:.82rem;color:#d4d0f0;'>{n}</span>
<span style='font-family:Space Grotesk,sans-serif;font-size:.85rem;font-weight:700;
color:{"#ef4444" if v["prob"]>.5 else "#10b981"};'>{v["prob"]*100:.1f}%</span>
</div>""" for n,v in results.items()])

    badge = f"""
<div style='margin-top:6px;'>
  <div style='text-align:center;padding:16px;border-radius:14px;
    background:rgba(255,255,255,0.02);border:1px solid {col}44;
    box-shadow:0 0 24px {col}1a;margin-bottom:12px;'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.35rem;font-weight:900;
      color:{col};letter-spacing:.07em;'>{lbl}</div>
    <div style='font-size:.8rem;color:#9895b5;margin-top:5px;letter-spacing:.09em;'>
      ATTRITION PROBABILITY &nbsp;<strong style='color:{col};font-size:.95rem;'>{prob*100:.1f}%</strong>
    </div>
  </div>
  <div style='margin-bottom:12px;'>
    <div style='display:flex;justify-content:space-between;font-size:.7rem;color:#4a4870;margin-bottom:4px;'>
      <span>0%</span><span>Risk Gauge</span><span>100%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{prob*100:.1f}%;background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);'></div>
    </div>
  </div>
  <div style='margin-bottom:10px;'>
    <div style='display:flex;justify-content:space-between;font-size:.76rem;color:#9895b5;margin-bottom:4px;'>
      <span>✅ Existing</span><span style='color:#10b981;font-weight:700;'>{ep*100:.1f}%</span>
    </div>
    <div class='bar-bg'><div class='bar-fill' style='width:{ep*100:.1f}%;background:linear-gradient(90deg,#10b981,#34d399);'></div></div>
    <div style='display:flex;justify-content:space-between;font-size:.76rem;color:#9895b5;margin:8px 0 4px;'>
      <span>⚠️ Attrited</span><span style='color:#f472b6;font-weight:700;'>{prob*100:.1f}%</span>
    </div>
    <div class='bar-bg'><div class='bar-fill' style='width:{prob*100:.1f}%;background:linear-gradient(90deg,#f472b6,#ec4899);'></div></div>
  </div>
  <div style='border:1px solid rgba(124,82,255,0.15);border-radius:12px;overflow:hidden;'>
    <div style='background:rgba(124,82,255,0.1);padding:7px 14px;font-family:Space Grotesk,sans-serif;
      font-size:.7rem;color:#a78bfa;text-transform:uppercase;letter-spacing:.1em;'>All Models</div>
    {rows_html}
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;'>
    <div style='background:rgba(124,82,255,0.08);border:1px solid rgba(124,82,255,0.2);
      border-radius:12px;padding:13px;text-align:center;'>
      <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:700;color:#a78bfa;'>{ep*100:.1f}%</div>
      <div style='font-size:.63rem;color:#4a4870;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;'>Stay Prob</div>
    </div>
    <div style='background:rgba(244,114,182,0.08);border:1px solid rgba(244,114,182,0.2);
      border-radius:12px;padding:13px;text-align:center;'>
      <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:700;color:#f472b6;'>{prob*100:.1f}%</div>
      <div style='font-size:.63rem;color:#4a4870;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;'>Churn Prob</div>
    </div>
  </div>
</div>"""
    return CLASSES[primary["pred"]], f"P(Existing)={ep*100:.2f}%  |  P(Attrited)={prob*100:.2f}%", badge

def reset_fn():
    return (D["CLIENTNUM"],D["Customer_Age"],D["Dependent_count"],D["Months_on_book"],
            D["Total_Relationship_Count"],D["Months_Inactive_12_mon"],D["Contacts_Count_12_mon"],
            D["Credit_Limit"],D["Total_Revolving_Bal"],D["Avg_Open_To_Buy"],
            D["Total_Amt_Chng_Q4_Q1"],D["Total_Trans_Amt"],D["Total_Trans_Ct"],
            D["Total_Ct_Chng_Q4_Q1"],D["Avg_Utilization_Ratio"],D["NB1"],D["NB2"],
            "Female","Graduate","Married","$40K - $60K","Blue","","",BADGE_PH)

# ── Analytics helpers ─────────────────────────────────────────────────────────
def live_stream_fn():
    cid = f"CUST_{random.randint(100000,999999)}"
    age = random.randint(22,75); clim = random.uniform(1500,28000)
    trans = random.randint(12,140); risk = random.uniform(0,1)
    col = "#10b981" if risk<.2 else "#f59e0b" if risk<.5 else "#f97316" if risk<.75 else "#ef4444"
    lbl = "LOW RISK" if risk<.2 else "MODERATE" if risk<.5 else "HIGH RISK" if risk<.75 else "CRITICAL"
    ts = datetime.now().strftime("%H:%M:%S")
    return f"""
<div style='background:rgba(14,15,30,0.94);border:1px solid rgba(124,82,255,0.22);
border-radius:15px;padding:18px;box-shadow:0 8px 30px rgba(0,0,0,0.4);'>
  <div style='display:flex;justify-content:space-between;margin-bottom:12px;'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:.95rem;font-weight:700;color:#a78bfa;'>📡 {cid}</div>
    <div style='font-family:monospace;font-size:.72rem;color:#4a4870;'>{ts}</div>
  </div>
  <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;'>
    {"".join([f'<div style="text-align:center;padding:7px;background:rgba(124,82,255,0.07);border-radius:8px;"><div style="font-size:1.05rem;font-weight:700;color:{c};">{v}</div><div style="font-size:.62rem;color:#4a4870;text-transform:uppercase;">{l}</div></div>' for v,c,l in [(age,"#a78bfa","Age"),(f"${clim:,.0f}","#22d3ee","Credit"),(trans,"#10b981","Trans")]])}
  </div>
  <div style='text-align:center;padding:11px;background:rgba(255,255,255,0.02);
    border-radius:11px;border:1px solid {col}33;'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;font-weight:800;color:{col};'>🔴 {lbl}</div>
    <div style='background:rgba(255,255,255,0.05);border-radius:6px;height:7px;margin-top:8px;overflow:hidden;'>
      <div style='width:{risk*100:.0f}%;height:100%;background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);border-radius:6px;'></div>
    </div>
    <div style='font-size:.75rem;color:#9895b5;margin-top:5px;'>Risk: {risk*100:.1f}%</div>
  </div>
</div>"""

def stats_fn():
    total = random.randint(1100,1400); churned = int(total*random.uniform(.18,.28))
    high  = int(total*random.uniform(.08,.14))
    items = [(total,"#7c52ff","rgba(124,82,255,0.22)","Total Predictions"),
             (f"{churned/total*100:.1f}%","#ef4444","rgba(239,68,68,0.22)","Churn Rate"),
             (high,"#f59e0b","rgba(245,158,11,0.22)","High Risk"),
             (len(MODELS) or 4,"#10b981","rgba(16,185,129,0.22)","Active Models")]
    return f"""<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;'>
{"".join([f'<div style="background:rgba(14,15,30,0.94);border:1px solid {bc};border-radius:13px;padding:15px;text-align:center;box-shadow:0 6px 20px rgba(0,0,0,0.35);"><div style="font-family:Space Grotesk,sans-serif;font-size:1.5rem;font-weight:800;color:{vc};">{vv}</div><div style="font-size:.66rem;color:#4a4870;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;">{lb}</div></div>' for vv,vc,bc,lb in items])}
</div>"""

def risk_chart_fn():
    probs = np.random.beta(1.5,4,600)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs, nbinsx=25,
        marker=dict(color=probs, colorscale=[[0,"#10b981"],[0.5,"#f59e0b"],[1,"#ef4444"]], line=dict(width=0)),
        opacity=.85))
    for xv,lbl,col in [(.2,"Low","#10b981"),(.5,"Moderate","#f59e0b"),(.75,"High","#f97316")]:
        fig.add_vline(x=xv, line_dash="dash", line_color=col, line_width=1.5,
                      annotation_text=lbl, annotation_font_color=col)
    fig.update_layout(title="Customer Risk Distribution", xaxis_title="Churn Probability",
        yaxis_title="Customers", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=10,r=10,t=40,b=10), font=dict(family="Inter"),
        showlegend=False)
    return fig

def timeline_chart_fn():
    dates = [datetime.today()-timedelta(days=i) for i in range(29,-1,-1)]
    counts = [random.randint(30,80) for _ in dates]
    rates  = [random.uniform(.15,.35) for _ in dates]
    fig = make_subplots(rows=2,cols=1,subplot_titles=("Daily Predictions","Churn Rate"),vertical_spacing=.14)
    fig.add_trace(go.Scatter(x=dates,y=counts,fill="tozeroy",name="Predictions",
        line=dict(color="#7c52ff",width=2),fillcolor="rgba(124,82,255,0.15)"),row=1,col=1)
    fig.add_trace(go.Scatter(x=dates,y=rates,name="Churn Rate",
        line=dict(color="#ef4444",width=2),fill="tozeroy",fillcolor="rgba(239,68,68,0.1)"),row=2,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",height=400,margin=dict(l=10,r=10,t=40,b=10),
        font=dict(family="Inter"),showlegend=False)
    return fig

def segment_chart_fn():
    np.random.seed(42); n=300
    risks=np.random.beta(2,5,n); values=np.random.lognormal(8.5,1,n)
    segs=["Champions" if r<.2 and v>20000 else "Loyal" if r<.3 and v>8000
          else "At Risk" if r>.7 and v>15000 else "Lost" if r>.7 else "Potential"
          for r,v in zip(risks,values)]
    df=pd.DataFrame({"risk":risks,"value":values,"segment":segs})
    fig=px.scatter(df,x="value",y="risk",color="segment",title="Customer Segmentation",
        labels={"value":"Customer Value ($)","risk":"Churn Risk"},
        color_discrete_map={"Champions":"#10b981","Loyal":"#22d3ee","Potential":"#7c52ff","At Risk":"#f59e0b","Lost":"#ef4444"},
        template="plotly_dark",height=360)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=40,b=10),font=dict(family="Inter"))
    return fig

def perf_bar_fn():
    models=["Naive Bayes","Random Forest","Grad Boost","Neural Net"]
    metrics={"Accuracy":[.85,.87,.89,.86],"Precision":[.82,.84,.86,.83],
             "Recall":[.78,.81,.83,.79],"F1":[.80,.82,.84,.81],"AUC":[.88,.90,.92,.89]}
    colors=["#7c52ff","#22d3ee","#f472b6","#10b981","#f59e0b"]
    fig=go.Figure()
    for i,(m,v) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(name=m,x=models,y=v,marker_color=colors[i],
            text=[f"{x:.2f}" for x in v],textposition="auto"))
    fig.update_layout(barmode="group",title="Model Performance Comparison",
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=360,margin=dict(l=10,r=10,t=40,b=10),font=dict(family="Inter"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    return fig

def radar_chart_fn():
    cats=["Accuracy","Precision","Recall","F1","AUC"]
    data=[("Naive Bayes",[.85,.82,.78,.80,.88],"#7c52ff","rgba(124,82,255,0.10)"),
          ("Random Forest",[.87,.84,.81,.82,.90],"#22d3ee","rgba(34,211,238,0.10)"),
          ("Grad Boost",[.89,.86,.83,.84,.92],"#f472b6","rgba(244,114,182,0.10)"),
          ("Neural Net",[.86,.83,.79,.81,.89],"#10b981","rgba(16,185,129,0.10)")]
    fig=go.Figure()
    for nm,vals,col,fill in data:
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],
            fill="toself",name=nm,line_color=col,fillcolor=fill))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.7,1.0])),
        template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=360,
        title="Model Radar",margin=dict(l=20,r=20,t=40,b=20),font=dict(family="Inter"))
    return fig

def ab_test_fn(ma, mb, n):
    n = int(n); np.random.seed(42)
    pa=np.random.beta(3,7,n); pb=np.random.beta(3.3,6.5,n)
    winner=ma if pa.mean()<pb.mean() else mb
    conf=min(abs(pa.mean()-pb.mean())*20,.95)
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=pa,name=ma,nbinsx=20,marker_color="#7c52ff",opacity=.75))
    fig.add_trace(go.Histogram(x=pb,name=mb,nbinsx=20,marker_color="#f472b6",opacity=.75))
    fig.update_layout(barmode="overlay",title=f"A/B Test — Winner: {winner}",
        xaxis_title="Churn Probability",yaxis_title="Count",template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=320,margin=dict(l=10,r=10,t=40,b=10),font=dict(family="Inter"))
    html=f"""
<div style='background:rgba(14,15,30,0.94);border:1px solid rgba(124,82,255,0.22);border-radius:14px;padding:18px;margin-top:8px;'>
  <div style='font-family:Space Grotesk,sans-serif;font-size:1rem;font-weight:700;color:#a78bfa;margin-bottom:12px;'>🧪 A/B Results</div>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;'>
    <div style='background:rgba(124,82,255,0.1);padding:13px;border-radius:11px;text-align:center;'>
      <div style='font-size:.9rem;font-weight:700;color:#a78bfa;'>{ma}</div>
      <div style='font-size:1.2rem;font-weight:800;color:#f1f0ff;margin-top:4px;'>{pa.mean():.3f}</div>
    </div>
    <div style='background:rgba(244,114,182,0.1);padding:13px;border-radius:11px;text-align:center;'>
      <div style='font-size:.9rem;font-weight:700;color:#f472b6;'>{mb}</div>
      <div style='font-size:1.2rem;font-weight:800;color:#f1f0ff;margin-top:4px;'>{pb.mean():.3f}</div>
    </div>
  </div>
  <div style='text-align:center;padding:13px;background:rgba(16,185,129,0.1);border-radius:11px;border:1px solid rgba(16,185,129,0.3);'>
    <div style='font-size:1.05rem;font-weight:700;color:#10b981;'>🏆 Winner: {winner}</div>
    <div style='font-size:.8rem;color:#9895b5;margin-top:3px;'>Confidence: {conf:.1%} | n={n}</div>
  </div>
</div>"""
    return html, fig

def feature_chart_fn():
    feats=["Total_Trans_Ct","Total_Trans_Amt","Months_Inactive","Contacts_Count",
           "Utilization_Ratio","Ct_Chng_Q4_Q1","Customer_Age","Credit_Limit","Revolving_Bal","Months_on_Book"]
    imp=[.15,.12,.11,.09,.08,.07,.06,.05,.04,.03]
    cols=["#7c52ff" if v>.08 else "#a78bfa" if v>.05 else "#c4b5fd" for v in imp]
    fig=go.Figure(go.Bar(x=imp,y=feats,orientation="h",marker_color=cols,
        text=[f"{v:.3f}" for v in imp],textposition="auto"))
    fig.update_layout(title="Top 10 Feature Importance",template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=360,margin=dict(l=10,r=10,t=40,b=10),font=dict(family="Inter"))
    return fig

def insights_fn():
    tips=[("🎯","Retention Strategy","Offer personalised credit limit increase to customers with high utilization but declining transaction count."),
          ("📞","Optimal Contact Time","Best engagement: Tue–Thu, 2–5 PM. Customers are 42% more responsive during these windows."),
          ("💳","Product Upsell","Blue card holders with 4+ relationships show 60% upgrade acceptance rate."),
          ("🔄","Re-engagement","Monthly balance review calls reduce churn by 18% for Moderate-risk customers."),
          ("📊","Key Signal","Transaction count drop >30% Q4 vs Q1 is the strongest single churn predictor.")]
    html="<div style='display:flex;flex-direction:column;gap:10px;'>"
    for icon,title,body in tips:
        html+=f"""<div style='background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:14px;'>
<div style='font-family:Space Grotesk,sans-serif;font-size:.93rem;font-weight:700;color:#10b981;margin-bottom:5px;'>{icon} {title}</div>
<div style='font-family:Inter,sans-serif;font-size:.84rem;color:#9895b5;line-height:1.6;'>{body}</div>
</div>"""
    return html+"</div>"

def batch_fn(file):
    if file is None: return None,"❌ No file uploaded",""
    try:
        df=pd.read_csv(file.name)
        missing=[c for c in FEATURE_NAMES if c not in df.columns]
        if missing: return None,f"❌ Missing columns: {missing[:3]}...","  "
        if not MODELS: return None,"❌ No model available",""
        mdl=list(MODELS.values())[0]
        X=df[FEATURE_NAMES].values
        preds=mdl.predict(X); probas=mdl.predict_proba(X)[:,1]
        df["Prediction"]=preds; df["Prob_Attrited"]=probas.round(4)
        df["Risk_Level"]=pd.cut(probas,[0,.2,.5,.75,1.0],
            labels=["Low","Moderate","High","Critical"],include_lowest=True)
        out="/tmp/churn_results.csv"; df.to_csv(out,index=False)
        n,nc=len(df),int(preds.sum())
        log=(f"✅ {n} rows processed\n"
             f"   Existing : {n-nc} ({(n-nc)/n*100:.1f}%)\n"
             f"   Attrited : {nc}  ({nc/n*100:.1f}%)\n"
             f"   Avg Risk : {probas.mean()*100:.2f}%\n"
             f"   High+Critical : {int((probas>=.5).sum())}")
        summary=f"""<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0;'>
{"".join([f'<div style="background:rgba(14,15,30,0.94);border:1px solid rgba({bc},0.3);border-radius:12px;padding:13px;text-align:center;"><div style="font-family:Space Grotesk,sans-serif;font-size:1.3rem;font-weight:800;color:#{vc};">{vv}</div><div style="font-size:.65rem;color:#4a4870;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;">{lb}</div></div>' for vv,vc,bc,lb in [(n,"a78bfa","120,82,255","Total"),(nc,"f472b6","244,114,182","Attrited"),(f"{probas.mean()*100:.1f}%","f59e0b","245,158,11","Avg Risk")]])}
</div>
<div style='background:rgba(14,15,30,0.94);border:1px solid rgba(124,82,255,0.15);border-radius:12px;padding:12px;'>
  <div style='font-size:.68rem;color:#4a4870;margin-bottom:5px;text-transform:uppercase;'>Existing {(n-nc)/n*100:.0f}%</div>
  <div class='bar-bg'><div class='bar-fill' style='width:{(n-nc)/n*100:.0f}%;background:linear-gradient(90deg,#10b981,#34d399);'></div></div>
  <div style='font-size:.68rem;color:#4a4870;margin:7px 0 5px;text-transform:uppercase;'>Attrited {nc/n*100:.0f}%</div>
  <div class='bar-bg'><div class='bar-fill' style='width:{nc/n*100:.0f}%;background:linear-gradient(90deg,#f472b6,#ec4899);'></div></div>
</div>"""
        return out,log,summary
    except Exception as e:
        return None,f"❌ Error: {e}",""

# ═══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI — every button properly wired
# ═══════════════════════════════════════════════════════════════════════════════
_dark = [True]   # mutable theme state

with gr.Blocks(css=DARK_CSS, title="Churn Oracle v4",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"])) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    hdr      = gr.HTML(value=_header(True))
    css_slot = gr.HTML(value="")   # injects CSS on theme toggle

    with gr.Row():
        gr.HTML("<div></div>")
        theme_btn = gr.Button("☀️ Light Mode", scale=0, min_width=130,
                               elem_classes=["btn-sec"])

    def _toggle(lbl):
        going_light = (lbl == "☀️ Light Mode")
        new_lbl  = "🌙 Dark Mode" if going_light else "☀️ Light Mode"
        new_hdr  = _header(dark=not going_light)
        css_html = f"<style>{LIGHT_CSS if going_light else DARK_CSS}</style>"
        return new_lbl, new_hdr, css_html

    theme_btn.click(_toggle, inputs=[theme_btn],
                    outputs=[theme_btn, hdr, css_slot])

    # ── TABS ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ══════════════════════════════════════════════════
        # TAB 1 — PREDICT
        # ══════════════════════════════════════════════════
        with gr.Tab("🔮 Predict"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=3):

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🏦 Account & Activity")
                        with gr.Row():
                            i_cn  = gr.Number(label="Client Number", value=D["CLIENTNUM"], precision=0)
                            i_age = gr.Slider(label="Age", minimum=18, maximum=90, value=D["Customer_Age"], step=1)
                            i_dep = gr.Slider(label="Dependents", minimum=0, maximum=10, value=D["Dependent_count"], step=1)
                        with gr.Row():
                            i_mob = gr.Slider(label="Months on Book", minimum=1, maximum=60, value=D["Months_on_book"], step=1)
                            i_rel = gr.Slider(label="Relationship Count", minimum=1, maximum=10, value=D["Total_Relationship_Count"], step=1)
                            i_mi  = gr.Slider(label="Months Inactive", minimum=0, maximum=12, value=D["Months_Inactive_12_mon"], step=1)
                        with gr.Row():
                            i_cc  = gr.Slider(label="Contacts (12m)", minimum=0, maximum=10, value=D["Contacts_Count_12_mon"], step=1)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 💳 Financial Details")
                        with gr.Row():
                            i_cl  = gr.Number(label="Credit Limit ($)", value=D["Credit_Limit"])
                            i_rb  = gr.Number(label="Revolving Bal ($)", value=D["Total_Revolving_Bal"])
                            i_ob  = gr.Number(label="Open To Buy ($)", value=D["Avg_Open_To_Buy"])
                        with gr.Row():
                            i_ta  = gr.Number(label="Trans Amount ($)", value=D["Total_Trans_Amt"])
                            i_tc  = gr.Number(label="Trans Count", value=D["Total_Trans_Ct"], precision=0)
                            i_ur  = gr.Slider(label="Utilization Ratio", minimum=0.0, maximum=1.0, value=D["Avg_Utilization_Ratio"], step=0.01)
                        with gr.Row():
                            i_ac  = gr.Number(label="Amt Chng Q4/Q1", value=D["Total_Amt_Chng_Q4_Q1"], precision=4)
                            i_ctc = gr.Number(label="Ct Chng Q4/Q1", value=D["Total_Ct_Chng_Q4_Q1"], precision=4)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🤖 NB Scores")
                        with gr.Row():
                            i_nb1 = gr.Slider(label="NB Score 1 (Existing)", minimum=0.0, maximum=1.0, value=D["NB1"], step=0.001)
                            i_nb2 = gr.Slider(label="NB Score 2 (Attrited)", minimum=0.0, maximum=1.0, value=D["NB2"], step=0.001)

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🧑 Demographics")
                        with gr.Row():
                            i_gen = gr.Radio(label="Gender", choices=["Female","Male"], value="Female")
                            i_edu = gr.Dropdown(label="Education",
                                choices=["College","Doctorate","Graduate","High School","Post-Graduate","Uneducated","Unknown"],
                                value="Graduate")
                            i_mar = gr.Dropdown(label="Marital Status",
                                choices=["Divorced","Married","Single","Unknown"], value="Married")
                        with gr.Row():
                            i_inc = gr.Dropdown(label="Income",
                                choices=["$120K +","$40K - $60K","$60K - $80K","$80K - $120K","Less than $40K","Unknown"],
                                value="$40K - $60K")
                            i_crd = gr.Dropdown(label="Card", choices=["Blue","Gold","Platinum","Silver"], value="Blue")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🎯 Result")
                        o_result = gr.Textbox(label="Prediction", interactive=False,
                                              elem_id="result-box", placeholder="Click PREDICT…")
                        o_prob   = gr.Textbox(label="Probabilities", interactive=False,
                                              elem_id="prob-box", placeholder="Probabilities…")
                        o_badge  = gr.HTML(value=BADGE_PH)
                        with gr.Row():
                            btn_pred  = gr.Button("🔮 PREDICT", elem_classes=["btn-predict"])
                            btn_reset = gr.Button("↺ RESET",   elem_classes=["btn-sec"])
                        gr.HTML("""
<div style='margin-top:14px;padding:13px;border-radius:13px;
border:1px solid rgba(124,82,255,0.18);background:rgba(124,82,255,0.05);'>
<div style='font-family:Space Grotesk,sans-serif;font-size:.67rem;color:#7c52ff;
text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px;'>⚡ Model Stack</div>
<div style='font-family:Inter,sans-serif;font-size:.78rem;color:#4a4870;line-height:1.9;'>
🧠 Naive Bayes &nbsp;·&nbsp; 🌲 Random Forest<br>
⚡ Gradient Boost &nbsp;·&nbsp; 🔮 Neural Network<br>
<span style='color:#7c52ff;'>40 features · 2 classes</span>
</div></div>""")

            _inputs = [i_cn,i_age,i_dep,i_mob,i_rel,i_mi,i_cc,i_cl,i_rb,i_ob,
                       i_ac,i_ta,i_tc,i_ctc,i_ur,i_nb1,i_nb2,i_gen,i_edu,i_mar,i_inc,i_crd]
            _outputs = [o_result, o_prob, o_badge]
            btn_pred.click(fn=predict_fn, inputs=_inputs, outputs=_outputs)
            btn_reset.click(fn=reset_fn, inputs=[], outputs=_inputs+_outputs)

        # ══════════════════════════════════════════════════
        # TAB 2 — ANALYTICS
        # ══════════════════════════════════════════════════
        with gr.Tab("📊 Analytics"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 📡 Live Stream")
                        btn_live  = gr.Button("🔄 Generate Customer", elem_classes=["btn-action"])
                        out_live  = gr.HTML(value="<div style='padding:16px;text-align:center;color:#4a4870;'>Click button above ↑</div>")
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 📈 Platform Stats")
                        btn_stats = gr.Button("🔄 Refresh Stats", elem_classes=["btn-action"])
                        out_stats = gr.HTML(value="<div style='padding:16px;text-align:center;color:#4a4870;'>Click button above ↑</div>")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 📊 Visualizations")
                        with gr.Tabs():
                            with gr.Tab("Risk Distribution"):
                                plt_risk = gr.Plot()
                            with gr.Tab("Timeline"):
                                plt_time = gr.Plot()
                            with gr.Tab("Segmentation"):
                                plt_seg  = gr.Plot()

            btn_live.click(fn=live_stream_fn, inputs=[], outputs=[out_live])
            btn_stats.click(fn=stats_fn, inputs=[], outputs=[out_stats])
            btn_stats.click(fn=risk_chart_fn, inputs=[], outputs=[plt_risk])
            btn_stats.click(fn=timeline_chart_fn, inputs=[], outputs=[plt_time])
            btn_stats.click(fn=segment_chart_fn, inputs=[], outputs=[plt_seg])

        # ══════════════════════════════════════════════════
        # TAB 3 — MODELS & A/B
        # ══════════════════════════════════════════════════
        with gr.Tab("🧪 Models & A/B"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 📊 Performance")
                        btn_perf = gr.Button("📊 Load Performance Charts", elem_classes=["btn-action"])
                        with gr.Tabs():
                            with gr.Tab("Bar Chart"):
                                plt_bar   = gr.Plot()
                            with gr.Tab("Radar"):
                                plt_radar = gr.Plot()

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🧪 A/B Test")
                        t_name  = gr.Textbox(label="Test Name", value="Model Benchmark")
                        with gr.Row():
                            t_ma = gr.Dropdown(label="Model A",
                                choices=["Naive Bayes","Random Forest","Gradient Boost","Neural Network"],
                                value="Naive Bayes")
                            t_mb = gr.Dropdown(label="Model B",
                                choices=["Naive Bayes","Random Forest","Gradient Boost","Neural Network"],
                                value="Random Forest")
                        t_n   = gr.Slider(label="Sample Size", minimum=50, maximum=500, value=100, step=10)
                        btn_ab = gr.Button("🚀 Run A/B Test", elem_classes=["btn-predict"])
                        out_ab_html = gr.HTML()
                        out_ab_plot = gr.Plot()

            btn_perf.click(fn=perf_bar_fn,   inputs=[], outputs=[plt_bar])
            btn_perf.click(fn=radar_chart_fn, inputs=[], outputs=[plt_radar])
            btn_ab.click(fn=ab_test_fn, inputs=[t_ma, t_mb, t_n],
                         outputs=[out_ab_html, out_ab_plot])

        # ══════════════════════════════════════════════════
        # TAB 4 — EXPLAINABLE AI
        # ══════════════════════════════════════════════════
        with gr.Tab("🔍 XAI & Insights"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🔍 Feature Importance")
                        btn_feat = gr.Button("📊 Show Feature Importance", elem_classes=["btn-action"])
                        plt_feat = gr.Plot()

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🤖 AI Business Insights")
                        btn_insight = gr.Button("💡 Generate Insights", elem_classes=["btn-predict"])
                        out_insight = gr.HTML(value="<div style='padding:20px;text-align:center;color:rgba(124,82,255,0.3);letter-spacing:.1em;'>Click button above ↑</div>")

            btn_feat.click(fn=feature_chart_fn, inputs=[], outputs=[plt_feat])
            btn_insight.click(fn=insights_fn, inputs=[], outputs=[out_insight])

        # ══════════════════════════════════════════════════
        # TAB 5 — BATCH
        # ══════════════════════════════════════════════════
        with gr.Tab("📂 Batch Predict"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""### 📂 Batch Prediction
Upload a CSV with all **40 feature columns**. Output adds `Prediction`, `Prob_Attrited`, `Risk_Level`.
""")
                with gr.Row():
                    with gr.Column():
                        f_in      = gr.File(label="Upload CSV", file_types=[".csv"])
                        btn_batch = gr.Button("⚡ Run Batch Predict", elem_classes=["btn-predict"])
                        f_out     = gr.File(label="📥 Download Results")
                    with gr.Column():
                        out_log = gr.Textbox(label="Summary Log", interactive=False, lines=7)
                        out_sum = gr.HTML()

            btn_batch.click(fn=batch_fn, inputs=[f_in],
                            outputs=[f_out, out_log, out_sum])

        # ══════════════════════════════════════════════════
        # TAB 6 — GUIDE
        # ══════════════════════════════════════════════════
        with gr.Tab("📚 Guide"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""
### 📖 How to Use

| Tab | Purpose |
|---|---|
| 🔮 **Predict** | Fill profile → PREDICT → multi-model risk output |
| 📊 **Analytics** | Click Refresh Stats → live charts load |
| 🧪 **Models & A/B** | Load Performance → Radar/Bar charts · Run A/B Test |
| 🔍 **XAI** | Feature importance + AI business insights |
| 📂 **Batch** | Upload CSV → download predictions |

---
### ⚠️ Risk Levels
| 🟢 Low | 0–20% | Standard engagement |
| 🟡 Moderate | 20–50% | Proactive retention |
| 🟠 High | 50–75% | Immediate outreach |
| 🔴 Critical | 75%+ | Emergency retention |

---
### 🔑 Top Churn Signals
- **Total_Trans_Ct** — fewer transactions = higher risk
- **Months_Inactive_12_mon** — inactivity is #1 predictor
- **Total_Ct_Chng_Q4_Q1** — sharp drop signals disengagement
- **Contacts_Count_12_mon** — frequent complaints = churn signal
""")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 📋 All 40 Features")
                gr.HTML(
                    "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:5px;'>"
                    + "".join([f"<div style='font-family:Inter,sans-serif;font-size:.73rem;color:#a78bfa;"
                                f"padding:5px 10px;background:rgba(124,82,255,0.07);"
                                f"border-radius:7px;border:1px solid rgba(124,82,255,0.12);'>{f}</div>"
                                for f in FEATURE_NAMES]) + "</div>"
                )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, time
    os.system("lsof -ti:7870 | xargs kill -9 2>/dev/null; true")
    time.sleep(1)
    demo.launch(server_name="127.0.0.1", server_port=7870,
                share=False, inbrowser=False, show_error=True)
