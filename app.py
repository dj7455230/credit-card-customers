"""
╔══════════════════════════════════════════════════════════════════╗
║     BANK CHURN PREDICTOR — Gaussian Naive Bayes Dashboard        ║
║     Powered by: naive.pkl  |  Gradio + Custom UI                 ║
╚══════════════════════════════════════════════════════════════════╝
USAGE:
  pip install gradio joblib scikit-learn numpy pandas
  python app.py
Then open:  http://localhost:7860
"""

import gradio as gr
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "naive.pkl"
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = list(model.feature_names_in_)
CLASSES = {0: "✅ Existing Customer", 1: "⚠️ Attrited Customer"}

# ─── Custom CSS — Purple / Violet Theme + Inter Font ──────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg:        #0d0d1a;
    --surface:   rgba(20, 18, 40, 0.92);
    --glass:     rgba(255,255,255,0.03);
    --border:    rgba(139,92,246,0.3);
    --purple:    #8b5cf6;
    --violet:    #7c3aed;
    --pink:      #ec4899;
    --amber:     #f59e0b;
    --emerald:   #10b981;
    --red:       #ef4444;
    --text:      #f1f0ff;
    --muted:     #9490b5;
    --glow-p:    0 0 20px rgba(139,92,246,0.6), 0 0 40px rgba(139,92,246,0.2);
    --glow-pk:   0 0 20px rgba(236,72,153,0.6), 0 0 40px rgba(236,72,153,0.2);
    --glow-e:    0 0 20px rgba(16,185,129,0.6), 0 0 40px rgba(16,185,129,0.2);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
    min-height: 100vh;
}

/* Animated mesh background */
.gradio-container::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(ellipse 70% 50% at 15% 15%, rgba(139,92,246,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 85% 85%, rgba(236,72,153,0.10) 0%, transparent 55%),
        radial-gradient(ellipse 60% 60% at 50% 50%, rgba(124,58,237,0.06) 0%, transparent 65%);
    animation: meshpulse 12s ease-in-out infinite alternate;
}
@keyframes meshpulse {
    0%   { opacity: 0.7; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.05); }
}

/* ── Header ── */
#app-header {
    background: linear-gradient(135deg, rgba(139,92,246,0.12), rgba(236,72,153,0.08));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 48px;
    text-align: center;
    margin-bottom: 12px;
    position: relative; overflow: hidden;
    box-shadow: var(--glow-p), inset 0 1px 0 rgba(255,255,255,0.07);
}
#app-header::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--purple), var(--pink), transparent);
    animation: sweep 4s linear infinite;
}
@keyframes sweep { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 8px 0;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 44px rgba(0,0,0,0.5), var(--glow-p);
}

/* ── Labels ── */
label, .label-wrap span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.07em !important;
    color: var(--purple) !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
input[type="number"], input[type="text"], .gr-input, textarea {
    background: rgba(139,92,246,0.07) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
input[type="number"]:focus, textarea:focus {
    border-color: var(--purple) !important;
    box-shadow: var(--glow-p) !important;
    outline: none !important;
}
input[type="range"] { accent-color: var(--purple); }

/* ── Dropdowns ── */
.gr-dropdown select, select {
    background: rgba(139,92,246,0.07) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── PREDICT Button ── */
#predict-btn {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em !important;
    padding: 15px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 5px 0 #4c1d95, 0 8px 24px rgba(139,92,246,0.45) !important;
    transform: translateY(0) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
#predict-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 0 #4c1d95, 0 14px 32px rgba(139,92,246,0.6) !important;
}
#predict-btn:active {
    transform: translateY(3px) !important;
    box-shadow: 0 2px 0 #4c1d95, 0 4px 12px rgba(139,92,246,0.3) !important;
}

/* ── RESET Button ── */
#clear-btn {
    background: rgba(236,72,153,0.1) !important;
    border: 1px solid rgba(236,72,153,0.4) !important;
    border-radius: 12px !important;
    color: var(--pink) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    padding: 12px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
#clear-btn:hover {
    background: rgba(236,72,153,0.22) !important;
    box-shadow: var(--glow-pk) !important;
    transform: translateY(-2px) !important;
}

/* ── Result Box ── */
#result-box textarea {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    color: var(--amber) !important;
    background: rgba(245,158,11,0.07) !important;
    border: 2px solid rgba(245,158,11,0.35) !important;
    border-radius: 14px !important;
    min-height: 72px !important;
    animation: glow-amber 2.5s ease-in-out infinite alternate;
}
@keyframes glow-amber {
    0%   { box-shadow: 0 0 10px rgba(245,158,11,0.25); }
    100% { box-shadow: 0 0 26px rgba(245,158,11,0.6); }
}

/* ── Prob Box ── */
#prob-box textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--emerald) !important;
    background: rgba(16,185,129,0.05) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 14px !important;
    text-align: center !important;
}

/* ── Tabs ── */
.tab-nav button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--purple) !important;
    border-color: var(--border) !important;
    background: var(--glass) !important;
}

/* ── Stat Cards ── */
.stat-card {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}

/* ── Progress bars ── */
.bar-wrap { margin: 8px 0; }
.bar-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}
.bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.9s cubic-bezier(.23,1,.32,1);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.35); border-radius: 3px; }

/* ── Markdown ── */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--purple) !important;
    font-weight: 700 !important;
}
.gr-markdown p, .gr-markdown li { color: var(--muted) !important; font-size: 0.9rem; line-height: 1.7; }
.gr-markdown strong { color: var(--text) !important; }
"""

# ─── Header HTML ──────────────────────────────────────────────────────────────
HEADER_HTML = """
<div id="app-header">
  <div style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:800;
               background:linear-gradient(135deg,#8b5cf6,#ec4899,#f59e0b);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;letter-spacing:0.04em;margin-bottom:8px;">
    CHURN ORACLE
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#9490b5;
               letter-spacing:0.16em;text-transform:uppercase;margin-bottom:20px;">
    Bank Customer Attrition Predictor &nbsp;·&nbsp; Gaussian Naïve Bayes
  </div>
  <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
    <div style="text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                  color:#8b5cf6;font-weight:700;">40</div>
      <div style="font-size:0.65rem;color:#4a4870;letter-spacing:0.14em;text-transform:uppercase;">Features</div>
    </div>
    <div style="width:1px;background:rgba(139,92,246,0.2);"></div>
    <div style="text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                  color:#ec4899;font-weight:700;">2</div>
      <div style="font-size:0.65rem;color:#4a4870;letter-spacing:0.14em;text-transform:uppercase;">Classes</div>
    </div>
    <div style="width:1px;background:rgba(139,92,246,0.2);"></div>
    <div style="text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                  color:#10b981;font-weight:700;">GNB</div>
      <div style="font-size:0.65rem;color:#4a4870;letter-spacing:0.14em;text-transform:uppercase;">Model</div>
    </div>
    <div style="width:1px;background:rgba(139,92,246,0.2);"></div>
    <div style="text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.6rem;
                  color:#f59e0b;font-weight:700;">v2</div>
      <div style="font-size:0.65rem;color:#4a4870;letter-spacing:0.14em;text-transform:uppercase;">Dashboard</div>
    </div>
  </div>
</div>
"""

# ─── Feature Defaults ─────────────────────────────────────────────────────────
DEFAULTS = {
    "CLIENTNUM":                     738000000,
    "Customer_Age":                  46,
    "Dependent_count":               2,
    "Months_on_book":                36,
    "Total_Relationship_Count":      4,
    "Months_Inactive_12_mon":        2,
    "Contacts_Count_12_mon":         3,
    "Credit_Limit":                  8500.0,
    "Total_Revolving_Bal":           1000.0,
    "Avg_Open_To_Buy":               7500.0,
    "Total_Amt_Chng_Q4_Q1":         0.70,
    "Total_Trans_Amt":               4000.0,
    "Total_Trans_Ct":                60,
    "Total_Ct_Chng_Q4_Q1":          0.62,
    "Avg_Utilization_Ratio":         0.27,
    "NB1":                           0.97,
    "NB2":                           0.03,
    # one-hot defaults
    "Gender":                        "Female",
    "Education_Level":               "Graduate",
    "Marital_Status":                "Married",
    "Income_Category":               "$40K - $60K",
    "Card_Category":                 "Blue",
}

# ─── Prediction Logic ─────────────────────────────────────────────────────────
def build_prob_bars(proba):
    p0 = proba[0] * 100
    p1 = proba[1] * 100
    return f"""
<div style='padding:6px 0;'>
  <div class='bar-wrap'>
    <div class='bar-label'>
      <span>✅ Existing Customer</span>
      <span style='color:#10b981;font-weight:700;'>{p0:.1f}%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{p0}%;background:linear-gradient(90deg,#10b981,#34d399);'></div>
    </div>
  </div>
  <div class='bar-wrap' style='margin-top:10px;'>
    <div class='bar-label'>
      <span>⚠️ Attrited Customer</span>
      <span style='color:#ec4899;font-weight:700;'>{p1:.1f}%</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{p1}%;background:linear-gradient(90deg,#ec4899,#f472b6);'></div>
    </div>
  </div>
</div>"""


def predict(
    clientnum, age, dep_cnt, months_book,
    rel_cnt, months_inact, contacts_cnt,
    credit_lim, revolv_bal, open_to_buy,
    amt_chng, trans_amt, trans_ct, ct_chng, util_ratio,
    nb1, nb2,
    gender, education, marital, income, card,
):
    gender_F = 1 if gender == "Female" else 0
    gender_M = 1 - gender_F

    edu_cols  = ["College","Doctorate","Graduate","High School","Post-Graduate","Uneducated","Unknown"]
    edu_vec   = [1 if e == education else 0 for e in edu_cols]

    mar_cols  = ["Divorced","Married","Single","Unknown"]
    mar_vec   = [1 if m == marital else 0 for m in mar_cols]

    inc_cols  = ["$120K +","$40K - $60K","$60K - $80K","$80K - $120K","Less than $40K","Unknown"]
    inc_vec   = [1 if i == income else 0 for i in inc_cols]

    card_cols = ["Blue","Gold","Platinum","Silver"]
    card_vec  = [1 if c == card else 0 for c in card_cols]

    row = np.array([[
        clientnum, age, dep_cnt, months_book,
        rel_cnt, months_inact, contacts_cnt,
        credit_lim, revolv_bal, open_to_buy,
        amt_chng, trans_amt, trans_ct, ct_chng, util_ratio,
        nb1, nb2,
        gender_F, gender_M,
        *edu_vec, *mar_vec, *inc_vec, *card_vec,
    ]])

    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    risk  = proba[1]

    result_text = CLASSES[pred]
    prob_text   = f"P(Existing) = {proba[0]*100:.2f}%     |     P(Attrited) = {proba[1]*100:.2f}%"
    bars_html   = build_prob_bars(proba)

    # Risk level
    if risk < 0.20:
        badge, badge_color, risk_label = "🟢  LOW RISK", "#10b981", "Low"
    elif risk < 0.50:
        badge, badge_color, risk_label = "🟡  MODERATE RISK", "#f59e0b", "Moderate"
    elif risk < 0.75:
        badge, badge_color, risk_label = "🟠  HIGH RISK", "#f97316", "High"
    else:
        badge, badge_color, risk_label = "🔴  CRITICAL RISK", "#ef4444", "Critical"

    # Gauge bar (0-100%)
    gauge_pct = risk * 100
    gauge_color = badge_color

    badge_html = f"""
<div style='margin-top:6px;'>
  <!-- Risk Badge -->
  <div style='text-align:center;padding:16px;border-radius:14px;
              background:rgba(255,255,255,0.03);border:1px solid {badge_color}44;
              box-shadow:0 0 20px {badge_color}22;margin-bottom:12px;'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.3rem;
                font-weight:800;color:{badge_color};letter-spacing:0.08em;'>{badge}</div>
    <div style='font-family:Inter,sans-serif;font-size:0.82rem;
                color:#9490b5;margin-top:6px;letter-spacing:0.1em;'>
      ATTRITION PROBABILITY: <strong style='color:{badge_color};'>{risk*100:.1f}%</strong>
    </div>
  </div>

  <!-- Gauge Bar -->
  <div style='margin-bottom:14px;'>
    <div style='display:flex;justify-content:space-between;
                font-family:Inter,sans-serif;font-size:0.72rem;
                color:#6b6890;margin-bottom:5px;'>
      <span>0%</span><span>Risk Gauge</span><span>100%</span>
    </div>
    <div style='background:rgba(255,255,255,0.06);border-radius:10px;height:14px;overflow:hidden;'>
      <div style='width:{gauge_pct:.1f}%;height:100%;border-radius:10px;
                  background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);
                  transition:width 0.9s ease;'></div>
    </div>
  </div>

  <!-- Probability Bars -->
  {bars_html}

  <!-- Stats Row -->
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;'>
    <div class='stat-card'>
      <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                  font-weight:700;color:#8b5cf6;'>{proba[0]*100:.1f}%</div>
      <div style='font-size:0.68rem;color:#6b6890;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:3px;'>Stay Probability</div>
    </div>
    <div class='stat-card'>
      <div style='font-family:Space Grotesk,sans-serif;font-size:1.1rem;
                  font-weight:700;color:#ec4899;'>{proba[1]*100:.1f}%</div>
      <div style='font-size:0.68rem;color:#6b6890;text-transform:uppercase;
                  letter-spacing:0.1em;margin-top:3px;'>Churn Probability</div>
    </div>
  </div>
</div>
"""
    return result_text, prob_text, badge_html


# ─── Badge Placeholder ────────────────────────────────────────────────────────
BADGE_PLACEHOLDER = """
<div style='text-align:center;padding:28px;color:#3d3a5c;
            font-family:Inter,sans-serif;font-size:0.9rem;
            letter-spacing:0.1em;border:1px dashed rgba(139,92,246,0.15);
            border-radius:14px;margin-top:6px;'>
  ✦ &nbsp; Run prediction to see risk analysis &nbsp; ✦
</div>"""


def clear_inputs():
    return (
        DEFAULTS["CLIENTNUM"], DEFAULTS["Customer_Age"], DEFAULTS["Dependent_count"],
        DEFAULTS["Months_on_book"], DEFAULTS["Total_Relationship_Count"],
        DEFAULTS["Months_Inactive_12_mon"], DEFAULTS["Contacts_Count_12_mon"],
        DEFAULTS["Credit_Limit"], DEFAULTS["Total_Revolving_Bal"],
        DEFAULTS["Avg_Open_To_Buy"], DEFAULTS["Total_Amt_Chng_Q4_Q1"],
        DEFAULTS["Total_Trans_Amt"], DEFAULTS["Total_Trans_Ct"],
        DEFAULTS["Total_Ct_Chng_Q4_Q1"], DEFAULTS["Avg_Utilization_Ratio"],
        DEFAULTS["NB1"], DEFAULTS["NB2"],
        "Female", "Graduate", "Married", "$40K - $60K", "Blue",
        "", "", BADGE_PLACEHOLDER,
    )


# ─── Batch Predict Logic ──────────────────────────────────────────────────────
def batch_predict(file):
    if file is None:
        return None, "❌ No file uploaded.", ""
    try:
        import pandas as pd
        df = pd.read_csv(file.name)
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            return None, f"❌ Missing columns: {missing[:5]}…", ""

        X = df[FEATURE_NAMES].values
        preds  = model.predict(X)
        probas = model.predict_proba(X)[:, 1]

        df["Prediction"]    = preds
        df["Prob_Attrited"] = probas.round(4)
        df["Risk_Level"]    = pd.cut(
            probas,
            bins=[0, 0.2, 0.5, 0.75, 1.0],
            labels=["Low", "Moderate", "High", "Critical"],
            include_lowest=True,
        )

        out_path = "/tmp/churn_predictions.csv"
        df.to_csv(out_path, index=False)

        n_total   = len(df)
        n_attrit  = int(preds.sum())
        n_exist   = n_total - n_attrit
        avg_risk  = probas.mean() * 100
        high_risk = int((probas >= 0.5).sum())

        log = (
            f"✅ Processed {n_total} rows\n"
            f"   Existing Customers : {n_exist} ({n_exist/n_total*100:.1f}%)\n"
            f"   Attrited Customers : {n_attrit} ({n_attrit/n_total*100:.1f}%)\n"
            f"   Avg Attrition Risk : {avg_risk:.2f}%\n"
            f"   High/Critical Risk : {high_risk} customers"
        )

        summary_html = f"""
<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px;'>
  <div class='stat-card'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;
                font-weight:700;color:#8b5cf6;'>{n_total}</div>
    <div style='font-size:0.68rem;color:#6b6890;text-transform:uppercase;
                letter-spacing:0.1em;margin-top:4px;'>Total Rows</div>
  </div>
  <div class='stat-card'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;
                font-weight:700;color:#ec4899;'>{n_attrit}</div>
    <div style='font-size:0.68rem;color:#6b6890;text-transform:uppercase;
                letter-spacing:0.1em;margin-top:4px;'>Attrited</div>
  </div>
  <div class='stat-card'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:1.4rem;
                font-weight:700;color:#f59e0b;'>{avg_risk:.1f}%</div>
    <div style='font-size:0.68rem;color:#6b6890;text-transform:uppercase;
                letter-spacing:0.1em;margin-top:4px;'>Avg Risk</div>
  </div>
</div>
<div style='margin-top:14px;'>
  <div class='bar-wrap'>
    <div class='bar-label'>
      <span>Existing ({n_exist/n_total*100:.1f}%)</span>
      <span style='color:#10b981;'>{n_exist}</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{n_exist/n_total*100:.1f}%;
           background:linear-gradient(90deg,#10b981,#34d399);'></div>
    </div>
  </div>
  <div class='bar-wrap' style='margin-top:8px;'>
    <div class='bar-label'>
      <span>Attrited ({n_attrit/n_total*100:.1f}%)</span>
      <span style='color:#ec4899;'>{n_attrit}</span>
    </div>
    <div class='bar-bg'>
      <div class='bar-fill' style='width:{n_attrit/n_total*100:.1f}%;
           background:linear-gradient(90deg,#ec4899,#f472b6);'></div>
    </div>
  </div>
</div>"""

        return out_path, log, summary_html

    except Exception as e:
        return None, f"❌ Error: {e}", ""


# ─── UI Build ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    css=CUSTOM_CSS,
    title="Churn Oracle — Bank Attrition Predictor",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.pink,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — PREDICT
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🔮  PREDICT"):
            with gr.Row(equal_height=False):

                # ── LEFT: Inputs ──────────────────────────────────────────────
                with gr.Column(scale=3, min_width=420):

                    # Section 1 — Account Info
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🏦 Account Information")
                        with gr.Row():
                            clientnum = gr.Number(
                                label="Client Number",
                                value=DEFAULTS["CLIENTNUM"], precision=0,
                            )
                            age = gr.Slider(
                                label="Customer Age",
                                minimum=18, maximum=90,
                                value=DEFAULTS["Customer_Age"], step=1,
                            )
                        with gr.Row():
                            dep_cnt = gr.Slider(
                                label="Dependents",
                                minimum=0, maximum=10,
                                value=DEFAULTS["Dependent_count"], step=1,
                            )
                            months_book = gr.Slider(
                                label="Months on Book",
                                minimum=1, maximum=60,
                                value=DEFAULTS["Months_on_book"], step=1,
                            )
                            rel_cnt = gr.Slider(
                                label="Relationship Count",
                                minimum=1, maximum=10,
                                value=DEFAULTS["Total_Relationship_Count"], step=1,
                            )
                        with gr.Row():
                            months_inact = gr.Slider(
                                label="Months Inactive",
                                minimum=0, maximum=12,
                                value=DEFAULTS["Months_Inactive_12_mon"], step=1,
                            )
                            contacts_cnt = gr.Slider(
                                label="Contacts (12 mon)",
                                minimum=0, maximum=10,
                                value=DEFAULTS["Contacts_Count_12_mon"], step=1,
                            )

                    # Section 2 — Financials
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 💳 Financial Details")
                        with gr.Row():
                            credit_lim = gr.Number(
                                label="Credit Limit ($)",
                                value=DEFAULTS["Credit_Limit"],
                            )
                            revolv_bal = gr.Number(
                                label="Revolving Balance ($)",
                                value=DEFAULTS["Total_Revolving_Bal"],
                            )
                            open_to_buy = gr.Number(
                                label="Avg Open To Buy ($)",
                                value=DEFAULTS["Avg_Open_To_Buy"],
                            )
                        with gr.Row():
                            trans_amt = gr.Number(
                                label="Total Trans Amount ($)",
                                value=DEFAULTS["Total_Trans_Amt"],
                            )
                            trans_ct = gr.Number(
                                label="Total Trans Count",
                                value=DEFAULTS["Total_Trans_Ct"], precision=0,
                            )
                            util_ratio = gr.Slider(
                                label="Utilization Ratio",
                                minimum=0.0, maximum=1.0,
                                value=DEFAULTS["Avg_Utilization_Ratio"], step=0.01,
                            )
                        with gr.Row():
                            amt_chng = gr.Number(
                                label="Amt Change Q4/Q1",
                                value=DEFAULTS["Total_Amt_Chng_Q4_Q1"], precision=4,
                            )
                            ct_chng = gr.Number(
                                label="Count Change Q4/Q1",
                                value=DEFAULTS["Total_Ct_Chng_Q4_Q1"], precision=4,
                            )

                    # Section 3 — NB Scores
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🤖 Internal NB Classifier Scores")
                        gr.Markdown(
                            "Prior NB classifier outputs used as model features."
                        )
                        with gr.Row():
                            nb1 = gr.Slider(
                                label="NB Score — Class 1 (Existing)",
                                minimum=0.0, maximum=1.0,
                                value=DEFAULTS["NB1"], step=0.001,
                            )
                            nb2 = gr.Slider(
                                label="NB Score — Class 2 (Attrited)",
                                minimum=0.0, maximum=1.0,
                                value=DEFAULTS["NB2"], step=0.001,
                            )

                    # Section 4 — Demographics
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### 🧑 Demographics & Card Profile")
                        with gr.Row():
                            gender = gr.Radio(
                                label="Gender",
                                choices=["Female", "Male"],
                                value="Female",
                            )
                            education = gr.Dropdown(
                                label="Education Level",
                                choices=["College","Doctorate","Graduate",
                                         "High School","Post-Graduate",
                                         "Uneducated","Unknown"],
                                value="Graduate",
                            )
                            marital = gr.Dropdown(
                                label="Marital Status",
                                choices=["Divorced","Married","Single","Unknown"],
                                value="Married",
                            )
                        with gr.Row():
                            income = gr.Dropdown(
                                label="Income Category",
                                choices=["$120K +","$40K - $60K","$60K - $80K",
                                         "$80K - $120K","Less than $40K","Unknown"],
                                value="$40K - $60K",
                            )
                            card = gr.Dropdown(
                                label="Card Category",
                                choices=["Blue","Gold","Platinum","Silver"],
                                value="Blue",
                            )

                # ── RIGHT: Output ─────────────────────────────────────────────
                with gr.Column(scale=2, min_width=300):
                    gr.Markdown("### 🎯 Prediction Result")

                    result_out = gr.Textbox(
                        label="Prediction",
                        interactive=False,
                        elem_id="result-box",
                        placeholder="Result appears here…",
                    )
                    prob_out = gr.Textbox(
                        label="Probabilities",
                        interactive=False,
                        elem_id="prob-box",
                        placeholder="Probabilities appear here…",
                    )
                    badge_out = gr.HTML(value=BADGE_PLACEHOLDER)

                    with gr.Row():
                        predict_btn = gr.Button("🔮  PREDICT", elem_id="predict-btn")
                        clear_btn   = gr.Button("↺  RESET",   elem_id="clear-btn")

                    gr.HTML("""
                    <div style='margin-top:18px;padding:16px;border-radius:14px;
                                border:1px solid rgba(139,92,246,0.2);
                                background:rgba(139,92,246,0.05);'>
                      <div style='font-family:Space Grotesk,sans-serif;font-size:0.68rem;
                                  color:#8b5cf6;letter-spacing:0.14em;
                                  text-transform:uppercase;margin-bottom:10px;'>
                        ⚡ Model Info
                      </div>
                      <div style='font-family:Inter,sans-serif;font-size:0.8rem;
                                  color:#5a5880;line-height:1.8;'>
                        <span style='color:#7c6fa0;'>Model:</span> Gaussian Naïve Bayes<br>
                        <span style='color:#7c6fa0;'>Classes:</span> 0 = Existing · 1 = Attrited<br>
                        <span style='color:#7c6fa0;'>Features:</span> 40 (numeric + one-hot)<br>
                        <span style='color:#7c6fa0;'>Var Smoothing:</span> 1e-09
                      </div>
                    </div>
                    """)

            # Wire up
            inputs_list  = [
                clientnum, age, dep_cnt, months_book, rel_cnt,
                months_inact, contacts_cnt, credit_lim, revolv_bal,
                open_to_buy, amt_chng, trans_amt, trans_ct, ct_chng,
                util_ratio, nb1, nb2,
                gender, education, marital, income, card,
            ]
            outputs_list = [result_out, prob_out, badge_out]

            predict_btn.click(fn=predict, inputs=inputs_list, outputs=outputs_list)
            clear_btn.click(fn=clear_inputs, inputs=[], outputs=inputs_list + outputs_list)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — BATCH PREDICT
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📂  BATCH"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""
### 📂 Batch Prediction via CSV

Upload a CSV with all **40 feature columns**. The app adds:
- `Prediction` — 0 (Existing) or 1 (Attrited)
- `Prob_Attrited` — churn probability (0–1)
- `Risk_Level` — Low / Moderate / High / Critical
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
                        batch_btn = gr.Button("⚡  RUN BATCH PREDICT", elem_id="predict-btn")
                        csv_output = gr.File(label="📥 Download Results")
                    with gr.Column(scale=1):
                        batch_log = gr.Textbox(
                            label="Summary Log",
                            interactive=False,
                            lines=6,
                        )
                        batch_summary = gr.HTML(value="")

            batch_btn.click(
                fn=batch_predict,
                inputs=[csv_input],
                outputs=[csv_output, batch_log, batch_summary],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — ANALYTICS (new tab)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📊  ANALYTICS"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 📊 Live Risk Analyzer")
                gr.Markdown(
                    "Adjust the sliders below to see how key features affect churn risk in real time."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        a_trans_ct = gr.Slider(
                            label="Total Transaction Count",
                            minimum=10, maximum=140, value=60, step=1,
                        )
                        a_months_inact = gr.Slider(
                            label="Months Inactive",
                            minimum=0, maximum=6, value=2, step=1,
                        )
                        a_util = gr.Slider(
                            label="Utilization Ratio",
                            minimum=0.0, maximum=1.0, value=0.27, step=0.01,
                        )
                        a_contacts = gr.Slider(
                            label="Contacts Count (12 mon)",
                            minimum=0, maximum=10, value=3, step=1,
                        )
                    with gr.Column(scale=1):
                        analytics_out = gr.HTML(value="<div style='padding:20px;color:#3d3a5c;text-align:center;'>Adjust sliders to analyze</div>")

            def analytics_predict(trans_ct_val, months_inact_val, util_val, contacts_val):
                # Use defaults for everything else, override the 4 sliders
                row = np.array([[
                    DEFAULTS["CLIENTNUM"],
                    DEFAULTS["Customer_Age"],
                    DEFAULTS["Dependent_count"],
                    DEFAULTS["Months_on_book"],
                    DEFAULTS["Total_Relationship_Count"],
                    months_inact_val,
                    contacts_val,
                    DEFAULTS["Credit_Limit"],
                    DEFAULTS["Total_Revolving_Bal"],
                    DEFAULTS["Avg_Open_To_Buy"],
                    DEFAULTS["Total_Amt_Chng_Q4_Q1"],
                    DEFAULTS["Total_Trans_Amt"],
                    trans_ct_val,
                    DEFAULTS["Total_Ct_Chng_Q4_Q1"],
                    util_val,
                    DEFAULTS["NB1"],
                    DEFAULTS["NB2"],
                    1, 0,           # Gender_F, Gender_M
                    0, 0, 1, 0, 0, 0, 0,  # Education: Graduate
                    0, 1, 0, 0,     # Marital: Married
                    0, 1, 0, 0, 0, 0,  # Income: $40K-$60K
                    1, 0, 0, 0,     # Card: Blue
                ]])
                proba = model.predict_proba(row)[0]
                risk  = proba[1]

                if risk < 0.20:   color, label = "#10b981", "LOW RISK"
                elif risk < 0.50: color, label = "#f59e0b", "MODERATE RISK"
                elif risk < 0.75: color, label = "#f97316", "HIGH RISK"
                else:             color, label = "#ef4444", "CRITICAL RISK"

                return f"""
<div style='padding:20px;'>
  <div style='text-align:center;margin-bottom:20px;'>
    <div style='font-family:Space Grotesk,sans-serif;font-size:2rem;
                font-weight:800;color:{color};'>{risk*100:.1f}%</div>
    <div style='font-family:Inter,sans-serif;font-size:0.8rem;
                color:{color};letter-spacing:0.12em;margin-top:4px;'>{label}</div>
  </div>
  <div class='bar-bg' style='height:16px;margin-bottom:20px;'>
    <div class='bar-fill' style='width:{risk*100:.1f}%;
         background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);'></div>
  </div>
  <div style='font-family:Inter,sans-serif;font-size:0.82rem;color:#6b6890;line-height:2;'>
    <div>📊 Trans Count: <strong style='color:#e0e0ff;'>{trans_ct_val}</strong></div>
    <div>😴 Months Inactive: <strong style='color:#e0e0ff;'>{months_inact_val}</strong></div>
    <div>💳 Utilization: <strong style='color:#e0e0ff;'>{util_val:.0%}</strong></div>
    <div>📞 Contacts: <strong style='color:#e0e0ff;'>{contacts_val}</strong></div>
  </div>
</div>"""

            for slider in [a_trans_ct, a_months_inact, a_util, a_contacts]:
                slider.change(
                    fn=analytics_predict,
                    inputs=[a_trans_ct, a_months_inact, a_util, a_contacts],
                    outputs=[analytics_out],
                )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — GUIDE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📚  GUIDE"):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("""
### 📖 How to Use

**Tab 1 — Predict**
Fill in the customer profile and click **🔮 PREDICT**. All fields are pre-filled with sensible defaults.

**Tab 2 — Batch**
Upload a CSV with all 40 feature columns. Download results with predictions, probabilities, and risk levels.

**Tab 3 — Analytics**
Use the live sliders to explore how key features affect churn risk interactively.

---

### 🔑 Key Features Explained

| Feature | Description |
|---|---|
| `Total_Trans_Ct` | Number of transactions in last 12 months — lower = higher risk |
| `Total_Trans_Amt` | Total transaction amount — lower = higher risk |
| `Months_Inactive_12_mon` | Months with no activity — higher = higher risk |
| `Contacts_Count_12_mon` | Customer service contacts — higher = higher risk |
| `Avg_Utilization_Ratio` | Credit utilization — very low can indicate disengagement |
| `Total_Ct_Chng_Q4_Q1` | Change in transaction count Q4 vs Q1 — drop = higher risk |
| `NB1 / NB2` | Prior NB classifier scores used as meta-features |

---

### ⚠️ Risk Levels

| Level | Attrition Probability |
|---|---|
| 🟢 Low | < 20% |
| 🟡 Moderate | 20% – 50% |
| 🟠 High | 50% – 75% |
| 🔴 Critical | > 75% |
""")

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 📋 All 40 Feature Names")
                gr.HTML(
                    "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:6px;'>"
                    + "".join([
                        f"<div style='font-family:Inter,sans-serif;font-size:0.78rem;"
                        f"color:#7c6fa0;padding:5px 10px;background:rgba(139,92,246,0.06);"
                        f"border-radius:6px;border:1px solid rgba(139,92,246,0.12);'>{f}</div>"
                        for f in FEATURE_NAMES
                    ])
                    + "</div>"
                )

# ─── Launch ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=False,  # Manual browser open
        favicon_path=None,
    )
