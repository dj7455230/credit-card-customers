"""
Train all models from BankChurners.csv real dataset
Saves: naive.pkl, random_forest.pkl, extra_trees.pkl, logistic.pkl
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── Load Data ─────────────────────────────────────────────────────────────────
print("📂 Loading BankChurners.csv...")
df = pd.read_csv("BankChurners.csv")
print(f"   Shape: {df.shape}")
print(f"   Target:\n{df['Attrition_Flag'].value_counts()}\n")

# ── Target ────────────────────────────────────────────────────────────────────
df["target"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

# ── Feature Engineering ───────────────────────────────────────────────────────
# Drop leaky NB columns and CLIENTNUM
leak_cols = [c for c in df.columns if c.startswith("Naive_Bayes")]
drop_raw   = ["Attrition_Flag", "CLIENTNUM"] + leak_cols
df_clean   = df.drop(columns=drop_raw)

# One-hot encode categorical columns
cat_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
df_enc = pd.get_dummies(df_clean, columns=cat_cols)
feature_cols = [c for c in df_enc.columns if c != "target"]

X = df_enc[feature_cols].values.astype(float)
y = df["target"].values

print(f"✅ Features: {len(feature_cols)}")
print(f"   Feature names: {feature_cols[:5]}... (+{len(feature_cols)-5} more)")
print(f"   Class distribution: 0={sum(y==0)}, 1={sum(y==1)}\n")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Models ──────────────────────────────────────────────────────────────
models = {
    "naive.pkl":          GaussianNB(),
    "random_forest.pkl":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "extra_trees.pkl":    ExtraTreesClassifier(n_estimators=100, random_state=7, n_jobs=-1),
    "logistic.pkl":       LogisticRegression(max_iter=1000, random_state=1),
}

trained = {}
for fname, model in models.items():
    print(f"🔧 Training {fname}...")
    model.fit(X_train, y_train)

    # Attach feature names so model knows column order
    model.feature_names_in_ = np.array(feature_cols)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=["Existing", "Attrited"], digits=3))

    joblib.dump(model, fname)
    print(f"   ✅ Saved {fname}\n")
    trained[fname] = model

print("=" * 60)
print("✅ All models trained and saved!")
print("Feature count:", len(feature_cols))
print("\nFeature names saved in models:")
for name in feature_cols:
    print(f"  - {name}")
