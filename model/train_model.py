import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import json
import os

# Load data
df = pd.read_csv("data/insurance_claims.csv")

# Drop truly useless columns
df = df.drop(columns=["policy_number", "incident_location", "insured_zip", "_c39"])

# ── Target column ──────────────────────────────────────────────
df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# ── Handle missing values ──────────────────────────────────────
df = df.replace("?", np.nan)
df["collision_type"] = df["collision_type"].fillna("Unknown")
df["property_damage"] = df["property_damage"].fillna("Unknown")
df["police_report_available"] = df["police_report_available"].fillna("Unknown")

# ── Date feature engineering ───────────────────────────────────
df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])

df["policy_age_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
df["incident_month"] = df["incident_date"].dt.month
df["incident_day_of_week"] = df["incident_date"].dt.dayofweek
df["is_weekend"] = df["incident_day_of_week"].isin([5, 6]).astype(int)
df["is_late_night"] = ((df["incident_hour_of_the_day"] >= 22) | (df["incident_hour_of_the_day"] <= 4)).astype(int)

df = df.drop(columns=["policy_bind_date", "incident_date"])

# ── Split policy_csl ───────────────────────────────────────────
df["csl_per_person"] = df["policy_csl"].str.split("/").str[0].astype(int)
df["csl_per_accident"] = df["policy_csl"].str.split("/").str[1].astype(int)
df = df.drop(columns=["policy_csl"])

# ── New engineered features ────────────────────────────────────
df["claim_to_premium_ratio"] = df["total_claim_amount"] / (df["policy_annual_premium"] + 1)
df["vehicle_claim_ratio"] = df["vehicle_claim"] / (df["total_claim_amount"] + 1)
df["injury_claim_ratio"] = df["injury_claim"] / (df["total_claim_amount"] + 1)
df["capital_net"] = df["capital-gains"] + df["capital-loss"]
df["auto_age"] = 2015 - df["auto_year"]

df["suspicion_score"] = (
    df["is_late_night"] +
    (df["witnesses"] == 0).astype(int) +
    (df["police_report_available"] == "NO").astype(int) +
    (df["authorities_contacted"] == "None").astype(int) +
    (df["number_of_vehicles_involved"] > 2).astype(int)
)

# ── One-hot encode low cardinality columns ─────────────────────
low_card_cols = [
    "insured_sex", "insured_education_level", "insured_relationship",
    "incident_type", "incident_severity", "authorities_contacted",
    "collision_type", "property_damage", "police_report_available",
    "policy_state", "incident_state"
]
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

# ── Label encode high cardinality columns ──────────────────────
high_card_cols = ["insured_occupation", "insured_hobbies", "incident_city",
                  "auto_make", "auto_model"]
for col in high_card_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ── Split features and target ──────────────────────────────────
X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]

print(f"Total features: {X.shape[1]}")
print(f"Fraud cases: {y.sum()} / {len(y)} ({y.mean():.1%})")

# ── Train/test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Calculate scale_pos_weight ─────────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos
print(f"\nscale_pos_weight: {scale:.2f} (ratio of legitimate to fraud)")

# ── Train XGBoost ──────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    random_state=42,
    eval_metric="auc",
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ── Evaluation ─────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {model.score(X_test, y_test):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

# ── Feature importance ─────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 most important features:")
print(feat_imp.nlargest(10))

# ── Save everything ────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

feature_names = list(X.columns)
with open("model/features.json", "w") as f:
    json.dump(feature_names, f)

print(f"\nModel saved! Features: {len(feature_names)}")