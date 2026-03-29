import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv("data/insurance_claims.csv")

# Drop useless columns
df = df.drop(columns=["policy_number", "policy_bind_date", "incident_date",
                       "incident_location", "insured_zip", "_c39"])

# Target column
df["fraud_reported"] = df["fraud_reported"].map({"Y": 1, "N": 0})

# Encode all text columns to numbers
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split features and target
X = df.drop(columns=["fraud_reported"])
y = df["fraud_reported"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Print accuracy
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
# Save feature names
import json
feature_names = list(X.columns)
with open("model/features.json", "w") as f:
    json.dump(feature_names, f)
print("Features saved:", feature_names)
# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")