from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import json
import os
import anthropic
from dotenv import load_dotenv
from app.schemas import ClaimRequest, ClaimResponse

load_dotenv()

app = FastAPI(title="Insurance Fraud Detection API")

model = joblib.load("model/model.pkl")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with open("model/features.json") as f:
    feature_names = json.load(f)

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: ClaimRequest):
    input_data = {
        "months_as_customer": request.months_as_customer,
        "age": request.age,
        "policy_deductable": request.policy_deductable,
        "policy_annual_premium": request.policy_annual_premium,
        "umbrella_limit": request.umbrella_limit,
        "capital-gains": request.capital_gains,
        "capital-loss": request.capital_loss,
        "incident_hour_of_the_day": request.incident_hour_of_the_day,
        "number_of_vehicles_involved": request.number_of_vehicles_involved,
        "bodily_injuries": request.bodily_injuries,
        "witnesses": request.witnesses,
        "total_claim_amount": request.total_claim_amount,
        "injury_claim": request.injury_claim,
        "property_claim": request.property_claim,
        "vehicle_claim": request.vehicle_claim,
        "auto_year": request.auto_year,
        "insured_occupation": 1,
        "insured_hobbies": 1,
        "incident_city": 1,
        "auto_make": 1,
        "auto_model": 1,
        "policy_age_days": 1000,
        "incident_month": 6,
        "incident_day_of_week": 2,
        "is_weekend": 0,
        "is_late_night": 1 if request.incident_hour_of_the_day >= 22 or request.incident_hour_of_the_day <= 4 else 0,
        "csl_per_person": 250,
        "csl_per_accident": 500,
        "claim_to_premium_ratio": request.total_claim_amount / (request.policy_annual_premium + 1),
        "vehicle_claim_ratio": request.vehicle_claim / (request.total_claim_amount + 1),
        "injury_claim_ratio": request.injury_claim / (request.total_claim_amount + 1),
        "capital_net": request.capital_gains + request.capital_loss,
        "auto_age": 2015 - request.auto_year,
        "suspicion_score": (
            (1 if request.incident_hour_of_the_day >= 22 or request.incident_hour_of_the_day <= 4 else 0) +
            (1 if request.witnesses == 0 else 0) +
            (1 if request.number_of_vehicles_involved > 2 else 0)
        ),
    }

    features = np.array([[input_data.get(f, 0) for f in feature_names]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": f"""You are an insurance fraud analyst reviewing a claim.

The ML fraud detection model has predicted: {"⚠️ FRAUDULENT" if prediction == 1 else "✅ LEGITIMATE"} (Confidence: {probability*100:.1f}%)

First acknowledge the ML model's prediction in one sentence. Then independently analyze the claim and list 2-3 red flags or reassuring factors you notice. Be concise and professional.

Claim details:
- Customer age: {request.age}, months as customer: {request.months_as_customer}
- Incident hour: {request.incident_hour_of_the_day}:00, vehicles involved: {request.number_of_vehicles_involved}
- Witnesses: {request.witnesses}, bodily injuries: {request.bodily_injuries}
- Total claim: ${request.total_claim_amount:,}, vehicle claim: ${request.vehicle_claim:,}
- Injury claim: ${request.injury_claim:,}, property claim: ${request.property_claim:,}
- Auto year: {request.auto_year}, claim to premium ratio: {request.total_claim_amount / (request.policy_annual_premium + 1):.2f}x

Format your response as:
ML Assessment: [acknowledge the model prediction]
Analysis: [your independent 2-3 red flags or reassuring factors]"""
            }
        ]
    )
    explanation = message.content[0].text

    return {
        "fraud_predicted": int(prediction),
        "probability": float(probability),
        "explanation": explanation
    }