from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from app.schemas import ClaimRequest, ClaimResponse
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Insurance Fraud Detection API")

model = joblib.load("model/model.pkl")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

@app.get("/", response_class=HTMLResponse)
def home():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: ClaimRequest):
    features = np.array([[
        request.months_as_customer,
        request.age,
        request.policy_state,
        request.policy_csl,
        request.policy_deductable,
        request.policy_annual_premium,
        request.umbrella_limit,
        request.insured_sex,
        request.insured_education_level,
        request.insured_occupation,
        request.insured_hobbies,
        request.insured_relationship,
        request.capital_gains,
        request.capital_loss,
        request.incident_type,
        request.collision_type,
        request.incident_severity,
        request.authorities_contacted,
        request.incident_state,
        request.incident_city,
        request.incident_hour_of_the_day,
        request.number_of_vehicles_involved,
        request.property_damage,
        request.bodily_injuries,
        request.witnesses,
        request.police_report_available,
        request.total_claim_amount,
        request.injury_claim,
        request.property_claim,
        request.vehicle_claim,
        request.auto_make,
        request.auto_model,
        request.auto_year,
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"""You are an insurance fraud analyst. Analyze this claim and explain in 2-3 sentences why it {"appears fraudulent" if prediction == 1 else "appears legitimate"}.

Claim details:
- Customer age: {request.age}, months as customer: {request.months_as_customer}
- Incident hour: {request.incident_hour_of_the_day}, vehicles involved: {request.number_of_vehicles_involved}
- Witnesses: {request.witnesses}, bodily injuries: {request.bodily_injuries}
- Total claim: ${request.total_claim_amount}, vehicle claim: ${request.vehicle_claim}
- Injury claim: ${request.injury_claim}, property claim: ${request.property_claim}
- Auto year: {request.auto_year}

Be concise and professional. Focus on the most suspicious or reassuring factors."""
            }
        ]
    )

    explanation = message.content[0].text

    return {
        "fraud_predicted": int(prediction),
        "probability": float(probability),
        "explanation": explanation
    }