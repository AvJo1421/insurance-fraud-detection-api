from fastapi import FastAPI
import joblib
import numpy as np
from app.schemas import ClaimRequest, ClaimResponse

app = FastAPI(title="Insurance Fraud Detection API")

model = joblib.load("model/model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ClaimResponse)
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

    return ClaimResponse(
        fraud_predicted=int(prediction),
        probability=float(probability)
    )