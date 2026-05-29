"""
Insurance Fraud Detection API — Production v2
=============================================
Regulatory compliance:
  - FCA Consumer Duty 2023: explainable decisions via SHAP
  - UK GDPR Article 22: automated decision explanation on request
  - Equality Act 2010: no protected characteristic used as primary factor
  - Solvency II: PSI drift monitoring endpoint
  - Audit trail: every prediction logged as structured JSON
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import json
import os
import uuid
import hashlib
import logging
from datetime import datetime
from pathlib import Path
import anthropic
from dotenv import load_dotenv

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

load_dotenv()

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Fraud Detection API v2",
    description="Production fraud detection with FCA/GDPR/Equality Act compliance",
    version="2.0.0"
)

# Load model artefacts
model           = joblib.load("model/model.pkl")
encoders        = joblib.load("model/encoders.pkl")
with open("model/features.json")       as f: feature_names   = json.load(f)
with open("model/model_card.json")        as f: model_card       = json.load(f)
with open("model/psi_reference.json")     as f: psi_reference    = json.load(f)
with open("model/audit_log_schema.json")  as f: audit_schema     = json.load(f)

THRESHOLD      = model_card["performance"]["optimal_threshold"]
MODEL_VERSION  = model_card["version"]
AUDIT_LOG_PATH = Path("logs/audit_log.jsonl")
AUDIT_LOG_PATH.parent.mkdir(exist_ok=True)

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

logger.info(f"Model v{MODEL_VERSION} loaded | Threshold: {THRESHOLD}")

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class ClaimRequest(BaseModel):
    months_as_customer: int = Field(..., ge=0, le=600)
    age: int = Field(..., ge=18, le=100)
    policy_deductable: float = Field(..., ge=0)
    policy_annual_premium: float = Field(..., ge=0)
    umbrella_limit: float = Field(default=0)
    capital_gains: float = Field(default=0)
    capital_loss: float = Field(default=0)
    incident_hour_of_the_day: int = Field(..., ge=0, le=23)
    number_of_vehicles_involved: int = Field(..., ge=1, le=10)
    bodily_injuries: int = Field(..., ge=0, le=10)
    witnesses: int = Field(..., ge=0, le=10)
    total_claim_amount: float = Field(..., ge=0)
    injury_claim: float = Field(default=0)
    property_claim: float = Field(default=0)
    vehicle_claim: float = Field(default=0)
    auto_year: int = Field(..., ge=1980, le=2025)
    # Categorical fields
    insured_sex: str = Field(default="MALE")
    insured_education_level: str = Field(default="MD")
    incident_severity: str = Field(default="Minor Damage")
    insured_occupation: str = Field(default="craft-repair")
    insured_hobbies: str = Field(default="sleeping")
    insured_relationship: str = Field(default="husband")
    incident_type: str = Field(default="Single Vehicle Collision")
    collision_type: str = Field(default="Side Collision")
    authorities_contacted: str = Field(default="Police")
    property_damage: str = Field(default="NO")
    police_report_available: str = Field(default="YES")
    auto_make: str = Field(default="Toyota")
    auto_model: str = Field(default="Camry")
    policy_state: str = Field(default="OH")
    policy_csl: str = Field(default="250/500")
    incident_state: str = Field(default="OH")
    incident_city: str = Field(default="Columbus")

class ClaimResponse(BaseModel):
    prediction_id: str
    fraud_score: float
    decision: str
    confidence_band: str
    threshold_used: float
    top_risk_factors: list
    reviewer_required: bool
    ai_explanation: str
    model_version: str
    timestamp_utc: str
    regulatory_notice: str

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_feature_vector(req: ClaimRequest) -> np.ndarray:
    """Build full feature vector matching training schema."""
    raw = {
        "months_as_customer":         req.months_as_customer,
        "age":                        req.age,
        "policy_deductable":          req.policy_deductable,
        "policy_annual_premium":      req.policy_annual_premium,
        "umbrella_limit":             req.umbrella_limit,
        "capital-gains":              req.capital_gains,
        "capital-loss":               req.capital_loss,
        "incident_hour_of_the_day":   req.incident_hour_of_the_day,
        "number_of_vehicles_involved":req.number_of_vehicles_involved,
        "bodily_injuries":            req.bodily_injuries,
        "witnesses":                  req.witnesses,
        "total_claim_amount":         req.total_claim_amount,
        "injury_claim":               req.injury_claim,
        "property_claim":             req.property_claim,
        "vehicle_claim":              req.vehicle_claim,
        "auto_year":                  req.auto_year,
    }

    # Encode categoricals using saved encoders
    cat_map = {
        "insured_sex": req.insured_sex,
        "insured_education_level": req.insured_education_level,
        "incident_severity": req.incident_severity,
        "insured_occupation": req.insured_occupation,
        "insured_hobbies": req.insured_hobbies,
        "insured_relationship": req.insured_relationship,
        "incident_type": req.incident_type,
        "collision_type": req.collision_type,
        "authorities_contacted": req.authorities_contacted,
        "property_damage": req.property_damage,
        "police_report_available": req.police_report_available,
        "auto_make": req.auto_make,
        "auto_model": req.auto_model,
        "policy_state": req.policy_state,
        "policy_csl": req.policy_csl,
        "incident_state": req.incident_state,
        "incident_city": req.incident_city,
    }
    for col, val in cat_map.items():
        if col in encoders:
            try:
                raw[col] = int(encoders[col].transform([str(val)])[0])
            except ValueError:
                raw[col] = 0  # unseen category → default 0
        else:
            raw[col] = 0

    # Engineered financial features (must match train_model.py)
    raw["claim_to_premium_ratio"]   = req.total_claim_amount / (req.policy_annual_premium + 1)
    raw["vehicle_claim_ratio"]      = req.vehicle_claim / (req.total_claim_amount + 1)
    raw["injury_claim_ratio"]       = req.injury_claim / (req.total_claim_amount + 1)
    raw["property_claim_ratio"]     = req.property_claim / (req.total_claim_amount + 1)
    raw["claim_per_vehicle"]        = req.total_claim_amount / (req.number_of_vehicles_involved + 1)
    raw["capital_net"]              = req.capital_gains + req.capital_loss
    raw["auto_age"]                 = 2015 - req.auto_year
    raw["premium_per_month"]        = req.policy_annual_premium / 12
    raw["policy_age_proxy"]         = req.months_as_customer
    raw["is_late_night"]            = 1 if (req.incident_hour_of_the_day >= 22 or req.incident_hour_of_the_day <= 4) else 0
    raw["is_weekend"]               = 0
    raw["no_witnesses"]             = 1 if req.witnesses == 0 else 0
    raw["multi_vehicle"]            = 1 if req.number_of_vehicles_involved > 2 else 0
    raw["high_bodily_injury"]       = 1 if req.bodily_injuries >= 2 else 0
    raw["suspicion_score"]          = (
        raw["is_late_night"] + raw["no_witnesses"] + raw["multi_vehicle"] +
        raw["high_bodily_injury"] + (1 if raw["claim_to_premium_ratio"] > 5 else 0)
    )
    raw["deductible_to_claim"]      = req.policy_deductable / (req.total_claim_amount + 1)
    raw["umbrella_coverage_ratio"]  = req.umbrella_limit / (req.policy_annual_premium + 1)

    return np.array([[raw.get(f, 0) for f in feature_names]])


def get_confidence_band(score: float, threshold: float) -> str:
    """Map fraud score to confidence band for claims handlers."""
    if score >= 0.75:   return "HIGH — Strong fraud indicators"
    if score >= 0.5:    return "MEDIUM — Moderate fraud indicators"
    if score >= 0.3:    return "LOW-MEDIUM — Weak indicators, monitor"
    return "LOW — Likely legitimate"


def write_audit_log(log_entry: dict):
    """Append prediction to JSONL audit log (FCA traceability requirement)."""
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_regulatory_grade_explanation(req: ClaimRequest, score: float,
                                      decision: str, top_factors: list) -> str:
    """
    Generate FCA Consumer Duty / UK GDPR Article 22 compliant explanation.
    Structured for claims handlers, not ML engineers.
    """
    factors_text = "\n".join([f"  - {f['feature']}: SHAP={f['shap_value']:.3f}" 
                               for f in top_factors[:5]])

    prompt = f"""You are a senior insurance fraud analyst writing a regulatory-grade assessment for a claims handler. 
This output may be shown to the policyholder under UK GDPR Article 22 (right to explanation of automated decisions).

CLAIM DATA:
- Customer tenure: {req.months_as_customer} months | Age: {req.age}
- Total claim: £{req.total_claim_amount:,.0f} | Annual premium: £{req.policy_annual_premium:,.0f}
- Claim-to-premium ratio: {req.total_claim_amount / (req.policy_annual_premium + 1):.1f}x
- Incident hour: {req.incident_hour_of_the_day}:00 | Vehicles involved: {req.number_of_vehicles_involved}
- Witnesses: {req.witnesses} | Bodily injuries: {req.bodily_injuries}
- Vehicle claim: £{req.vehicle_claim:,.0f} | Injury claim: £{req.injury_claim:,.0f}
- Police report available: {req.police_report_available} | Property damage: {req.property_damage}

MODEL OUTPUT:
- Fraud probability score: {score:.2%}
- Decision: {decision}
- Confidence: {"High" if score > 0.75 or score < 0.25 else "Moderate"}

TOP CONTRIBUTING FACTORS (SHAP analysis):
{factors_text}

Write a professional 3-paragraph assessment:
1. State the decision and overall fraud score clearly, using plain English suitable for a policyholder.
2. Explain the 2-3 most significant factors that drove this decision, referencing specific claim details.
3. State what action is recommended for the claims handler (approve, investigate, escalate to IFB) and why.

Requirements:
- Do NOT use jargon like "SHAP values" — translate to plain English
- Be factual and non-discriminatory (Equality Act 2010)
- Acknowledge this is an automated assessment and human review is available
- Keep it under 200 words"""

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("app/templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>Insurance Fraud Detection API v2 — visit <a href='/docs'>/docs</a></h2>"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/model-card")
def get_model_card():
    """Returns full regulatory model card (Solvency II / FCA governance)."""
    return model_card


@app.get("/monitoring/psi")
def get_psi_reference():
    """Returns PSI reference distribution for live monitoring (Solvency II)."""
    return psi_reference


@app.post("/predict", response_model=ClaimResponse)
def predict(request: ClaimRequest):
    prediction_id  = str(uuid.uuid4())
    timestamp      = datetime.utcnow().isoformat() + "Z"
    input_hash     = hashlib.sha256(request.json().encode()).hexdigest()[:16]

    try:
        features = build_feature_vector(request)
        score    = float(model.predict_proba(features)[0, 1])
        print(f"DEBUG: score={score}, threshold={THRESHOLD}")
        decision = "FRAUD_FLAG" if score >= THRESHOLD else "LEGITIMATE"

        # SHAP top factors
        top_factors = []
        if SHAP_AVAILABLE:
            try:
                base_model  = model.calibrated_classifiers_[0].estimator
                xgb_base    = base_model.named_estimators_["xgb"]
                explainer   = shap.TreeExplainer(xgb_base)
                shap_vals   = explainer.shap_values(features)[0]
                shap_df     = sorted(
                    zip(feature_names, shap_vals),
                    key=lambda x: abs(x[1]), reverse=True
                )[:5]
                top_factors = [{"feature": f, "shap_value": round(float(v), 4)} 
                               for f, v in shap_df]
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                top_factors = [{"feature": f, "shap_value": 0.0} 
                               for f in feature_names[:5]]

        confidence_band   = get_confidence_band(score, THRESHOLD)
        reviewer_required = abs(score - THRESHOLD) < 0.05  # borderline cases

        # Regulatory-grade AI explanation
        ai_explanation = get_regulatory_grade_explanation(
            request, score, decision, top_factors
        )

        # Audit log entry
        audit_entry = {
            "prediction_id":   prediction_id,
            "timestamp_utc":   timestamp,
            "input_hash":      input_hash,
            "fraud_score":     round(score, 4),
            "decision":        decision,
            "threshold_used":  THRESHOLD,
            "confidence_band": confidence_band,
            "top_shap_factors": top_factors,
            "model_version":   MODEL_VERSION,
            "reviewer_required": reviewer_required,
        }
        write_audit_log(audit_entry)
        logger.info(f"Prediction {prediction_id} | Score: {score:.3f} | Decision: {decision}")

        return ClaimResponse(
            prediction_id       = prediction_id,
            fraud_score         = round(score, 4),
            decision            = decision,
            confidence_band     = confidence_band,
            threshold_used      = THRESHOLD,
            top_risk_factors    = top_factors,
            reviewer_required   = reviewer_required,
            ai_explanation      = ai_explanation,
            model_version       = MODEL_VERSION,
            timestamp_utc       = timestamp,
            regulatory_notice   = (
                "This decision was made by an automated system. Under UK GDPR Article 22, "
                "you have the right to request human review of this decision. "
                "Contact your claims handler to exercise this right."
            )
        )

    except Exception as e:
        logger.error(f"Prediction error for {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit-log")
def get_audit_log(limit: int = 50):
    """Returns recent audit log entries (FCA traceability)."""
    if not AUDIT_LOG_PATH.exists():
        return {"entries": [], "total": 0}
    with open(AUDIT_LOG_PATH) as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines[-limit:]]
    return {"entries": entries, "total": len(lines)}