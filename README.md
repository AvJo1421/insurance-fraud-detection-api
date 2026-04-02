# Insurance Fraud Detection API

An end-to-end machine learning system that detects insurance fraud in real-time, powered by XGBoost and Claude AI.

## Live Demo
**[Try it here](http://13.218.98.72:8000)**

## What it does
- Accepts insurance claim details via a web form
- Predicts whether the claim is fraudulent using an XGBoost ML model
- Generates an AI-powered explanation of key risk factors using Claude API
- Returns results with confidence score and numbered findings

## Tech Stack
- **ML Model** — XGBoost with feature engineering (ROC-AUC: 0.81)
- **API** — FastAPI + Uvicorn
- **AI** — Anthropic Claude API for explainability
- **Frontend** — HTML/CSS
- **Containerisation** — Docker
- **Cloud** — AWS EC2 + ECR
- **CI/CD** — GitHub Actions (auto-deploy on push)

## Features
- 70 engineered features including claim ratios, policy age, suspicion score
- Handles class imbalance with XGBoost scale_pos_weight
- Real-time fraud explanation with numbered red flags
- Fully automated deployment pipeline

## Architecture
```
User → FastAPI → XGBoost Model → Prediction
                      ↓
               Claude AI API → Explanation
```

## Local Setup
```bash
git clone https://github.com/AvJo1421/insurance-fraud-detection-api
cd insurance-fraud-detection-api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.80 |
| ROC-AUC | 0.81 |
| Fraud Recall | 0.57 |
| Fraud Precision | 0.60 |

## Author
Atharva Atul Joshi — Data Scientist
[LinkedIn](https://www.linkedin.com/in/atharva-joshi-344754213/)
