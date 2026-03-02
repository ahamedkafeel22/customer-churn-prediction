from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Customer Churn Prediction API",
              description="Predicts telecom customer churn using Random Forest",
              version="1.0")

# Input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float
    tenure_group: int
    charge_category: int

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!",
            "model": "Random Forest",
            "roc_auc": 0.9291}

@app.post("/predict")
def predict(data: CustomerData):
    # Convert to dataframe
    input_data = pd.DataFrame([data.dict()])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_label": "Will Churn" if prediction == 1 else "Will Not Churn",
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}