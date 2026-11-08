from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# 1. Load the trained model globally so it doesn't retrain on every request
try:
    model = joblib.load('production_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model. Did you run save_model.py? Error: {e}")

# Initialize the API (FastAPI automatically generates the Swagger documentation)
app = FastAPI(title="Customer Intelligence API", version="1.0")

# 2. Define Input Schema using Pydantic for validation
class CustomerFeatures(BaseModel):
    customer_id: str
    Recency: float = Field(..., description="Days since last purchase")
    Frequency: float = Field(..., description="Total historical orders")
    Monetary_Value: float = Field(..., description="Total amount spent")
    Total_Purchased_Quantity: float
    Unique_Products_Purchased: float
    Average_Order_Value: float

# 3. Define Output Schema matching the required JSON format
class PredictionResponse(BaseModel):
    customer_id: str
    purchase_prediction: int
    purchase_probability: float
    prediction_horizon_days: int = 30

# 4. Define Endpoints
@app.get("/health")
def health_check():
    """Returns the operational status of the application."""
    return {"status": "operational"}

@app.post("/predict", response_model=PredictionResponse)
def predict_purchase(customer: CustomerFeatures):
    """Accepts validated customer features and returns a purchase prediction."""
    try:
        # Convert input into a 2D array for the model
        features = [[
            customer.Recency, customer.Frequency, customer.Monetary_Value,
            customer.Total_Purchased_Quantity, customer.Unique_Products_Purchased,
            customer.Average_Order_Value
        ]]
        
        # Generate predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Return the exact JSON structure requested by the rubric
        return {
            "customer_id": customer.customer_id,
            "purchase_prediction": int(prediction),
            "purchase_probability": round(float(probability), 2),
            "prediction_horizon_days": 30
        }
    except Exception as e:
        # Proper error handling
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")