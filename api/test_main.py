from fastapi.testclient import TestClient
from main import app

# Create a test client that simulates a web browser talking to your API
client = TestClient(app)

def test_health_check():
    """Tests if the health endpoint is returning the correct 200 OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "operational"}

def test_predict_endpoint():
    """Tests if the prediction endpoint accepts data and returns the correct schema."""
    sample_payload = {
        "customer_id": "TEST-999",
        "Recency": 15,
        "Frequency": 12,
        "Monetary_Value": 1500.50,
        "Total_Purchased_Quantity": 300,
        "Unique_Products_Purchased": 45,
        "Average_Order_Value": 125.04
    }
    
    response = client.post("/predict", json=sample_payload)
    
    # 1. Check that the request was successful
    assert response.status_code == 200
    
    # 2. Check that the response contains all the required keys
    data = response.json()
    assert data["customer_id"] == "TEST-999"
    assert "purchase_prediction" in data
    assert "purchase_probability" in data
    assert data["prediction_horizon_days"] == 30