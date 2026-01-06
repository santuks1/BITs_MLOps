import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, 'api')

from app import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_info():
    """Test info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    assert "models" in response.json()


def test_predict_valid_input():
    """Test prediction with valid input."""
    input_data = {
        "age": 50,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 140,
        "exang": 1,
        "oldpeak": 0.5,
        "slope": 2,
        "ca": 1,
        "thal": 3
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert result["prediction"] in [0, 1]
    assert 0 <= result["confidence"] <= 1


def test_predict_missing_field():
    """Test prediction with missing required field."""
    input_data = {
        "age": 50,
        "sex": 1,
        # Missing other required fields
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Validation error


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predictions_total" in response.text