from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
import time
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model'])
prediction_histogram = Histogram('prediction_duration_seconds', 
                               'Prediction duration in seconds', ['model'])
error_counter = Counter('errors_total', 'Total errors', ['error_type'])

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Load models
try:
    lr_model = joblib.load('models/logistic_regression.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")


class PredictionInput(BaseModel):
    """Input schema for prediction."""
    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="Sex (1=male, 0=female)")
    cp: int = Field(..., description="Chest pain type")
    trestbps: float = Field(..., description="Resting blood pressure")
    chol: float = Field(..., description="Serum cholesterol")
    fbs: int = Field(..., description="Fasting blood sugar")
    restecg: int = Field(..., description="Resting ECG")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression")
    slope: int = Field(..., description="Slope of ST segment")
    ca: int = Field(..., description="Coronary calcium")
    thal: int = Field(..., description="Thalassemia")


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    prediction: int
    confidence: float
    model: str
    timestamp: str


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """Make prediction using the best model."""
    try:
        with prediction_histogram.labels(model='random_forest').time():
            # Prepare input
            X = pd.DataFrame([{
                'age': input_data.age,
                'sex': input_data.sex,
                'cp': input_data.cp,
                'trestbps': input_data.trestbps,
                'chol': input_data.chol,
                'fbs': input_data.fbs,
                'restecg': input_data.restecg,
                'thalach': input_data.thalach,
                'exang': input_data.exang,
                'oldpeak': input_data.oldpeak,
                'slope': input_data.slope,
                'ca': input_data.ca,
                'thal': input_data.thal
            }])
            
            # Make prediction using Random Forest (better performance)
            prediction = rf_model.predict(X)[0]
            confidence = float(max(rf_model.predict_proba(X)[0]))
            
            prediction_counter.labels(model='random_forest').inc()
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
            
            return PredictionOutput(
                prediction=int(prediction),
                confidence=confidence,
                model="random_forest",
                timestamp=datetime.utcnow().isoformat()
            )
    
    except Exception as e:
        error_counter.labels(error_type='prediction_error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(inputs: list[PredictionInput]):
    """Batch prediction endpoint."""
    try:
        results = []
        for input_data in inputs:
            result = await predict(input_data)
            results.append(result)
        return {"predictions": results}
    except Exception as e:
        error_counter.labels(error_type='batch_prediction_error').inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.get("/info", tags=["Info"])
async def info():
    """API information endpoint."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "models": ["logistic_regression", "random_forest"],
        "description": "Predicts risk of heart disease based on patient health data"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
