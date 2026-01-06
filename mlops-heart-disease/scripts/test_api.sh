#!/bin/bash

# Test API endpoints

set -e

# Configuration
API_URL=${1:-"http://localhost:8000"}
HEALTH_ENDPOINT="$API_URL/health"
PREDICT_ENDPOINT="$API_URL/predict"
INFO_ENDPOINT="$API_URL/info"

echo "=== Testing Heart Disease Prediction API ==="
echo "API URL: $API_URL"
echo ""

# Health check
echo "1. Testing Health Endpoint..."
curl -s -X GET "$HEALTH_ENDPOINT" | python -m json.tool
echo ""

# API Info
echo "2. Testing Info Endpoint..."
curl -s -X GET "$INFO_ENDPOINT" | python -m json.tool
echo ""

# Single prediction
echo "3. Testing Prediction Endpoint..."
RESPONSE=$(curl -s -X POST "$PREDICT_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{
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
  }')

echo "$RESPONSE" | python -m json.tool
echo ""

# Extract prediction
PREDICTION=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['prediction'])")
CONFIDENCE=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['confidence'])")

echo "Prediction: $PREDICTION (Confidence: $CONFIDENCE)"
echo ""
echo "=== All Tests Completed Successfully ==="