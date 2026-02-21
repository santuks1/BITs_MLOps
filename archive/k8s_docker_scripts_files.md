# KUBERNETES, DOCKER-COMPOSE, SCRIPTS & DVC FILES

## FILE: kubernetes/deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cats-dogs-api
  labels:
    app: cats-dogs-api
    version: v1
spec:
  replicas: 3
  revisionHistoryLimit: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: cats-dogs-api
  template:
    metadata:
      labels:
        app: cats-dogs-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: cats-dogs-api
        image: ghcr.io/REPLACE_OWNER/cats-dogs-api:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        env:
        - name: DEVICE
          value: "cpu"
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-volume
        configMap:
          name: cats-dogs-model
```

## FILE: kubernetes/service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: cats-dogs-api-service
  labels:
    app: cats-dogs-api
spec:
  type: LoadBalancer
  selector:
    app: cats-dogs-api
  ports:
  - name: http
    port: 8000
    targetPort: http
    protocol: TCP
```

## FILE: kubernetes/hpa.yaml
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cats-dogs-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cats-dogs-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## FILE: docker-compose/docker-compose.yml
```yaml
version: '3.9'

services:
  api:
    build:
      context: ..
      dockerfile: Dockerfile
    container_name: cats-dogs-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ../models:/app/models:ro
      - ../artifacts:/app/artifacts
    depends_on:
      - mlflow
    networks:
      - mlops-network
    restart: unless-stopped

  mlflow:
    image: python:3.11-slim
    container_name: mlflow-server
    working_dir: /mlflow
    command: >
      bash -c "pip install mlflow && 
               mlflow ui --backend-store-uri sqlite:////mlflow/mlflow.db 
               --default-artifact-root /mlflow/artifacts 
               --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    networks:
      - mlops-network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - mlops-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - mlops-network
    restart: unless-stopped

volumes:
  mlflow_data:
  prometheus_data:
  grafana_data:

networks:
  mlops-network:
    driver: bridge
```

## FILE: docker-compose/prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'cats-dogs-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

## FILE: scripts/download_dataset.py
```python
#!/usr/bin/env python3
"""
Download Cats and Dogs dataset from Kaggle
Make sure you have kaggle API installed: pip install kaggle
"""

import os
import subprocess
from pathlib import Path
import zipfile

from src.config import RAW_DATA_DIR

def download_dataset():
    """Download dataset from Kaggle"""
    
    # Check if kaggle credentials exist
    kaggle_dir = Path.home() / ".kaggle"
    if not (kaggle_dir / "kaggle.json").exists():
        print("Error: Kaggle credentials not found at ~/.kaggle/kaggle.json")
        print("Please download from https://www.kaggle.com/account and place kaggle.json")
        return False
    
    print("Downloading Cats and Dogs dataset from Kaggle...")
    
    try:
        # Download dataset
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "bhavikjikadara/dog-and-cat-classification-dataset",
            "-p", str(RAW_DATA_DIR)
        ], check=True)
        
        # Extract dataset
        print("Extracting dataset...")
        zip_path = RAW_DATA_DIR / "dog-and-cat-classification-dataset.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DATA_DIR)
            zip_path.unlink()
        
        print("✓ Dataset downloaded and extracted successfully!")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
```

## FILE: scripts/smoke_test.sh
```bash
#!/bin/bash

API_URL=${1:-"http://localhost:8000"}
TIMEOUT=30
RETRIES=5

echo "Running smoke tests against $API_URL"

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 $RETRIES); do
    if curl -s "$API_URL/health" > /dev/null; then
        echo "✓ API is ready"
        break
    fi
    if [ $i -eq $RETRIES ]; then
        echo "✗ API failed to start"
        exit 1
    fi
    sleep $((TIMEOUT / RETRIES))
done

# Test health endpoint
echo "Testing /health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    echo "Response: $HEALTH"
    exit 1
fi

# Test metrics endpoint
echo "Testing /metrics endpoint..."
METRICS=$(curl -s "$API_URL/metrics")
if [ ! -z "$METRICS" ]; then
    echo "✓ Metrics endpoint works"
else
    echo "✗ Metrics endpoint failed"
    exit 1
fi

echo "✓ All smoke tests passed!"
exit 0
```

## FILE: scripts/monitor.py
```python
#!/usr/bin/env python3
"""
Monitor deployed model performance
"""

import json
import time
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_predictions(api_url, num_samples=100):
    """Simulate collecting predictions and ground truth"""
    import requests
    from PIL import Image
    import io
    
    predictions = []
    
    # This would normally fetch actual data
    # For now, we simulate with random data
    for i in range(num_samples):
        logger.info(f"Collected sample {i+1}/{num_samples}")
        time.sleep(0.1)
    
    return predictions

def analyze_performance(predictions):
    """Analyze model performance on collected data"""
    if not predictions:
        logger.warning("No predictions to analyze")
        return {}
    
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("correct", False))
    accuracy = correct / total if total > 0 else 0
    
    analysis = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_predictions": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_confidence": sum(p.get("confidence", 0) for p in predictions) / total if total > 0 else 0
    }
    
    return analysis

def main():
    logger.info("Starting model monitoring...")
    
    api_url = "http://localhost:8000"
    
    # Collect predictions
    logger.info(f"Collecting predictions from {api_url}...")
    predictions = collect_predictions(api_url, num_samples=100)
    
    # Analyze performance
    logger.info("Analyzing performance...")
    analysis = analyze_performance(predictions)
    
    # Save analysis
    output_file = Path("monitoring_report.json")
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Monitoring report saved to {output_file}")
    logger.info(f"Accuracy: {analysis.get('accuracy', 0):.4f}")
    logger.info(f"Avg Confidence: {analysis.get('avg_confidence', 0):.4f}")

if __name__ == "__main__":
    main()
```

## FILE: dvc.yaml
```yaml
stages:
  preprocess:
    cmd: python -m src.data_prep
    deps:
      - src/data_prep.py
      - src/config.py
      - data/raw
    params:
      - data
    outs:
      - data/processed:
          cache: true

  train:
    cmd: python -m src.train
    deps:
      - src/data_prep.py
      - src/train.py
      - src/model.py
      - src/config.py
      - data/processed
    params:
      - training
      - data
    outs:
      - models/best_model.pt:
          cache: true
    metrics:
      - artifacts/metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.train
    deps:
      - models/best_model.pt
      - src/infer.py
    metrics:
      - artifacts/metrics.json:
          cache: false
```

## FILE: .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Data and models
data/raw/
data/processed/
models/*.pt
artifacts/
logs/

# DVC
.dvc/
.dvc.lock
dvc.lock

# MLflow
mlflow.db
mlruns/
.mlflow

# Docker
.dockerignore

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Miscellaneous
*.tmp
*.log
```

## FILE: .dockerignore
```
.git
.gitignore
.dvc
.dvc.lock
dvc.lock
.env
.vscode
.idea
__pycache__
*.pyc
*.pyo
*.egg-info
.pytest_cache
mlflow.db
mlruns/
logs/
artifacts/
data/raw/
.DS_Store
README.md
.github
notebooks/
```

## FILE: setup.py
```python
from setuptools import setup, find_packages

setup(
    name="cats_dogs_mlops",
    version="1.0.0",
    description="MLOps pipeline for cats vs dogs classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.24.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0",
        "mlflow>=2.0.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "flake8>=6.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ]
    },
)
```

