# MLOps Cats vs Dogs Classification - Complete Project Guide

## Project Structure Overview

```
mlops_cats_dogs/
├── data/
│   ├── raw/                    # Original Kaggle dataset
│   └── processed/              # DVC-tracked preprocessed data
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & paths
│   ├── data_prep.py           # Data loading & preprocessing
│   ├── model.py               # CNN model definition
│   ├── train.py               # Training loop with MLflow
│   ├── evaluate.py            # Evaluation utilities
│   ├── infer.py               # Inference utilities
│   └── api.py                 # FastAPI service
├── tests/
│   ├── __init__.py
│   ├── test_data_prep.py      # Unit tests for data
│   ├── test_infer.py          # Unit tests for inference
│   └── sample_image.jpg       # Test image
├── models/
│   └── best_model.pt          # Trained model weights
├── notebooks/
│   └── eda.ipynb              # EDA notebook
├── kubernetes/
│   ├── deployment.yaml        # K8s deployment
│   ├── service.yaml           # K8s service
│   └── ingress.yaml           # Optional ingress
├── docker-compose/
│   └── docker-compose.yml     # Docker Compose setup
├── .github/workflows/
│   └── ci_cd.yaml             # GitHub Actions CI/CD
├── scripts/
│   ├── download_dataset.py    # Kaggle download script
│   ├── smoke_test.sh          # Post-deploy smoke tests
│   └── monitor.py             # Monitoring script
├── Dockerfile                 # Container image
├── .dockerignore
├── .gitignore
├── dvc.yaml                   # DVC pipeline
├── dvc.lock
├── params.yaml                # Hyperparameters
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation
└── setup.py                   # Package setup

```

## Quick Start Commands

```bash
# 1. Clone and setup
git clone <repo>
cd mlops_cats_dogs
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download dataset from Kaggle
python scripts/download_dataset.py

# 3. Run DVC pipeline (preprocessing)
dvc repro

# 4. Start MLflow tracking server
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 5. Train model
python -m src.train

# 6. Test API locally
uvicorn src.api:app --reload --port 8000

# 7. Build Docker image
docker build -t cats-dogs-api:v1 .

# 8. Run with Docker Compose
docker-compose -f docker-compose/docker-compose.yml up

# 9. Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

## Key Features Implemented

✅ **M1: Model Development & Experiment Tracking**
- Git + DVC versioning for code and datasets
- PyTorch CNN baseline model (SimpleCNN)
- MLflow experiment tracking with artifacts
- Confusion matrices and loss curves logged

✅ **M2: Model Packaging & Containerization**
- FastAPI with /health and /predict endpoints
- Request/response validation with Pydantic
- Pinned dependencies for reproducibility
- Production-grade Dockerfile

✅ **M3: CI Pipeline (GitHub Actions)**
- Automated testing with pytest
- Linting and code quality checks
- Docker image build and push to GHCR
- Triggered on every push/PR to main

✅ **M4: CD Pipeline & Deployment**
- Kubernetes manifests (Deployment + Service)
- Docker Compose for local stack
- Smoke tests post-deployment
- Automatic rollout on image update

✅ **M5: Monitoring & Logging**
- JSON structured logging for all requests
- Request count and latency tracking
- Prometheus metrics endpoint
- Model performance tracking script

