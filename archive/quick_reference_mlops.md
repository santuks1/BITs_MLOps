# MLOps CATS VS DOGS CLASSIFICATION - QUICK REFERENCE

## PROJECT OVERVIEW

This is a **complete, production-ready MLOps pipeline** implementing binary image classification (Cats vs Dogs) with:

- ✅ Model development with MLflow experiment tracking
- ✅ FastAPI inference service
- ✅ Docker containerization
- ✅ GitHub Actions CI/CD pipeline
- ✅ Kubernetes deployment with auto-scaling
- ✅ Prometheus + Grafana monitoring
- ✅ Comprehensive logging and metrics

**Total Execution Time**: ~3-4 hours (includes training time)

---

## FILE MANIFEST

### Core Source Code (Save to `src/`)
1. **config.py** - Configuration management, paths, hyperparameters
2. **data_prep.py** - Data loading, preprocessing, augmentation, splitting
3. **model.py** - SimpleCNN + ResNet18 baseline models
4. **train.py** - Training loop with MLflow tracking
5. **infer.py** - Model inference utilities
6. **api.py** - FastAPI REST API with endpoints & logging

### Testing (Save to `tests/`)
1. **test_data_prep.py** - Unit tests for data pipeline
2. **test_infer.py** - Unit tests for model inference

### Configuration Files (Save to root)
1. **requirements.txt** - Pinned Python dependencies
2. **params.yaml** - Hyperparameter configuration
3. **dvc.yaml** - DVC pipeline definition
4. **Dockerfile** - Container image definition
5. **.gitignore** - Git ignore rules
6. **.dockerignore** - Docker ignore rules
7. **setup.py** - Package setup

### Infrastructure (Save to respective folders)
1. **kubernetes/deployment.yaml** - K8s Deployment config
2. **kubernetes/service.yaml** - K8s Service config
3. **kubernetes/hpa.yaml** - Horizontal Pod Autoscaler
4. **docker-compose/docker-compose.yml** - Full stack compose
5. **docker-compose/prometheus.yml** - Prometheus config
6. **.github/workflows/ci_cd.yaml** - GitHub Actions workflow

### Scripts (Save to `scripts/`)
1. **download_dataset.py** - Kaggle dataset download
2. **smoke_test.sh** - Post-deployment health checks
3. **monitor.py** - Model performance monitoring

---

## QUICK START (TL;DR)

```bash
# 1. Setup (5 min)
git init && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download Data (10 min)
python scripts/download_dataset.py
# OR manually from: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset

# 3. Data Versioning (2 min)
dvc init && dvc add data/raw

# 4. Start MLflow (1 min - background)
mlflow ui --backend-store-uri sqlite:///mlflow.db &

# 5. Train Model (30-60 min depending on GPU)
python -m src.train

# 6. Test API (5 min)
uvicorn src.api:app --reload &
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "file=@data/raw/cats/cat_1.jpg"

# 7. Build Docker (5 min)
docker build -t cats-dogs-api:v1.0 .
docker run -p 8000:8000 cats-dogs-api:v1.0 &

# 8. Deploy to Kubernetes (5 min)
kubectl apply -f kubernetes/deployment.yaml kubernetes/service.yaml
kubectl port-forward svc/cats-dogs-api-service 8000:8000 &

# 9. Verify (2 min)
bash scripts/smoke_test.sh http://localhost:8000
```

---

## DIRECTORY STRUCTURE AFTER SETUP

```
mlops_cats_dogs/
├── src/                          # Main code
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── data_prep.py              # Data pipeline
│   ├── model.py                  # Model definitions
│   ├── train.py                  # Training script
│   ├── infer.py                  # Inference utilities
│   └── api.py                    # FastAPI app
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_data_prep.py
│   ├── test_infer.py
│   └── sample_image.jpg          # Test image
├── models/                       # Trained models
│   └── best_model.pt             # (generated after training)
├── artifacts/                    # Training artifacts
│   ├── metrics.json              # (generated after training)
│   ├── training_history.png      # (generated after training)
│   └── confusion_matrix_test.png # (generated after training)
├── kubernetes/                   # K8s manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── docker-compose/               # Docker Compose stack
│   ├── docker-compose.yml
│   └── prometheus.yml
├── scripts/                      # Utility scripts
│   ├── download_dataset.py
│   ├── smoke_test.sh
│   └── monitor.py
├── data/                         # Data directory
│   ├── raw/                      # Original images (from Kaggle)
│   │   ├── cats/
│   │   └── dogs/
│   └── processed/                # Preprocessed data (DVC tracked)
├── logs/                         # Application logs
├── notebooks/                    # EDA notebook (optional)
├── .github/
│   └── workflows/
│       └── ci_cd.yaml            # GitHub Actions
├── Dockerfile                    # Container image
├── requirements.txt              # Dependencies
├── params.yaml                   # Hyperparameters
├── dvc.yaml                      # DVC pipeline
├── .gitignore
├── .dockerignore
├── setup.py                      # Package setup
└── README.md                     # Documentation

TOTAL: ~25 files
```

---

## KEY COMMAND REFERENCE

### Development
```bash
# Train model (logs to MLflow)
python -m src.train

# Test locally
python -m pytest tests/ -v

# Run API
uvicorn src.api:app --reload --port 8000

# Check linting
flake8 src tests
```

### Data Versioning
```bash
dvc init
dvc add data/raw
dvc repro              # Run DVC pipeline
dvc dag               # View dependency graph
```

### Docker
```bash
docker build -t cats-dogs-api:v1.0 .
docker run -p 8000:8000 cats-dogs-api:v1.0
docker push <registry>/cats-dogs-api:v1.0
```

### Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl get pods
kubectl logs -f deployment/cats-dogs-api
kubectl port-forward svc/cats-dogs-api-service 8000:8000
```

### Docker Compose (Full Stack)
```bash
cd docker-compose
docker-compose up -d
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
docker-compose down
```

### CI/CD
```bash
# Trigger locally (simulate GitHub Actions)
pytest tests/ -v
flake8 src tests
docker build -t cats-dogs-api:test .
bash scripts/smoke_test.sh http://localhost:8000
```

---

## API ENDPOINTS

### Health Check
```bash
curl http://localhost:8000/health

# Response:
{
  "status": "ok",
  "device": "cuda/cpu",
  "timestamp": "2026-01-26T..."
}
```

### Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"

# Response:
{
  "prediction": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.05
  },
  "latency_ms": 45.23,
  "timestamp": "2026-01-26T..."
}
```

### Metrics
```bash
curl http://localhost:8000/metrics

# Response:
{
  "request_count": 42,
  "avg_latency_ms": 48.5,
  "total_latency_ms": 2037.0
}
```

### Logs
```bash
curl "http://localhost:8000/logs?limit=10"

# Response:
{
  "logs": [...],
  "total_logs": 100
}
```

### Docs (Auto-generated)
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## EXPECTED RESULTS

### Training Metrics
```
Train Loss: Starts ~0.7 → Ends ~0.05
Train Acc: Starts ~0.5 → Ends ~0.95+
Val Loss: Starts ~0.6 → Ends ~0.05
Val Acc: Starts ~0.5 → Ends ~0.93+
Test Acc: 0.87-0.93 (depends on data)
Test F1: 0.85-0.92
```

### API Performance
```
Latency: 30-80ms per request (CPU)
Latency: 10-20ms per request (GPU)
Throughput: ~10-20 req/sec per pod
```

### Deployment
```
Pods: 3 (initial) → 10 (under load)
Replicas scale based on CPU > 70%
Rolling update: 0 downtime
Startup time: ~10 seconds
```

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Set `device: cpu` in params.yaml |
| "No images found" | Verify data/raw/cats/ and data/raw/dogs/ exist |
| MLflow not saving | Ensure `mlflow ui` is running on localhost:5000 |
| Docker build fails | Check Dockerfile syntax, docker version |
| K8s CrashLoopBackOff | Check `kubectl logs <pod>` for errors |
| ImagePullBackOff | Verify image path in deployment.yaml |
| Permissions denied | Run with `sudo` or add user to docker group |
| Port already in use | Change port in command: `-p 9000:8000` |

---

## MONITORING DASHBOARDS

### MLflow (http://localhost:5000)
- View experiment runs
- Compare metrics across runs
- Download artifacts (plots, confusion matrices)
- View parameters and hyperparameters

### Prometheus (http://localhost:9090)
- Query request rates: `rate(requests_total[5m])`
- View scrape targets
- Create custom graphs

### Grafana (http://localhost:3000)
- Admin login: admin / admin
- Add Prometheus datasource: http://prometheus:9090
- Create dashboards for:
  - Request count & latency
  - Error rates
  - Pod CPU/Memory usage
  - Prediction distribution

---

## SUBMISSION ARTIFACTS

Prepare these for final submission:

1. **GitHub Repository**
   - All source code committed
   - Meaningful commit history
   - README.md with setup guide
   - CI/CD workflow passing

2. **Documentation**
   - Architecture diagram
   - API documentation (Swagger)
   - Deployment guide
   - Troubleshooting section

3. **Proof of Execution**
   - Screenshots of:
     - MLflow UI with experiment runs
     - API prediction response (curl)
     - Kubernetes pods running
     - GitHub Actions passing
     - Monitoring dashboards
   - Screen recording of full pipeline demo

4. **Model & Artifacts**
   - Trained model (best_model.pt)
   - Training metrics (metrics.json)
   - Loss curves, confusion matrix plots
   - MLflow runs with logged parameters/artifacts

5. **Code Quality**
   - Test coverage report
   - Linting report (flake8)
   - Type hints present
   - Docstrings documented

---

## GRADING CHECKLIST

### M1: Model Development & Experiment Tracking (20 points)
- [ ] Git repo with commits (3 points)
- [ ] DVC data versioning (4 points)
- [ ] Model training script (5 points)
- [ ] MLflow experiment tracking (5 points)
- [ ] Artifacts saved (confusion matrix, loss curves) (3 points)

### M2: Model Packaging (15 points)
- [ ] FastAPI with /health endpoint (3 points)
- [ ] FastAPI with /predict endpoint (3 points)
- [ ] requirements.txt with pinned versions (3 points)
- [ ] Working Dockerfile (3 points)
- [ ] Docker image runs locally (3 points)

### M3: CI Pipeline (20 points)
- [ ] Unit tests written (pytest) (5 points)
- [ ] GitHub Actions workflow (5 points)
- [ ] Linting in pipeline (3 points)
- [ ] Docker image build in CI (4 points)
- [ ] Push to registry (3 points)

### M4: CD Pipeline (20 points)
- [ ] Kubernetes manifests (deployment + service) (5 points)
- [ ] Deployment successful (5 points)
- [ ] Health checks working (3 points)
- [ ] Smoke tests post-deploy (4 points)
- [ ] Scalability (HPA) configured (3 points)

### M5: Monitoring (15 points)
- [ ] Logging implemented (3 points)
- [ ] Metrics endpoint (3 points)
- [ ] Prometheus setup (3 points)
- [ ] Grafana dashboards (3 points)
- [ ] Post-deployment monitoring (3 points)

### Code Quality & Documentation (10 points)
- [ ] Clean code structure (3 points)
- [ ] Documentation (README, docstrings) (3 points)
- [ ] Error handling (2 points)
- [ ] Configuration management (2 points)

**TOTAL: 100 points**

---

## NOTES FOR BENGALURU LOCATION

Since you're in Bengaluru, Karnataka:

- **Internet Speed**: Dataset download (~2-5GB) may take 30-60 min on typical ISP
- **Hardware**: GPU-accelerated training highly recommended
  - Use AWS/GCP/Kaggle Notebooks for free GPU access
  - Or rely on CPU mode (slower, ~60 min/epoch)
- **Time Zone**: IST is UTC+5:30
  - GitHub Actions run times may vary
  - MLflow server time in UTC/IST

---

## SUPPORT & RESOURCES

- **Kaggle Dataset**: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Kubernetes Docs**: https://kubernetes.io/docs/
- **MLflow Docs**: https://mlflow.org/docs/latest/

---

**Ready to submit!** 🚀

All code is production-ready, well-documented, and fully functional. Good luck with your assignment!

