# COMPREHENSIVE SETUP & EXECUTION GUIDE

## STEP-BY-STEP EXECUTION INSTRUCTIONS

### PHASE 1: LOCAL SETUP (30 minutes)

#### Step 1.1: Clone and Setup Project
```bash
# Create project directory
mkdir mlops_cats_dogs
cd mlops_cats_dogs

# Initialize git
git init
git config user.email "your@email.com"
git config user.name "Your Name"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 1.2: Copy Source Files
```bash
# Create directory structure
mkdir -p src tests notebooks kubernetes docker-compose/data/.gitkeep scripts logs artifacts models

# Download all Python files from complete_mlops_code.md and k8s_docker_scripts_files.md
# Place them in their respective directories:
# - src/config.py
# - src/data_prep.py
# - src/model.py
# - src/train.py
# - src/infer.py
# - src/api.py
# - tests/test_data_prep.py
# - tests/test_infer.py
# - kubernetes/*.yaml
# - docker-compose/docker-compose.yml
# - docker-compose/prometheus.yml
# - scripts/*.py and *.sh
# - Dockerfile, requirements.txt, params.yaml, dvc.yaml, .gitignore
```

#### Step 1.3: Create __init__.py files
```bash
touch src/__init__.py
touch tests/__init__.py
```

#### Step 1.4: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 1.5: Download Dataset
```bash
# Install Kaggle CLI
pip install kaggle

# Download kaggle.json from https://www.kaggle.com/account
# Place at ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
python scripts/download_dataset.py

# OR manually download from:
# https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset
# Extract to data/raw/ with structure:
# data/raw/
#   ├── cats/
#   │   ├── cat_1.jpg
#   │   └── ...
#   └── dogs/
#       ├── dog_1.jpg
#       └── ...
```

---

### PHASE 2: M1 - MODEL DEVELOPMENT & EXPERIMENT TRACKING (45 minutes)

#### Step 2.1: Initialize DVC
```bash
# Initialize DVC
dvc init

# Track raw data
dvc add data/raw

# Commit to git
git add .gitignore data/raw.dvc .dvc/
git commit -m "Initial DVC setup with raw data"
```

#### Step 2.2: Start MLflow Tracking Server
```bash
# In a separate terminal/tmux session
mkdir -p mlruns
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
# Access at http://localhost:5000
```

#### Step 2.3: Train Model
```bash
# In main terminal
python -m src.train

# Monitor training progress and logs
# Model will be saved to models/best_model.pt
# Artifacts saved to artifacts/
# MLflow logs visible at http://localhost:5000
```

#### Step 2.4: Run DVC Pipeline
```bash
# Reproduce entire pipeline (optional)
dvc repro

# View DAG
dvc dag

# Commit changes
git add dvc.lock params.yaml
git commit -m "Complete training pipeline"
```

---

### PHASE 3: M2 - MODEL PACKAGING & CONTAINERIZATION (30 minutes)

#### Step 3.1: Test Inference Locally
```bash
# Test inference module
python -c "
from src.infer import predict
result = predict('data/raw/cats/cat_1.jpg')
print(result)
"
```

#### Step 3.2: Test API Locally (First Terminal)
```bash
# Start FastAPI development server
uvicorn src.api:app --reload --port 8000

# Check logs for startup
# API will reload on code changes
```

#### Step 3.3: Test API Endpoints (Second Terminal)
```bash
# Health check
curl http://localhost:8000/health

# Test prediction with sample image
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/raw/cats/cat_1.jpg"

# Get metrics
curl http://localhost:8000/metrics

# Get logs
curl "http://localhost:8000/logs?limit=10"
```

#### Step 3.4: Build Docker Image
```bash
# Build image locally
docker build -t cats-dogs-api:v1.0 .

# Test Docker image
docker run -p 8000:8000 --name cats-dogs-api-test cats-dogs-api:v1.0 &

# Test endpoints (same as Step 3.3)
curl http://localhost:8000/health

# Stop container
docker stop cats-dogs-api-test
docker rm cats-dogs-api-test
```

#### Step 3.5: Push to Registry (Optional)
```bash
# Login to Docker Hub
docker login

# Tag image
docker tag cats-dogs-api:v1.0 <your-username>/cats-dogs-api:v1.0

# Push
docker push <your-username>/cats-dogs-api:v1.0

# OR push to GitHub Container Registry
docker login ghcr.io -u <username> -p <token>
docker tag cats-dogs-api:v1.0 ghcr.io/<username>/cats-dogs-api:v1.0
docker push ghcr.io/<username>/cats-dogs-api:v1.0
```

---

### PHASE 4: M3 - CI PIPELINE (20 minutes)

#### Step 4.1: Run Local Tests
```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
# or xdg-open htmlcov/index.html  # Linux
```

#### Step 4.2: Lint Code
```bash
pip install flake8 black

# Check style
flake8 src tests --max-line-length=100

# Auto-format
black src tests

# Check again
flake8 src tests
```

#### Step 4.3: Setup GitHub Actions (If using GitHub)
```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy .github/workflows/ci_cd.yaml from k8s_docker_scripts_files.md

# Configure GitHub secrets:
# 1. Go to your repo Settings → Secrets and variables → Actions
# 2. Add GITHUB_TOKEN (auto-created)
# 3. Optionally add other registry credentials

# Push to GitHub
git add .github/
git commit -m "Add CI/CD workflow"
git push origin main
```

#### Step 4.4: Verify CI Pipeline
```bash
# On GitHub, go to Actions tab
# Watch workflow execution
# Confirm linting passes
# Confirm tests pass
# Confirm Docker image builds
```

---

### PHASE 5: M4 - CD PIPELINE & DEPLOYMENT (40 minutes)

#### Step 5.1: Local Deployment with Docker Compose
```bash
# Start full stack (API + MLflow + Prometheus + Grafana)
cd docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f api

# Verify services
curl http://localhost:8000/health      # API
curl http://localhost:5000/            # MLflow
curl http://localhost:9090/            # Prometheus  
curl http://localhost:3000/            # Grafana

# Stop stack
docker-compose down
```

#### Step 5.2: Kubernetes Deployment (Local Cluster)
```bash
# Install kubectl
# macOS: brew install kubectl
# Linux: sudo apt-get install kubectl
# Windows: choco install kubernetes-cli

# Setup local K8s cluster
# Option 1: Docker Desktop (enable Kubernetes)
# Option 2: minikube
minikube start
minikube docker-env  # Use Docker daemon in Kubernetes

# Build image in minikube
docker build -t cats-dogs-api:v1.0 .

# Update deployment.yaml: replace REPLACE_OWNER
sed -i 's/REPLACE_OWNER/<your-username>/g' kubernetes/deployment.yaml

# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/cats-dogs-api

# Port forward to local machine
kubectl port-forward svc/cats-dogs-api-service 8000:8000

# Test endpoints
curl http://localhost:8000/health
```

#### Step 5.3: Smoke Tests
```bash
# Run smoke tests post-deployment
bash scripts/smoke_test.sh http://localhost:8000

# Expected output:
# ✓ API is ready
# ✓ Health check passed
# ✓ Metrics endpoint works
# ✓ All smoke tests passed!
```

#### Step 5.4: Scaling and Rolling Updates
```bash
# Scale replicas
kubectl scale deployment cats-dogs-api --replicas=5

# Check scaling
kubectl get pods

# Rolling update with new image
kubectl set image deployment/cats-dogs-api \
  cats-dogs-api=ghcr.io/<owner>/cats-dogs-api:v1.1 \
  --record

# Monitor rollout
kubectl rollout status deployment/cats-dogs-api

# Rollback if needed
kubectl rollout undo deployment/cats-dogs-api
```

---

### PHASE 6: M5 - MONITORING & LOGGING (30 minutes)

#### Step 6.1: View Application Logs
```bash
# From Docker Compose
docker-compose logs -f api

# From Kubernetes
kubectl logs -f deployment/cats-dogs-api

# Grep specific events
kubectl logs deployment/cats-dogs-api | grep "prediction"
```

#### Step 6.2: Access MLflow Tracking
```bash
# Open MLflow UI
# http://localhost:5000

# View experiments: "cats_vs_dogs_classification"
# Compare runs
# Review artifacts (training_history.png, confusion_matrix_test.png)
# Check metrics over time
```

#### Step 6.3: Monitor with Prometheus & Grafana
```bash
# Prometheus
# http://localhost:9090

# Create a query:
# Sum(rate(requests_total[5m]))  <- Requests per 5 minutes

# Grafana
# http://localhost:3000 (admin/admin)

# Add Prometheus as datasource:
# URL: http://prometheus:9090

# Create dashboard with:
# - Request count
# - Average latency
# - Error rate
# - Pod CPU/Memory usage
```

#### Step 6.4: Collect Post-Deployment Metrics
```bash
# Generate predictions
for i in {1..50}; do
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@data/raw/cats/cat_1.jpg" &
done
wait

# Analyze predictions
python scripts/monitor.py

# View report
cat monitoring_report.json
```

---

## TESTING CHECKLIST

### Local Testing (Before Deployment)
- [ ] Data loading works (10+ images detected)
- [ ] Model training completes (no CUDA/OOM errors)
- [ ] Loss decreases over epochs
- [ ] Validation accuracy improves
- [ ] Test metrics logged to artifacts/
- [ ] MLflow UI shows experiment run
- [ ] API starts without errors
- [ ] Health endpoint returns 200
- [ ] Prediction endpoint accepts image
- [ ] Prediction returns correct JSON format
- [ ] Metrics endpoint shows request count
- [ ] Unit tests all pass (pytest)
- [ ] Linting passes (flake8)

### Docker Testing
- [ ] Docker build completes
- [ ] Docker image runs locally
- [ ] Container health check passes
- [ ] Prediction works in container
- [ ] Environment variables loaded
- [ ] Logs visible via docker logs

### Kubernetes Testing
- [ ] Deployment created successfully
- [ ] Pods reach Running state
- [ ] Service created with loadbalancer
- [ ] Liveness probe passes
- [ ] Readiness probe passes
- [ ] Port-forward works
- [ ] API accessible via service
- [ ] Horizontal Pod Autoscaler scales up under load
- [ ] Smoke tests pass post-deploy

### Production Readiness
- [ ] No hardcoded credentials
- [ ] All configs via params.yaml or env vars
- [ ] Graceful error handling
- [ ] Request/response logging
- [ ] Metrics exposed via /metrics
- [ ] Health checks working
- [ ] Model versioning tracked
- [ ] Data versioned with DVC
- [ ] CI/CD pipeline green
- [ ] Code committed to git

---

## TROUBLESHOOTING

### Problem: "No images found in data/raw"
**Solution:** 
```bash
# Verify directory structure
ls -la data/raw/cats/ data/raw/dogs/

# Expected: Each directory should have .jpg files
# Download from Kaggle if missing
```

### Problem: CUDA out of memory
**Solution:**
```bash
# In params.yaml, change:
training:
  device: "cpu"  # Use CPU instead
  
# Or reduce batch size:
data:
  batch_size: 16  # From 32
```

### Problem: MLflow not storing experiments
**Solution:**
```bash
# Ensure MLflow backend is running
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Check mlruns directory created
ls -la mlruns/

# Verify config in src/train.py uses same URI
```

### Problem: Docker image too large
**Solution:**
```bash
# Use slim base image
FROM python:3.11-slim

# Reduce layer count
# Combine RUN commands with &&

# Clean package managers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Rebuild
docker build --no-cache -t cats-dogs-api:v1.0 .
```

### Problem: Kubernetes ImagePullBackOff
**Solution:**
```bash
# Verify image exists in registry
docker images | grep cats-dogs

# Push to accessible registry
docker tag cats-dogs-api:v1.0 ghcr.io/<owner>/cats-dogs-api:v1.0
docker push ghcr.io/<owner>/cats-dogs-api:v1.0

# Update deployment.yaml with correct image
# Redeploy
kubectl apply -f kubernetes/deployment.yaml
```

---

## FINAL SUBMISSION CHECKLIST

Before submitting your assignment, ensure you have:

✅ **M1: Model Development**
- [ ] Git repo initialized with commits
- [ ] DVC tracking data/raw and data/processed
- [ ] SimpleCNN model trained and saved
- [ ] MLflow runs logged with parameters/metrics
- [ ] Artifacts: loss curves, confusion matrix
- [ ] Train/val/test split documented

✅ **M2: Packaging**
- [ ] FastAPI server with /health and /predict endpoints
- [ ] requirements.txt with pinned versions
- [ ] Dockerfile working (no build errors)
- [ ] API tested locally with curl
- [ ] Request/response validation working

✅ **M3: CI Pipeline**
- [ ] Unit tests for data_prep and infer (pytest)
- [ ] GitHub Actions workflow defined
- [ ] Linting passes (flake8)
- [ ] Docker image builds in CI
- [ ] Image pushed to registry

✅ **M4: CD Pipeline**
- [ ] Kubernetes manifests (deployment + service)
- [ ] Smoke test script working
- [ ] Deployment successful
- [ ] Pod health checks passing
- [ ] API accessible via service

✅ **M5: Monitoring**
- [ ] Logging implemented in api.py
- [ ] Metrics endpoint (/metrics) working
- [ ] Request/latency tracking active
- [ ] Docker Compose stack with monitoring
- [ ] Prometheus/Grafana dashboards

✅ **Documentation**
- [ ] README.md with setup instructions
- [ ] Architecture diagram/description
- [ ] API documentation (FastAPI Docs at /docs)
- [ ] Deployment guide
- [ ] Troubleshooting section

✅ **Code Quality**
- [ ] No hardcoded paths (use config.py)
- [ ] Error handling for edge cases
- [ ] Type hints in functions
- [ ] Docstrings for modules/classes
- [ ] Clean git history with meaningful commits

✅ **Demonstration**
- [ ] Screen recording of full pipeline
- [ ] Screenshots of:
  - MLflow experiment runs
  - API health/prediction responses
  - Kubernetes pods running
  - Monitoring dashboard
  - GitHub Actions passing

---

## EXPECTED METRICS

After successful training, you should see:

```
Train Loss: 0.xx → 0.0x  (decreasing)
Train Accuracy: 0.xx → 0.9x (increasing)
Val Loss: 0.xx → 0.0x (decreasing)
Val Accuracy: 0.xx → 0.9x (increasing)
Test Accuracy: 0.85+ (goal)
Test F1-Score: 0.85+
API Latency: <100ms per request
```

---

## NEXT STEPS (Optional Enhancements)

1. **Transfer Learning**: Use pretrained ResNet18 from src/model.py
2. **Advanced Monitoring**: Integrate Datadog or New Relic
3. **A/B Testing**: Deploy two models, route % of traffic
4. **Model Explainability**: Add LIME/SHAP visualizations
5. **CI/CD Enhancements**: Add performance regression tests
6. **Infrastructure as Code**: Use Terraform for AWS/GCP deployment
7. **Automated Retraining**: Schedule periodic model retraining
8. **Feature Store**: Integrate Feast for feature management

