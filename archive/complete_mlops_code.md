# COMPLETE MLOPS CATS VS DOGS PIPELINE - ALL SOURCE CODE

## FILE: src/config.py
```python
import os
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class DataConfig:
    raw_dir: Path = RAW_DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR
    img_height: int = 224
    img_width: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 42
    model_save_path: Path = MODELS_DIR / "best_model.pt"

@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "cats_vs_dogs_classification"
    artifact_uri: str = str(ARTIFACTS_DIR)

def load_params(params_file: str = "params.yaml") -> Dict[str, Any]:
    params_path = PROJECT_ROOT / params_file
    if params_path.exists():
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_config() -> tuple:
    params = load_params()
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    mlflow_cfg = MLflowConfig()
    
    if params:
        for key, val in params.get("data", {}).items():
            if hasattr(data_cfg, key):
                setattr(data_cfg, key, val)
        for key, val in params.get("training", {}).items():
            if hasattr(train_cfg, key):
                setattr(train_cfg, key, val)
        for key, val in params.get("mlflow", {}).items():
            if hasattr(mlflow_cfg, key):
                setattr(mlflow_cfg, key, val)
    
    return data_cfg, train_cfg, mlflow_cfg

DATA_CFG, TRAIN_CFG, MLFLOW_CFG = get_config()
```

## FILE: src/data_prep.py
```python
import random
from typing import Tuple, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

from .config import DATA_CFG

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class CatsDogsDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], transforms_fn=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms_fn
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if self.transforms:
                img = self.transforms(img)
        
        return img, torch.tensor(label, dtype=torch.long)

def get_transforms(augment: bool = False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        return transforms.Compose([
            transforms.Resize((DATA_CFG.img_height, DATA_CFG.img_width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((DATA_CFG.img_height, DATA_CFG.img_width)),
            transforms.ToTensor(),
            normalize
        ])

def collect_image_paths(raw_dir: Path) -> Tuple[List[str], List[int]]:
    cats_dir = raw_dir / "cats"
    dogs_dir = raw_dir / "dogs"
    
    image_paths = []
    labels = []
    
    if cats_dir.exists():
        for img_file in list(cats_dir.glob("*.jpg")) + list(cats_dir.glob("*.png")) + list(cats_dir.glob("*.jpeg")):
            image_paths.append(str(img_file))
            labels.append(0)
    
    if dogs_dir.exists():
        for img_file in list(dogs_dir.glob("*.jpg")) + list(dogs_dir.glob("*.png")) + list(dogs_dir.glob("*.jpeg")):
            image_paths.append(str(img_file))
            labels.append(1)
    
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)
    
    return list(image_paths), list(labels)

def create_train_val_test_split(image_paths: List[str], labels: List[int]) -> Tuple:
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels,
        test_size=(1 - DATA_CFG.train_ratio),
        stratify=labels,
        random_state=42
    )
    
    val_test_ratio = DATA_CFG.val_ratio / (DATA_CFG.val_ratio + DATA_CFG.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_test_ratio),
        stratify=y_temp,
        random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def prepare_data() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    image_paths, labels = collect_image_paths(DATA_CFG.raw_dir)
    
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {DATA_CFG.raw_dir}")
    
    print(f"Found {len(image_paths)} images: {sum(1 for l in labels if l == 0)} cats, {sum(1 for l in labels if l == 1)} dogs")
    
    train_data, val_data, test_data = create_train_val_test_split(image_paths, labels)
    
    train_dataset = CatsDogsDataset(train_data[0], train_data[1], transforms_fn=get_transforms(augment=True))
    val_dataset = CatsDogsDataset(val_data[0], val_data[1], transforms_fn=get_transforms(augment=False))
    test_dataset = CatsDogsDataset(test_data[0], test_data[1], transforms_fn=get_transforms(augment=False))
    
    train_loader = DataLoader(train_dataset, batch_size=DATA_CFG.batch_size, shuffle=True, num_workers=DATA_CFG.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CFG.batch_size, shuffle=False, num_workers=DATA_CFG.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=DATA_CFG.batch_size, shuffle=False, num_workers=DATA_CFG.num_workers, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, ["cat", "dog"]
```

## FILE: src/model.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetBaseline, self).__init__()
        from torchvision.models import resnet18
        
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.model(x)
```

## FILE: src/train.py
```python
import os
import json
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .config import TRAIN_CFG, MLFLOW_CFG, DATA_CFG
from .data_prep import prepare_data
from .model import SimpleCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1, np.array(all_labels), np.array(all_preds)

def plot_training_history(train_losses, val_losses, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Cat', 'Dog'])
    ax.set_yticklabels(['Cat', 'Dog'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)
    plt.savefig(save_path, dpi=100)
    plt.close()

def train():
    # Setup
    device = TRAIN_CFG.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, class_names = prepare_data()
    
    # Initialize model
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CFG.learning_rate, weight_decay=TRAIN_CFG.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # MLflow tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_CFG.tracking_uri)
        mlflow.set_experiment(MLFLOW_CFG.experiment_name)
        run = mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        mlflow.log_params({
            "model": "SimpleCNN",
            "epochs": TRAIN_CFG.epochs,
            "learning_rate": TRAIN_CFG.learning_rate,
            "batch_size": DATA_CFG.batch_size,
            "optimizer": "Adam",
            "image_size": f"{DATA_CFG.img_height}x{DATA_CFG.img_width}"
        })
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses, val_accs = [], [], []
    
    for epoch in range(TRAIN_CFG.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{TRAIN_CFG.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_labels, val_preds = evaluate(
            model, val_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        scheduler.step(val_loss)
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1
            }, step=epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            logger.info(f"Best model saved with val_loss: {val_loss:.4f}")
    
    # Load best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
    
    # Save model
    os.makedirs(TRAIN_CFG.model_save_path.parent, exist_ok=True)
    torch.save(model.state_dict(), TRAIN_CFG.model_save_path)
    logger.info(f"Model saved to {TRAIN_CFG.model_save_path}")
    
    # Generate artifacts
    artifacts_dir = Path(MLFLOW_CFG.artifact_uri)
    artifacts_dir.mkdir(exist_ok=True)
    
    # Plot training history
    history_path = artifacts_dir / "training_history.png"
    plot_training_history(train_losses, val_losses, val_accs, history_path)
    
    # Plot confusion matrix
    cm_path = artifacts_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(test_labels, test_preds, cm_path)
    
    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1)
    }
    
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1
        })
        
        mlflow.log_artifact(str(history_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.pytorch.log_model(model, "model")
        
        mlflow.end_run()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train()
```

## FILE: src/infer.py
```python
import torch
from PIL import Image
from pathlib import Path

from .config import TRAIN_CFG, DATA_CFG
from .model import SimpleCNN

def load_model(model_path=None, device=None):
    if model_path is None:
        model_path = TRAIN_CFG.model_save_path
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SimpleCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image_path, device):
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((DATA_CFG.img_height, DATA_CFG.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def predict(image_path, model_path=None):
    model, device = load_model(model_path)
    image_tensor = preprocess_image(image_path, device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = ["cat", "dog"]
    return {
        "class": class_names[predicted_class],
        "confidence": float(confidence),
        "probabilities": {
            "cat": float(probabilities[0][0]),
            "dog": float(probabilities[0][1])
        }
    }
```

## FILE: src/api.py
```python
import json
import logging
import time
from datetime import datetime
from typing import Optional
import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

from .config import TRAIN_CFG, DATA_CFG, ARTIFACTS_DIR
from .infer import load_model, preprocess_image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Binary image classification API for cats and dogs",
    version="1.0.0"
)

# Load model on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, device = load_model(device=device)

# Metrics
request_count = 0
total_latency = 0.0
logs = []

class HealthResponse(BaseModel):
    status: str
    device: str
    timestamp: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    latency_ms: float
    timestamp: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": device,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Prediction endpoint for image classification"""
    global request_count, total_latency
    
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess and predict
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((DATA_CFG.img_height, DATA_CFG.img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ["cat", "dog"]
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        request_count += 1
        total_latency += latency
        
        # Log request
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "prediction": class_names[predicted_class],
            "confidence": float(confidence),
            "latency_ms": latency,
            "request_count": request_count,
            "avg_latency_ms": total_latency / request_count
        }
        logs.append(log_entry)
        logger.info(json.dumps(log_entry))
        
        return {
            "prediction": class_names[predicted_class],
            "confidence": float(confidence),
            "probabilities": {
                "cat": float(probabilities[0][0]),
                "dog": float(probabilities[0][1])
            },
            "latency_ms": latency,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Get API metrics"""
    if request_count == 0:
        return {"request_count": 0, "avg_latency_ms": 0}
    
    return {
        "request_count": request_count,
        "avg_latency_ms": total_latency / request_count,
        "total_latency_ms": total_latency
    }

@app.get("/logs")
def get_logs(limit: Optional[int] = 100):
    """Get prediction logs"""
    return {"logs": logs[-limit:], "total_logs": len(logs)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## FILE: tests/test_data_prep.py
```python
import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.data_prep import collect_image_paths, create_train_val_test_split

@pytest.fixture
def mock_dataset():
    """Create temporary mock dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create cats directory
        cats_dir = tmpdir / "cats"
        cats_dir.mkdir()
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(i, 0, 0))
            img.save(cats_dir / f"cat_{i}.jpg")
        
        # Create dogs directory
        dogs_dir = tmpdir / "dogs"
        dogs_dir.mkdir()
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(0, i, 0))
            img.save(dogs_dir / f"dog_{i}.jpg")
        
        yield tmpdir

def test_collect_image_paths(mock_dataset):
    """Test image collection"""
    paths, labels = collect_image_paths(mock_dataset)
    
    assert len(paths) == 20
    assert len(labels) == 20
    assert sum(1 for l in labels if l == 0) == 10  # 10 cats
    assert sum(1 for l in labels if l == 1) == 10  # 10 dogs

def test_create_train_val_test_split(mock_dataset):
    """Test data splitting"""
    paths, labels = collect_image_paths(mock_dataset)
    train_data, val_data, test_data = create_train_val_test_split(paths, labels)
    
    train_size = len(train_data[0])
    val_size = len(val_data[0])
    test_size = len(test_data[0])
    
    assert train_size + val_size + test_size == 20
    assert train_size >= 16  # 80% of 20
    assert val_size >= 2    # 10% of 20
    assert test_size >= 2   # 10% of 20
```

## FILE: tests/test_infer.py
```python
import pytest
import torch
import tempfile
from pathlib import Path
from PIL import Image

from src.model import SimpleCNN
from src.infer import load_model, preprocess_image

@pytest.fixture
def trained_model():
    """Create and save a simple trained model"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create model
        model = SimpleCNN(num_classes=2)
        model_path = tmpdir / "test_model.pt"
        torch.save(model.state_dict(), model_path)
        
        yield str(model_path)

@pytest.fixture
def test_image():
    """Create a test image"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        img = Image.new('RGB', (224, 224), color=(255, 0, 0))
        img_path = tmpdir / "test.jpg"
        img.save(img_path)
        
        yield str(img_path)

def test_load_model(trained_model):
    """Test model loading"""
    model, device = load_model(trained_model)
    
    assert isinstance(model, SimpleCNN)
    assert device in ["cpu", "cuda"]
    assert next(model.parameters()).is_cuda == (device == "cuda")

def test_model_forward_pass():
    """Test model forward pass"""
    model = SimpleCNN(num_classes=2)
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 2)

def test_preprocess_image(test_image):
    """Test image preprocessing"""
    device = "cpu"
    tensor = preprocess_image(test_image, device)
    
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == device
```

## FILE: requirements.txt
```
torch==2.3.0
torchvision==0.18.0
pillow==10.1.0
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
scikit-learn==1.5.0
numpy==1.24.3
matplotlib==3.9.0
pyyaml==6.0.2
mlflow==2.16.0
pytest==8.3.0
requests==2.32.0
```

## FILE: Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## FILE: params.yaml
```yaml
data:
  img_height: 224
  img_width: 224
  batch_size: 32
  num_workers: 4
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  epochs: 15
  learning_rate: 0.001
  weight_decay: 0.0001
  device: "cuda"
  seed: 42

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "cats_vs_dogs_classification"
```

## FILE: .github/workflows/ci_cd.yaml
```yaml
name: MLOps CI/CD Pipeline

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository_owner }}/cats-dogs-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Lint with flake8
        run: |
          pip install flake8
          flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
      
      - name: Test with pytest
        run: |
          pytest tests/ -v --tb=short

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: ${{ github.event_name == 'push' }}
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=registry,ref=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-to: type=inline

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/cats-dogs-api \
            cats-dogs-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            --record
      
      - name: Run smoke tests
        run: |
          sleep 10
          bash scripts/smoke_test.sh http://localhost:8000
```

