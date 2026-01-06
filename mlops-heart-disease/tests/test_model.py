import sys
sys.path.insert(0, 'src')

import pytest
from model_training import ModelTrainer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    X, y = make_classification(n_samples=100, n_features=13, n_informative=10,
                              n_redundant=3, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_train_logistic_regression(sample_classification_data):
    """Test Logistic Regression training."""
    X_train, X_test, y_train, y_test = sample_classification_data
    
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_train_random_forest(sample_classification_data):
    """Test Random Forest training."""
    X_train, X_test, y_train, y_test = sample_classification_data
    
    trainer = ModelTrainer()
    model = trainer.train_random_forest(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_evaluate_model(sample_classification_data):
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = sample_classification_data
    
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X_train, y_train)
    metrics = trainer.evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'roc_auc' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())


def test_cross_validate_model(sample_classification_data):
    """Test cross-validation."""
    X_train, X_test, y_train, y_test = sample_classification_data
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X_train, y_train)
    cv_results = trainer.cross_validate_model(model, X_combined, y_combined, cv=3)
    
    assert 'test_accuracy' in cv_results


def test_save_model(sample_classification_data, tmp_path):
    """Test model saving."""
    X_train, X_test, y_train, y_test = sample_classification_data
    
    trainer = ModelTrainer()
    model = trainer.train_logistic_regression(X_train, y_train)
    
    model_path = tmp_path / "test_model.pkl"
    trainer.save_model(model, str(model_path))
    
    assert model_path.exists()