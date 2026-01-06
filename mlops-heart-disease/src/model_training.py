from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            roc_auc_score, f1_score, confusion_matrix)
import mlflow
import mlflow.sklearn
import logging
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation."""
    
    def __init__(self, mlflow_uri: str = None):
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
    
    def train_logistic_regression(self, X_train, y_train, 
                                 hyperparams: dict = None):
        """Train Logistic Regression model."""
        if hyperparams is None:
            hyperparams = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
            }
        
        model = LogisticRegression(**hyperparams)
        model.fit(X_train, y_train)
        
        logger.info("Trained Logistic Regression model")
        return model
    
    def train_random_forest(self, X_train, y_train, 
                           hyperparams: dict = None):
        """Train Random Forest model."""
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        
        model = RandomForestClassifier(**hyperparams)
        model.fit(X_train, y_train)
        
        logger.info("Trained Random Forest model")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = "model"):
        """Evaluate model performance."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Evaluation metrics for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv=5):
        """Perform k-fold cross-validation."""
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
        )
        
        logger.info(f"Cross-validation results (cv={cv}):")
        for metric in ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']:
            scores = cv_results[f'test_{metric}']
            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model, X_train, y_train, 
                             param_grid: dict, cv=5):
        """Perform hyperparameter tuning with GridSearchCV."""
        grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def log_experiment(self, model_name: str, params: dict, metrics: dict, 
                      model=None, artifacts_path: str = None):
        """Log experiment to MLflow."""
        with mlflow.start_run():
            mlflow.set_tag("model_type", model_name)
            
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)
            
            if model is not None:
                mlflow.sklearn.log_model(model, "model")
            
            if artifacts_path:
                mlflow.log_artifact(artifacts_path)
            
            logger.info(f"Logged experiment for {model_name}")
    
    def save_model(self, model, path: str):
        """Save model to disk."""
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")