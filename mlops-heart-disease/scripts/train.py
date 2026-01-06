import sys
sys.path.insert(0, 'src')

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load and process data
    processor = DataProcessor('data/raw/heart.csv')
    df = processor.load_data()
    df = processor.preprocess(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.train_test_split(
        df, target_col='target'
    )
    
    # Train models
    trainer = ModelTrainer(mlflow_uri='http://localhost:5000')
    
    # Logistic Regression
    lr_model = trainer.train_logistic_regression(X_train, y_train)
    lr_metrics = trainer.evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    trainer.log_experiment("Logistic Regression", {}, lr_metrics, lr_model)
    trainer.save_model(lr_model, 'models/logistic_regression.pkl')
    
    # Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_metrics = trainer.evaluate_model(rf_model, X_test, y_test, "Random Forest")
    trainer.log_experiment("Random Forest", {}, rf_metrics, rf_model)
    trainer.save_model(rf_model, 'models/random_forest.pkl')
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()