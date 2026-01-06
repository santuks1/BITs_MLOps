import pandas as pd
import urllib.request
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Heart Disease UCI Dataset URL
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def download_dataset(output_path: str = "data/raw/heart.csv"):
    """Download Heart Disease UCI dataset."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset from {DATASET_URL}...")
    
    try:
        urllib.request.urlretrieve(DATASET_URL, "temp_heart.data")
        
        # Define column names
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        # Load data with ? as missing value indicator
        df = pd.read_csv("temp_heart.data", header=None, names=columns, 
                        na_values='?')
        
        # Convert target (0 = no disease, 1+ = disease) to binary
        df['target'] = (df['target'] > 0).astype(int)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['target'].value_counts()}")
        
        # Cleanup
        import os
        os.remove("temp_heart.data")
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    download_dataset()