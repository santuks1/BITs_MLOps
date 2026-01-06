import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing."""
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self):
        """Load dataset from CSV."""
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame, strategy='mean') -> pd.DataFrame:
        """Handle missing values."""
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing > 0:
            logger.info(f"Found {initial_missing} missing values")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if strategy == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method='iqr', threshold=1.5):
        """Remove outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        initial_rows = len(df)
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} outliers")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit=True):
        """Encode categorical variables."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
                logger.info(f"Encoded {col}")
            else:
                if col in self.encoders:
                    df[col] = self.encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit=True, 
                      exclude_cols=None):
        """Standardize numeric features."""
        if exclude_cols is None:
            exclude_cols = []
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"Fitted scaler on {len(numeric_cols)} features")
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit=True):
        """Complete preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.encode_categorical(df, fit=fit)
        
        target_col = 'target' if 'target' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != target_col]
        
        df[feature_cols] = self.scale_features(
            df[feature_cols], fit=fit, exclude_cols=[target_col]
        )
        
        logger.info("Preprocessing completed")
        return df
    
    def train_test_split(self, df: pd.DataFrame, target_col: str, 
                        test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
