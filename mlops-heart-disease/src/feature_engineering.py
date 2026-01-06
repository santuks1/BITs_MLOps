import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and transformation."""
    
    def __init__(self):
        self.transformers = {}
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                    features_to_interact: list):
        """Create interaction features."""
        df_copy = df.copy()
        
        for i in range(len(features_to_interact)):
            for j in range(i+1, len(features_to_interact)):
                feat1 = features_to_interact[i]
                feat2 = features_to_interact[j]
                
                if feat1 in df_copy.columns and feat2 in df_copy.columns:
                    new_col = f"{feat1}_x_{feat2}"
                    df_copy[new_col] = df_copy[feat1] * df_copy[feat2]
                    logger.info(f"Created interaction feature: {new_col}")
        
        return df_copy
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                   degree: int = 2):
        """Create polynomial features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_cols])
        
        feature_names = poly.get_feature_names_out(numeric_cols)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, 
                              index=df.index)
        
        logger.info(f"Created {poly_df.shape[1]} polynomial features")
        return poly_df
    
    def build_preprocessing_pipeline(self, numeric_features: list, 
                                     categorical_features: list = None):
        """Build preprocessing pipeline with ColumnTransformer."""
        if categorical_features is None:
            categorical_features = []
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
            ]
        )
        
        self.transformers['preprocessor'] = preprocessor
        logger.info("Built preprocessing pipeline")
        
        return preprocessor
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method='correlation', threshold=0.1):
        """Select important features."""
        if method == 'correlation':
            correlations = pd.concat([X, y], axis=1).corr()[y.name]
            important_features = correlations[abs(correlations) > threshold].index.tolist()
            important_features.remove(y.name) if y.name in important_features else None
        
        logger.info(f"Selected {len(important_features)} features")
        return important_features