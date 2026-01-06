from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)


def build_full_pipeline(numeric_features: list, 
                       categorical_features: list = None):
    """Build complete preprocessing pipeline."""
    if categorical_features is None:
        categorical_features = []
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ]) if categorical_features else None
    
    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])
    
    logger.info("Built complete preprocessing pipeline")
    return preprocessor