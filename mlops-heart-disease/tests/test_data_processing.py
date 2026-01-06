import sys
sys.path.insert(0, 'src')

import pytest
import pandas as pd
import numpy as np
from data_processing import DataProcessor
from pathlib import Path


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'age': [45, 50, 55, 60, 65],
        'sex': [1, 0, 1, 0, 1],
        'cp': [2, 1, 2, 3, 1],
        'trestbps': [130, 120, 140, 150, 125],
        'chol': [250, 200, 260, 300, 220],
        'target': [0, 1, 1, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def processor(tmp_path):
    """Create DataProcessor instance."""
    csv_file = tmp_path / "test_data.csv"
    data = {
        'age': [45, 50, 55, 60, 65],
        'sex': [1, 0, 1, 0, 1],
        'cp': [2, 1, 2, 3, 1],
        'trestbps': [130, 120, 140, 150, 125],
        'chol': [250, 200, 260, 300, 220],
        'target': [0, 1, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    
    return DataProcessor(str(csv_file))


def test_load_data(processor):
    """Test data loading."""
    df = processor.load_data()
    assert df.shape[0] == 5
    assert 'age' in df.columns


def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    sample_data.loc[0, 'age'] = np.nan
    processor = DataProcessor('dummy_path')
    
    df = processor.handle_missing_values(sample_data)
    assert df.isnull().sum().sum() == 0


def test_remove_outliers(sample_data):
    """Test outlier removal."""
    processor = DataProcessor('dummy_path')
    df = processor.remove_outliers(sample_data)
    
    assert len(df) <= len(sample_data)


def test_scale_features(sample_data):
    """Test feature scaling."""
    processor = DataProcessor('dummy_path')
    
    scaled = processor.scale_features(sample_data.copy(), fit=True)
    
    # Check that features are scaled (mean ~0, std ~1)
    assert abs(scaled['age'].mean()) < 0.1
    assert abs(scaled['age'].std() - 1) < 0.1


def test_train_test_split(sample_data):
    """Test train-test split."""
    processor = DataProcessor('dummy_path')
    
    X_train, X_test, y_train, y_test = processor.train_test_split(
        sample_data, target_col='target', test_size=0.2
    )
    
    assert len(X_train) + len(X_test) == len(sample_data)
    assert X_train.shape[1] == sample_data.shape[1] - 1