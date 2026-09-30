import pytest
import pandas as pd
import numpy as np
from src.train import train_models
from src.evaluate import evaluate_model
from src.explain import get_explainer

@pytest.fixture
def dummy_training_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'monthly_revenue': np.random.uniform(5000, 20000, 100),
        'cash_balance': np.random.uniform(1000, 10000, 100),
        'industry': np.random.choice(['Retail', 'Tech', 'Services'], 100),
        'default_flag': np.random.randint(0, 2, 100)
    })
    return df

def test_model_training_and_evaluation(dummy_training_data):
    """
    Tests Phase 4A machine learning training and evaluation pipeline.
    """
    target_col = 'default_flag'
    numeric_features = ['monthly_revenue', 'cash_balance']
    categorical_features = ['industry']
    
    # Test Training
    artifacts = train_models(
        dummy_training_data, 
        target_col, 
        numeric_features, 
        categorical_features, 
        test_size=0.2
    )
    
    assert 'models' in artifacts
    assert 'logistic_regression' in artifacts['models']
    assert 'random_forest' in artifacts['models']
    assert 'preprocessor' in artifacts
    
    # Test Evaluation
    rf_model = artifacts['models']['random_forest']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    
    metrics = evaluate_model(rf_model, X_test, y_test)
    assert 'roc_auc' in metrics
    assert 'f1' in metrics
    assert 'confusion_matrix' in metrics

def test_explainer_initialization(dummy_training_data):
    """
    Tests Phase 4A explainability module initialization.
    """
    artifacts = train_models(
        dummy_training_data, 
        'default_flag', 
        ['monthly_revenue'], 
        ['industry']
    )
    
    rf_model = artifacts['models']['random_forest']
    explainer = get_explainer(rf_model, X_background=None, model_type='tree')
    
    assert explainer is not None