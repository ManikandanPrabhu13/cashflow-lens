import pytest
import numpy as np
import pandas as pd
from src.fairness import calculate_fairness_metrics, fallback_metrics_by_group

def test_fairness_metrics_calculation():
    """
    Tests Phase 4B fairness and disparity metric calculations.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7])
    sensitive_features = pd.Series(['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Female'])
    
    # Use fallback explicitly for deterministic isolated testing
    metrics = fallback_metrics_by_group(y_true, y_pred, sensitive_features)
    
    assert 'Male' in metrics
    assert 'Female' in metrics
    assert 'tpr' in metrics['Male']
    assert 'fpr' in metrics['Female']
    
    # Test standard pipeline
    full_metrics = calculate_fairness_metrics(y_true, y_pred, y_prob, sensitive_features)
    assert 'group_metrics' in full_metrics
    assert 'disparities' in full_metrics