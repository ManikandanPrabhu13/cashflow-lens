import logging
import pandas as pd
import numpy as np
try:
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from .evaluate import evaluate_model
from .fairness import calculate_fairness_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from sklearn.base import BaseEstimator, ClassifierMixin

class DummyEstimator(BaseEstimator, ClassifierMixin):
    """
    A wrapper to allow Fairlearn's ThresholdOptimizer to accept pre-computed probabilities 
    from an already trained pipeline without retraining.
    """
    def __init__(self, y_prob):
        self.y_prob = y_prob
        # Add required attributes so scikit-learn recognizes it as a fitted model
        self.classes_ = np.array([0, 1]) 
        
    def fit(self, X, y):
        self.is_fitted_ = True
        return self
        
    def predict(self, X):
        return (self.y_prob >= 0.5).astype(int)
        
    def predict_proba(self, X):
        # ThresholdOptimizer expects shape (n_samples, 2)
        return np.vstack((1 - self.y_prob, self.y_prob)).T
        
def apply_threshold_mitigation(y_true, y_prob, sensitive_features, constraint='equalized_odds'):
    """
    Applies post-processing threshold adjustments to balance fairness metrics 
    across groups while maintaining predictive performance.
    """
    if not FAIRLEARN_AVAILABLE:
        logger.error("Fairlearn is required for threshold mitigation.")
        return y_prob >= 0.5 # Return baseline as fallback
        
    logger.info(f"Applying Fairlearn ThresholdOptimizer with constraint: {constraint}")
    
    # We use the DummyEstimator to pass our already-predicted probabilities
    dummy_model = DummyEstimator(y_prob)
    
    # Note: ThresholdOptimizer requires a fit step, we use the validation set
    optimizer = ThresholdOptimizer(
        estimator=dummy_model,
        constraints=constraint,
        prefit=True,
        predict_method='predict_proba'
    )
    
    # We pass dummy X since the dummy model just uses the stored y_prob
    X_dummy = np.zeros((len(y_true), 1))
    
    optimizer.fit(X_dummy, y_true, sensitive_features=sensitive_features)
    mitigated_predictions = optimizer.predict(X_dummy, sensitive_features=sensitive_features)
    
    return mitigated_predictions, optimizer

def compare_baseline_vs_mitigated(y_true, y_prob, sensitive_features):
    """
    Compares the baseline model against the mitigated model across performance and fairness.
    """
    # 1. Baseline Metrics
    baseline_pred = (y_prob >= 0.5).astype(int)
    baseline_fairness = calculate_fairness_metrics(y_true, baseline_pred, y_prob, sensitive_features)
    
    # 2. Mitigate
    if FAIRLEARN_AVAILABLE:
        mitigated_pred, _ = apply_threshold_mitigation(y_true, y_prob, sensitive_features, constraint='equalized_odds')
    else:
        mitigated_pred = baseline_pred
        
    # 3. Mitigated Metrics
    # Note: Since ThresholdOptimizer outputs hard predictions, y_prob proxy is just the pred
    mitigated_fairness = calculate_fairness_metrics(y_true, mitigated_pred, mitigated_pred, sensitive_features)
    
    comparison_report = {
        'baseline': {
            'fairness': baseline_fairness
        },
        'mitigated': {
            'fairness': mitigated_fairness
        },
        'improvement': {}
    }
    
    if FAIRLEARN_AVAILABLE:
        comparison_report['improvement'] = {
            'equal_opportunity_diff_change': baseline_fairness['disparities'].get('equal_opportunity_difference', 0) - mitigated_fairness['disparities'].get('equal_opportunity_difference', 0),
            'equalized_odds_diff_change': baseline_fairness['disparities'].get('equalized_odds_difference', 0) - mitigated_fairness['disparities'].get('equalized_odds_difference', 0)
        }
        
    return comparison_report