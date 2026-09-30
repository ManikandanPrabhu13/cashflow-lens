import json
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss
)
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def calculate_ks_statistic(y_true, y_prob):
    """
    Calculates the Kolmogorov-Smirnov statistic for risk model separation.
    """
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    return ks_2samp(pos_probs, neg_probs).statistic

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Computes a comprehensive suite of evaluation metrics for a single model.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'pr_auc': float(average_precision_score(y_test, y_prob)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'brier_score': float(brier_score_loss(y_test, y_prob)),
        'ks_statistic': float(calculate_ks_statistic(y_test, y_prob))
    }
    
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    return metrics

def evaluate_all_models(artifacts, output_dir="backend/artifacts"):
    """
    Evaluates all trained models in the artifact dictionary and saves the results.
    """
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    models = artifacts['models']
    
    evaluation_results = {}
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        evaluation_results[name] = evaluate_model(model, X_test, y_test)
        
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_file = out_path / 'evaluation_metrics.json'
        
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
            
        logger.info(f"Evaluation metrics saved to {results_file}")
        
    return evaluation_results