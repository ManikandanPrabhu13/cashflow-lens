import logging
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
try:
    from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, false_negative_rate, selection_rate
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def fallback_metrics_by_group(y_true, y_pred, sensitive_features):
    """
    Fallback manual calculation of group-based fairness metrics if Fairlearn is unavailable.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': sensitive_features})
    results = {}
    
    for group, group_df in df.groupby('group'):
        tn, fp, fn, tp = confusion_matrix(group_df['y_true'], group_df['y_pred'], labels=[0, 1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        results[str(group)] = {
            'tpr': float(tpr),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'precision': float(precision),
            'selection_rate': float((tp + fp) / len(group_df))
        }
        
    return results

def calculate_fairness_metrics(y_true, y_pred, y_prob, sensitive_features):
    """
    Calculates fairness and disparity metrics across identified protected groups.
    Computes equal opportunity and equalized odds violations.
    """
    logger.info("Calculating fairness metrics across protected attributes...")
    
    if not FAIRLEARN_AVAILABLE:
        logger.warning("Fairlearn not installed. Using fallback manual calculation.")
        metrics = fallback_metrics_by_group(y_true, y_pred, sensitive_features)
        return {"group_metrics": metrics, "disparities": {}}
        
    metrics_dict = {
        'tpr': true_positive_rate,
        'fpr': false_positive_rate,
        'fnr': false_negative_rate,
        'selection_rate': selection_rate,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0)
    }
    
    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    group_metrics = metric_frame.by_group.to_dict(orient='index')
    
    # Calculate Disparities
    disparities = {
        'equal_opportunity_difference': float(metric_frame.difference(method='between_groups')['tpr']),
        'equalized_odds_difference': float(max(
            metric_frame.difference(method='between_groups')['tpr'],
            metric_frame.difference(method='between_groups')['fpr']
        )),
        'demographic_parity_difference': float(metric_frame.difference(method='between_groups')['selection_rate']),
        'predictive_equality_difference': float(metric_frame.difference(method='between_groups')['fpr'])
    }
    
    return {
        'group_metrics': group_metrics,
        'disparities': disparities
    }