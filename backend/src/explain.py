import numpy as np
import pandas as pd
import shap
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_explainer(model, X_background, model_type='tree'):
    """
    Instantiates the appropriate SHAP explainer based on the model class.
    """
    if model_type == 'tree':
        return shap.TreeExplainer(model)
    else:
        return shap.LinearExplainer(model, X_background)

def generate_global_explanations(model, X_train, feature_names, model_name='lightgbm', output_dir="backend/artifacts"):
    """
    Generates global feature importance metrics using SHAP values.
    """
    logger.info(f"Generating global SHAP explanations for {model_name}...")
    is_tree = model_name in ['random_forest', 'lightgbm']
    
    # Background dataset for linear models, unused for tree models
    X_bg = shap.sample(X_train, 100) if not is_tree else None
    explainer = get_explainer(model, X_bg, model_type='tree' if is_tree else 'linear')
    
    # Use a sample for global explanation to maintain reasonable execution times
    X_sample = shap.sample(X_train, 1000) if X_train.shape[0] > 1000 else X_train
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-class / binary array formatting returned by some tree explainers
    if isinstance(shap_values, list):
        shap_values = shap_values[1] 
        
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(out_path / f'{model_name}_global_importance.csv', index=False)
        
        # Save the explainer itself for API reuse
        joblib.dump(explainer, out_path / f'{model_name}_explainer.joblib')
        logger.info(f"Global SHAP values and explainer saved for {model_name}")
        
    return importance_df

def generate_local_explanation(explainer, X_instance, feature_names):
    """
    Provides a borrower-level explanation containing signed feature contributions.
    Constructs a human-readable reason-code structure for front-end consumption.
    """
    shap_values = explainer.shap_values(X_instance)
    expected_value = explainer.expected_value
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]
    else:
        shap_values = shap_values[0]
        
    contributions = pd.DataFrame({
        'feature': feature_names,
        'contribution': shap_values,
        'value': X_instance[0]
    }).sort_values('contribution', key=abs, ascending=False)
    
    # Generate human-readable reason codes
    top_positive = contributions[contributions['contribution'] > 0].head(3)
    top_negative = contributions[contributions['contribution'] < 0].head(3)
    
    reason_codes = []
    for _, row in top_positive.iterrows():
        reason_codes.append(f"Increased risk due to {row['feature']} (value: {row['value']:.2f})")
    for _, row in top_negative.iterrows():
        reason_codes.append(f"Decreased risk due to {row['feature']} (value: {row['value']:.2f})")
        
    return {
        'expected_value': float(expected_value),
        'contributions': contributions.to_dict(orient='records'),
        'reason_codes': reason_codes
    }