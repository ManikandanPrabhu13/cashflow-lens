import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def apply_scenario_shocks(features_dict, scenario):
    """
    Applies configurable macroeconomic and business stress shocks to a borrower's financial features.
    
    Expected scenario keys:
    - sales_shock: e.g., -0.10 for -10%
    - cost_shock: e.g., 0.10 for +10%
    - interest_shock: e.g., 0.02 for +2% (additive to rate, multiplicative to expense proxy)
    - opex_shock: configurable e.g., 0.05 for +5%
    """
    stressed = features_dict.copy()
    
    # Apply Sales / Revenue Shock
    sales_shock = scenario.get('sales_shock', 0.0)
    for col in stressed.keys():
        if 'revenue' in col.lower() or 'sales' in col.lower() or 'inflow' in col.lower():
            stressed[col] = float(stressed[col] * (1 + sales_shock))
            
    # Apply Input Cost & Opex Shock
    cost_shock = scenario.get('cost_shock', 0.0)
    opex_shock = scenario.get('opex_shock', 0.0)
    total_expense_shock = cost_shock + opex_shock
    
    for col in stressed.keys():
        if 'expense' in col.lower() or 'cost' in col.lower() or 'outflow' in col.lower():
            stressed[col] = float(stressed[col] * (1 + total_expense_shock))
            
    # Apply Interest Rate Shock (proxy via interest expense or debt service)
    interest_shock = scenario.get('interest_shock', 0.0)
    for col in stressed.keys():
        if 'interest' in col.lower() or 'debt_service' in col.lower():
            # Simplistic scaling of debt cost assuming floating rates
            stressed[col] = float(stressed[col] * (1 + interest_shock))
            
    # Recalculate derived metrics like net cash flow and liquidity
    # Note: Adapts to standard naming conventions found in Batch 1-3 features
    revenue_val = stressed.get('monthly_revenue', stressed.get('revenue', 0))
    expense_val = stressed.get('monthly_expenses', stressed.get('expenses', 0))
    
    if 'net_cash_flow' in stressed:
        stressed['net_cash_flow'] = float(revenue_val - expense_val)
        
    if 'liquidity' in stressed and 'net_cash_flow' in stressed:
        stressed['liquidity'] = float(stressed.get('cash_balance', 0) + stressed['net_cash_flow'])
        
    return stressed

def get_risk_segment(pd_value):
    """Maps Probability of Default to Risk Segments."""
    if pd_value < 0.15:
        return "Low Risk"
    elif pd_value < 0.35:
        return "Medium Risk"
    else:
        return "High Risk"

def run_stress_test(borrower_features, model, preprocessor, scenario):
    """
    Executes a stress test by comparing baseline predictions against scenario-shocked predictions.
    """
    logger.info(f"Running stress test with scenario parameters: {scenario}")
    
    # Create baseline and stressed dataframes
    baseline_df = pd.DataFrame([borrower_features])
    stressed_features = apply_scenario_shocks(borrower_features, scenario)
    stressed_df = pd.DataFrame([stressed_features])
    
    # Preprocess
    try:
        X_base = preprocessor.transform(baseline_df)
        X_stress = preprocessor.transform(stressed_df)
    except Exception as e:
        logger.error(f"Preprocessing failed during stress test: {e}")
        raise
        
    # Predict
    base_pd = float(model.predict_proba(X_base)[0, 1])
    stress_pd = float(model.predict_proba(X_stress)[0, 1])
    
    base_segment = get_risk_segment(base_pd)
    stress_segment = get_risk_segment(stress_pd)
    
    return {
        'scenario': scenario,
        'baseline': {
            'metrics': borrower_features,
            'pd': base_pd,
            'risk_segment': base_segment
        },
        'stressed': {
            'metrics': stressed_features,
            'pd': stress_pd,
            'risk_segment': stress_segment
        },
        'impact': {
            'pd_change': float(stress_pd - base_pd),
            'risk_migration': f"{base_segment} -> {stress_segment}",
            'revenue_impact': float(stressed_features.get('monthly_revenue', 0) - borrower_features.get('monthly_revenue', 0)),
            'net_cash_flow_impact': float(stressed_features.get('net_cash_flow', 0) - borrower_features.get('net_cash_flow', 0))
        }
    }