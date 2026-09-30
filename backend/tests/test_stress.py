import pytest
from src.stress_test import apply_scenario_shocks, get_risk_segment

def test_apply_scenario_shocks():
    """
    Tests Phase 6 stress-testing macroeconomic shocks applied to borrower features.
    """
    baseline_features = {
        'monthly_revenue': 10000.0,
        'monthly_expenses': 6000.0,
        'cash_balance': 15000.0,
        'interest_expense': 500.0
    }
    
    scenario = {
        'sales_shock': -0.20, # -20%
        'cost_shock': 0.10,   # +10%
        'interest_shock': 0.05,
        'opex_shock': 0.05
    }
    
    stressed_features = apply_scenario_shocks(baseline_features, scenario)
    
    # Revenue should decrease by 20%
    assert stressed_features['monthly_revenue'] == 8000.0
    
    # Expenses should increase by (cost_shock + opex_shock) = 15%
    assert stressed_features['monthly_expenses'] == 6900.0
    
    # Net cash flow should be recalculated
    assert stressed_features['net_cash_flow'] == 1100.0 (8000.0 - 6900.0)

def test_risk_segment_mapping():
    """
    Tests Phase 6 PD-to-segment logic mapping.
    """
    assert get_risk_segment(0.10) == "Low Risk"
    assert get_risk_segment(0.25) == "Medium Risk"
    assert get_risk_segment(0.50) == "High Risk"