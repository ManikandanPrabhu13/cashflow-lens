import pytest
import pandas as pd
import numpy as np
from src.forecasting import aggregate_historical_cashflows, calculate_forecast

def test_cashflow_aggregation():
    """
    Tests Phase 4B historical cashflow aggregation logic.
    """
    dates = pd.date_range(start='2023-01-01', periods=10)
    df = pd.DataFrame({
        'date': dates,
        'amount': [1000, 500, 1200, 300, 400, 1500, 200, 800, 100, 600],
        'transaction_type': ['credit', 'debit', 'credit', 'debit', 'debit', 'credit', 'debit', 'credit', 'debit', 'credit']
    })
    
    daily_cf = aggregate_historical_cashflows(df)
    
    assert not daily_cf.empty
    assert 'inflow' in daily_cf.columns
    assert 'outflow' in daily_cf.columns
    assert 'net_cash_flow' in daily_cf.columns
    
def test_forecast_calculation():
    """
    Tests Phase 4B forecasting logic.
    """
    dates = pd.date_range(start='2023-01-01', periods=5)
    daily_cf = pd.DataFrame({
        'inflow': [1000, 1100, 1050, 1200, 1150],
        'outflow': [500, 600, 550, 500, 450],
        'net_cash_flow': [500, 500, 500, 700, 700]
    }, index=dates)
    
    forecasts = calculate_forecast(daily_cf, periods=[30, 60], current_balance=5000)
    
    assert '30d' in forecasts
    assert '60d' in forecasts
    assert forecasts['30d']['expected_inflow'] > 0
    assert forecasts['30d']['projected_cash_balance'] > 5000
    assert forecasts['30d']['liquidity_pressure'] in ["Low", "Medium", "High"]