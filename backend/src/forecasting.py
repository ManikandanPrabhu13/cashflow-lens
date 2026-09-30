import pandas as pd
import numpy as np
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def aggregate_historical_cashflows(transactions_df, date_col='date', amount_col='amount', type_col='transaction_type'):
    """
    Aggregates historical transactions into daily inflows and outflows.
    Assumes transaction_type is categorized generally as 'credit' (inflow) or 'debit' (outflow).
    """
    df = transactions_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Ensure consistent daily index
    df['date_only'] = df[date_col].dt.date
    
    inflows = df[df[type_col] == 'credit'].groupby('date_only')[amount_col].sum().reset_index()
    inflows.rename(columns={amount_col: 'inflow'}, inplace=True)
    
    outflows = df[df[type_col] == 'debit'].groupby('date_only')[amount_col].sum().reset_index()
    outflows.rename(columns={amount_col: 'outflow'}, inplace=True)
    
    daily_cf = pd.merge(inflows, outflows, on='date_only', how='outer').fillna(0)
    daily_cf['date_only'] = pd.to_datetime(daily_cf['date_only'])
    daily_cf = daily_cf.sort_values('date_only').set_index('date_only')
    
    # Resample to ensure all days are represented
    daily_cf = daily_cf.resample('D').sum().fillna(0)
    daily_cf['net_cash_flow'] = daily_cf['inflow'] - daily_cf['outflow']
    
    return daily_cf

def calculate_forecast(daily_cf, periods=[30, 60, 90], current_balance=0.0, monthly_debt_service=0.0):
    """
    Generates 30, 60, and 90-day cash flow forecasts using rolling historical averages
    (exponentially weighted) to project future baseline liquidity and debt capacity.
    """
    logger.info(f"Generating cash flow forecasts for periods: {periods} days")
    
    # Use Exponential Moving Average (EMA) to weight recent history more heavily
    span = 30 # 30-day lookback window for EMA
    ema_inflow = daily_cf['inflow'].ewm(span=span, adjust=False).mean().iloc[-1]
    ema_outflow = daily_cf['outflow'].ewm(span=span, adjust=False).mean().iloc[-1]
    
    forecasts = {}
    
    for days in periods:
        projected_inflow = ema_inflow * days
        projected_outflow = ema_outflow * days
        projected_net = projected_inflow - projected_outflow
        projected_balance = current_balance + projected_net
        
        # Calculate proportional debt service for the period
        period_debt_service = (monthly_debt_service / 30.0) * days
        
        # Debt-Service Coverage Ratio (DSCR) proxy based on operating cash flow
        if period_debt_service > 0:
            dscr = projected_inflow / (projected_outflow + period_debt_service)
        else:
            dscr = float('inf') if projected_inflow > 0 else 0.0
            
        liquidity_pressure = "High" if projected_balance < period_debt_service else ("Medium" if projected_balance < (period_debt_service * 1.5) else "Low")
        
        forecasts[f'{days}d'] = {
            'expected_inflow': float(projected_inflow),
            'expected_outflow': float(projected_outflow),
            'expected_net_cash_flow': float(projected_net),
            'projected_cash_balance': float(projected_balance),
            'debt_service_capacity_dscr': float(dscr),
            'liquidity_pressure': liquidity_pressure
        }
        
    return forecasts

def generate_borrower_forecast(borrower_transactions, current_balance=0.0, monthly_debt=0.0):
    """
    Pipeline function to process a single borrower's transactions into a forecast artifact.
    """
    if borrower_transactions.empty:
        logger.warning("No transactions provided for forecast.")
        return None
        
    daily_cf = aggregate_historical_cashflows(borrower_transactions)
    forecast = calculate_forecast(daily_cf, current_balance=current_balance, monthly_debt_service=monthly_debt)
    
    return forecast