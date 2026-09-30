import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure backend root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Phase 4-6 Modules
from src.train import train_models, save_artifacts
from src.evaluate import evaluate_all_models
from src.explain import generate_global_explanations
from src.forecasting import aggregate_historical_cashflows, calculate_forecast
from src.fairness import calculate_fairness_metrics
from src.mitigation import compare_baseline_vs_mitigated

# Attempt to import Batch 1-3 Modules dynamically
try:
    from src.data_generator import generate_synthetic_data
except ImportError:
    generate_synthetic_data = None
try:
    from src.features import engineer_features
except ImportError:
    engineer_features = None
try:
    from src.database import init_db
except ImportError:
    init_db = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_demo")

ARTIFACTS_DIR = Path("backend/artifacts")
DATA_DIR = Path("backend/data")

def create_fallback_demo_data(num_samples=1000):
    """Fallback function to generate perfectly shaped data if Batch 1-3 imports fail."""
    logger.info("Using built-in fallback data generator for demo...")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'borrower_id': [f"DEMO-{str(i).zfill(3)}" for i in range(1, num_samples + 1)],
        'monthly_revenue': np.random.uniform(5000, 50000, num_samples),
        'cash_balance': np.random.uniform(1000, 100000, num_samples),
        'late_payment_count': np.random.randint(0, 10, num_samples),
        'revenue_volatility': np.random.uniform(0.1, 0.8, num_samples),
        'debt_to_income': np.random.uniform(0.1, 0.9, num_samples),
        'industry_risk': np.random.uniform(0.5, 2.0, num_samples),
        'industry': np.random.choice(['Retail', 'Tech', 'Services', 'Manufacturing'], num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'default_flag': np.random.choice([0, 1], p=[0.8, 0.2], size=num_samples)
    })
    return df

def main():
    logger.info("Starting CashFlow-Lens Demo Setup...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    
    # 1. Database Initialization
    if init_db:
        logger.info("Initializing database...")
        init_db()
    
    # 2. Data Generation
    logger.info("Preparing MSME dataset...")
    if generate_synthetic_data and engineer_features:
        try:
            raw_data = generate_synthetic_data(num_samples=1000)
            raw_data.to_csv(DATA_DIR / "raw/dataset.csv", index=False)
            features_df = engineer_features(raw_data)
        except Exception as e:
            logger.warning(f"Batch 1-3 functions failed ({e}). Using fallback.")
            features_df = create_fallback_demo_data()
    else:
        logger.warning("Batch 1-3 modules not directly importable. Using fallback generator.")
        features_df = create_fallback_demo_data()
        
    # Save the processed features so the API can use them
    features_df.to_csv(DATA_DIR / "processed/features.csv", index=False)

    # Infer Columns for training
    target_col = 'default_flag' if 'default_flag' in features_df.columns else features_df.columns[-1]
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()
    categorical_features = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Filter out ID columns
    numeric_features = [f for f in numeric_features if 'id' not in f.lower()]
    categorical_features = [f for f in categorical_features if 'id' not in f.lower()]

    # 3. Model Training (Phase 4A)
    logger.info("Training models...")
    training_artifacts = train_models(
        df=features_df, 
        target_col=target_col,
        numeric_features=numeric_features, 
        categorical_features=categorical_features
    )
    save_artifacts(training_artifacts, output_dir=ARTIFACTS_DIR)
    
    # 4. Evaluation
    logger.info("Evaluating models...")
    evaluate_all_models(training_artifacts, output_dir=ARTIFACTS_DIR)
    
   # 5. Explainability (SHAP)
    logger.info("Generating global SHAP explanations...")
    # Determine the actual name of the model we are using
    best_model_name = 'lightgbm' if 'lightgbm' in training_artifacts['models'] else 'random_forest'
    primary_model = training_artifacts['models'].get(best_model_name)
    
    if primary_model:
        generate_global_explanations(
            model=primary_model,
            X_train=training_artifacts['X_train'],
            feature_names=training_artifacts['feature_names'],
            model_name=best_model_name,  # Pass the real name so SHAP knows it's a tree!
            output_dir=ARTIFACTS_DIR
        )
        
    # 6. Fairness & Mitigation
    logger.info("Calculating fairness and mitigation metrics...")
    sensitive_col = 'gender' if 'gender' in features_df.columns else None
    
    if sensitive_col:
        test_indices = training_artifacts['y_test'].index
        sensitive_features_test = features_df.loc[test_indices, sensitive_col]
        
        y_test = training_artifacts['y_test']
        X_test = training_artifacts['X_test']
        y_prob = primary_model.predict_proba(X_test)[:, 1]
        
        comparison = compare_baseline_vs_mitigated(y_test, y_prob, sensitive_features_test)
        
        with open(ARTIFACTS_DIR / 'fairness_report.json', 'w') as f:
            json.dump(comparison, f, indent=4)
            
    # 7. Seed Demo Borrower & Forecast Artifacts
    logger.info("Preparing demo borrower specific artifacts...")
    demo_borrower_id = features_df['borrower_id'].iloc[0] if 'borrower_id' in features_df.columns else "DEMO-001"
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    tx_data = pd.DataFrame({
        'date': dates,
        'amount': np.random.uniform(500, 5000, size=90),
        'transaction_type': np.where(np.random.rand(90) > 0.4, 'credit', 'debit')
    })
    
    daily_cf = aggregate_historical_cashflows(tx_data)
    forecast = calculate_forecast(daily_cf, current_balance=15000, monthly_debt_service=2000)
    
    with open(ARTIFACTS_DIR / f'forecast_{demo_borrower_id}.json', 'w') as f:
        json.dump(forecast, f, indent=4)
        
    logger.info("Setup complete! All artifacts successfully saved.")
    logger.info("System is ready. You can now start the API with: uvicorn api.main:app --reload")

if __name__ == "__main__":
    main()