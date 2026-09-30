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

# Import Batch 1-3 Modules (Dynamic integration based on standard structure)
try:
    from src.data_generator import generate_synthetic_data
except ImportError:
    generate_synthetic_data = None
try:
    from src.features import engineer_features
except ImportError:
    engineer_features = None
try:
    from src.database import init_db, get_db_session
except ImportError:
    init_db = None
    get_db_session = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_demo")

ARTIFACTS_DIR = Path("backend/artifacts")
DATA_DIR = Path("backend/data")

def main():
    logger.info("Starting CashFlow-Lens Demo Setup...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    
    # 1. Database Initialization
    if init_db:
        logger.info("Initializing SQLite database...")
        init_db()
    
    # 2. Data Generation & Feature Engineering
    logger.info("Generating synthetic MSME dataset...")
    if generate_synthetic_data and engineer_features:
        raw_data = generate_synthetic_data(num_samples=1000)
        raw_data.to_csv(DATA_DIR / "raw/dataset.csv", index=False)
        features_df = engineer_features(raw_data)
        features_df.to_csv(DATA_DIR / "processed/features.csv", index=False)
    else:
        logger.warning("Batch 1-3 modules not directly importable. Attempting to use existing processed data.")
        features_df = pd.read_csv(DATA_DIR / "processed/features.csv") if (DATA_DIR / "processed/features.csv").exists() else None
        
    if features_df is None or features_df.empty:
        logger.error("No feature data available. Cannot proceed with setup.")
        return

    # Infer Columns
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
    primary_model = training_artifacts['models'].get('lightgbm', training_artifacts['models'].get('random_forest'))
    if primary_model:
        generate_global_explanations(
            model=primary_model,
            X_train=training_artifacts['X_train'],
            feature_names=training_artifacts['feature_names'],
            model_name='primary_model',
            output_dir=ARTIFACTS_DIR
        )
        
    # 6. Fairness & Mitigation
    logger.info("Calculating fairness and mitigation metrics...")
    sensitive_col = 'gender' if 'gender' in features_df.columns else ('minority_owned' if 'minority_owned' in features_df.columns else None)
    
    if sensitive_col:
        # Align sensitive feature with test set
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
    
    # Create a mock transaction history for the forecasting module
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
        
    logger.info("Setup complete. Artifacts successfully saved.")
    logger.info("System is ready for FastAPI server launch.")

if __name__ == "__main__":
    main()