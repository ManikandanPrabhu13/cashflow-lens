import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def build_preprocessor(numeric_features, categorical_features):
    """
    Builds a scikit-learn ColumnTransformer for robust, leakage-safe preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_models(df, target_col, numeric_features, categorical_features, test_size=0.2, random_state=42):
    """
    Trains baseline and primary candidate models on the provided features.
    Maintains clean train/test separation.
    """
    logger.info("Initializing model training pipeline...")
    
    X = df[numeric_features + categorical_features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    logger.info("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Extract feature names post-encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + encoded_cat_features
    
    models = {
        'logistic_regression': LogisticRegression(
            random_state=random_state, max_iter=1000, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=random_state, n_estimators=100, class_weight='balanced'
        ),
        'lightgbm': lgb.LGBMClassifier(
            random_state=random_state, class_weight='balanced', verbose=-1
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        
    return {
        'preprocessor': preprocessor,
        'models': trained_models,
        'feature_names': feature_names,
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train,
        'y_test': y_test
    }

def save_artifacts(artifacts, output_dir="backend/artifacts"):
    """
    Persists trained models and preprocessing pipelines for inference.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(artifacts['preprocessor'], out_path / 'preprocessor.joblib')
    for name, model in artifacts['models'].items():
        joblib.dump(model, out_path / f'{name}.joblib')
        
    joblib.dump(artifacts['feature_names'], out_path / 'feature_names.joblib')
    logger.info(f"Training artifacts successfully saved to {output_dir}")