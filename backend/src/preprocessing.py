"""Cleaning, validation and model-preprocessing utilities.

Leakage controls implemented here:
  * ``split_borrowers`` splits by borrower BEFORE any fitting.
  * The preprocessor (winsorising clipper + imputer + scaler + one-hot) is fitted ONLY on the
    training split; it is then applied unchanged to validation/test/serving data.
  * ``assert_no_leakage`` blocks target, ID and scenario-label columns from the feature list.

Cleaning deliberately does NOT de-duplicate invoices: duplicate invoices are evidence signals
that the evidence-integrity engine must see. Only duplicate PRIMARY KEYS (system artefacts)
are removed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import get_paths, get_seed, get_settings
from src.utils import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "default_flag"
ID_COLUMN = "borrower_id"
SENSITIVE_COLUMNS = ["enterprise_size", "sector", "region", "digital_adoption_level", "credit_history_band"]
FORBIDDEN_FEATURE_COLUMNS = {TARGET_COLUMN, ID_COLUMN, "scenario_types", "n_scenarios", "demo_scenario"}
FORBIDDEN_PREFIXES = ("outcome_", "target_", "label_")

DATE_COLUMNS: Dict[str, List[str]] = {
    "borrowers": ["application_date"],
    "bank_transactions": ["txn_date"],
    "upi_transactions": ["txn_date"],
    "invoices": ["invoice_date", "due_date"],
    "vendors": ["relationship_start_date"],
    "inventory": ["movement_date"],
    "gst_records": ["invoice_date", "filing_date"],
    "loans": ["disbursal_date"],
    "repayments": ["due_date", "paid_date"],
    "scenarios": [],
}
NUMERIC_COLUMNS: Dict[str, List[str]] = {
    "borrowers": ["credit_history_months", "requested_amount", "requested_tenure_months",
                  "requested_interest_rate", "requested_emi"],
    "bank_transactions": ["amount", "balance_after", "txn_hour"],
    "upi_transactions": ["amount", "balance_after", "txn_hour"],
    "invoices": ["amount", "tax_amount", "total_amount", "quantity"],
    "vendors": ["avg_payment_terms_days"],
    "inventory": ["quantity", "unit_cost", "stock_after"],
    "gst_records": ["taxable_value", "tax_amount"],
    "loans": ["principal", "annual_interest_rate", "tenure_months", "emi"],
    "repayments": ["due_amount", "paid_amount", "days_past_due"],
    "scenarios": [],
}
PRIMARY_KEYS = {
    "borrowers": "borrower_id", "bank_transactions": "txn_id", "upi_transactions": "txn_id",
    "invoices": "invoice_id", "vendors": "vendor_id", "inventory": "inventory_id",
    "gst_records": "gst_record_id", "loans": "loan_id", "repayments": "repayment_id",
}
AMOUNT_COLUMNS = {"amount", "total_amount", "taxable_value", "principal", "emi", "due_amount"}


# --------------------------------------------------------------------------- #
# Cleaning + validation
# --------------------------------------------------------------------------- #
def clean_and_validate(data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Coerce types, strip strings, drop structurally invalid rows, and report data quality."""
    cleaned: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Any] = {"tables": {}, "warnings": []}
    known_ids = set(data["borrowers"][ID_COLUMN].astype(str))

    for name, df in data.items():
        df = df.copy()
        info: Dict[str, Any] = {"rows_in": int(len(df))}
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
        for c in DATE_COLUMNS.get(name, []):
            if c in df:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in NUMERIC_COLUMNS.get(name, []):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if ID_COLUMN in df.columns:
            bad = ~df[ID_COLUMN].astype(str).isin(known_ids) | df[ID_COLUMN].isna()
            info["orphan_rows_dropped"] = int(bad.sum())
            df = df[~bad]
        key_dates = [c for c in DATE_COLUMNS.get(name, []) if c != "paid_date"]
        if key_dates:
            miss = df[key_dates].isna().any(axis=1)
            info["rows_missing_key_date_dropped"] = int(miss.sum())
            df = df[~miss]
        pk = PRIMARY_KEYS.get(name)
        if pk and pk in df:
            dup = df.duplicated(subset=[pk], keep="first")
            info["duplicate_primary_keys_dropped"] = int(dup.sum())
            df = df[~dup]
        neg = {c: int((df[c] < 0).sum()) for c in AMOUNT_COLUMNS if c in df and (df[c] < 0).any()}
        if neg:
            info["negative_amounts"] = neg
            report["warnings"].append(f"{name}: negative amounts {neg}")
        info["null_counts"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
        if name == "invoices" and "invoice_number" in df:
            info["repeated_invoice_numbers_kept"] = int(
                df.duplicated(subset=[ID_COLUMN, "invoice_type", "invoice_number"], keep=False).sum())
        info["rows_out"] = int(len(df))
        cleaned[name] = df.reset_index(drop=True)
        report["tables"][name] = info
    logger.info("Cleaning complete: %s", {k: v["rows_out"] for k, v in report["tables"].items()})
    return cleaned, report


# --------------------------------------------------------------------------- #
# Split + leakage guard
# --------------------------------------------------------------------------- #
def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    """Raise if the feature list contains target/ID/scenario-label columns."""
    bad = [c for c in feature_columns
           if c in FORBIDDEN_FEATURE_COLUMNS or c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise ValueError(f"Potential target leakage: forbidden feature columns {bad}")


def split_borrowers(df: pd.DataFrame, test_size: Optional[float] = None, seed: Optional[int] = None,
                    target: str = TARGET_COLUMN) -> Tuple[pd.Index, pd.Index]:
    """Stratified borrower-level train/test split. Returns (train_index, test_index)."""
    ts = float(test_size if test_size is not None else get_settings()["model"]["test_size"])
    train_idx, test_idx = train_test_split(
        df.index, test_size=ts, random_state=get_seed() if seed is None else seed, stratify=df[target]
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def infer_feature_types(df: pd.DataFrame, feature_columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split feature names into (numeric, categorical) by dtype."""
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in feature_columns if c not in numeric]
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise features to quantiles learned on the TRAINING data only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        self.feature_names_in_ = np.asarray(
            X.columns if hasattr(X, "columns") else [f"x{i}" for i in range(arr.shape[1])], dtype=object)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def build_preprocessor(numeric: Sequence[str], categorical: Sequence[str] = ()) -> ColumnTransformer:
    """Unfitted ColumnTransformer producing a pandas DataFrame with stable feature names."""
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("clip", QuantileClipper()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(numeric)))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical)))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre


def fit_preprocessor(X_train: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str] = ()) -> ColumnTransformer:
    """Fit the preprocessor on training rows ONLY."""
    assert_no_leakage(list(numeric) + list(categorical))
    pre = build_preprocessor(numeric, categorical)
    pre.fit(X_train[list(numeric) + list(categorical)])
    return pre


def transform_features(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor; returns a DataFrame indexed like ``X``."""
    cols = list(getattr(pre, "feature_names_in_", X.columns))
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"Missing feature columns for preprocessing: {missing}")
    return pre.transform(X[cols])


def save_preprocessor(pre: ColumnTransformer, path: str | Path | None = None) -> Path:
    p = Path(path) if path else get_paths().preprocessor
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, p)
    return p


def load_preprocessor(path: str | Path | None = None) -> ColumnTransformer:
    p = Path(path) if path else get_paths().preprocessor
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor not found at {p}. Run scripts/setup_demo.py first.")
    return joblib.load(p)


# --------------------------------------------------------------------------- #
# Processed-data IO
# --------------------------------------------------------------------------- #
def save_processed(df: pd.DataFrame, name: str, directory: str | Path | None = None) -> Path:
    d = Path(directory) if directory else get_paths().processed_data
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def load_processed(name: str, directory: str | Path | None = None) -> pd.DataFrame:
    d = Path(directory) if directory else get_paths().processed_data
    return pd.read_csv(d / f"{name}.csv")