"""Central configuration loader for CashFlow-Lens.

Settings are read from ``config/settings.yaml`` and deep-merged over
built-in defaults, so the backend still runs if a key is missing.
Nothing in this module depends on any frontend.
"""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

# Environment variable that can point to an alternative settings file.
SETTINGS_ENV_VAR = "CASHFLOW_LENS_SETTINGS"

DISCLAIMER_SYNTHETIC = (
    "All data in this system is synthetically generated for demonstration. "
    "No live connection to bank accounts, UPI, GST or government databases exists."
)
DISCLAIMER_EVIDENCE = (
    "Evidence signals indicate inconsistencies or gaps that require review. "
    "They are not proof of fraud or misrepresentation."
)
DISCLAIMER_DECISION = (
    "This is a human-in-the-loop decision-support tool. It does not make "
    "automatic lending decisions."
)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "project": {"name": "CashFlow-Lens", "version": "0.1.0"},
    "seed": 42,
    "paths": {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "models": "models",
        "artifacts": "artifacts",
        "database": "data/cashflow_lens.db",
    },
    "data": {
        "n_borrowers": 600,
        "n_months": 24,
        "start_date": "2023-01-01",
        "loan_application_lookback_days": 45,
        "enterprise_sizes": {"Micro": 0.55, "Small": 0.33, "Medium": 0.12},
        "sectors": {
            "Retail": 0.28,
            "Manufacturing": 0.22,
            "Services": 0.2,
            "Agri-processing": 0.12,
            "Logistics": 0.1,
            "Hospitality": 0.08,
        },
        "regions": {"North": 0.22, "South": 0.28, "East": 0.16, "West": 0.24, "Central": 0.10},
        "inconsistency_scenario_rate": 0.12,
        "demo_borrowers": 5,
    },
    "features": {
        "observation_months": 18,
        "outcome_months": 6,
        "min_balance_floor": 0.0,
    },
    "evidence": {
        "amount_tolerance_pct": 0.02,
        "date_tolerance_days": 10,
        "near_duplicate_amount_pct": 0.01,
        "near_duplicate_days": 7,
        "gst_amount_tolerance_pct": 0.05,
        "inventory_quantity_tolerance_pct": 0.10,
        "new_counterparty_days": 60,
        "round_amount_modulus": 1000,
        "liquidity_spike_zscore": 2.5,
        "round_trip_window_days": 14,
        "weights": {
            "bank_invoice_match": 0.25,
            "gst_revenue_match": 0.20,
            "vendor_invoice_match": 0.15,
            "inventory_purchase_match": 0.15,
            "duplicate_invoice_signal": 0.10,
            "transaction_anomaly_signal": 0.15,
        },
        "review_threshold": 0.60,
    },
    "model": {
        "test_size": 0.25,
        "cv_folds": 5,
        "decision_threshold": 0.5,
        "logistic": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        },
        "lightgbm": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "verbose": -1,
        },
        "calibration_bins": 10,
        # Model used for serving. "auto" = best validation ROC-AUC, computed at training time.
        "serving_model": "auto",
    },
    "risk": {
        # Configurable, illustrative cut-offs. NOT universal banking standards.
        "low_max_pd": 0.15,
        "medium_max_pd": 0.35,
        "top_n_drivers": 5,
    },
    "forecast": {
        "horizons_days": [30, 60, 90],
        "history_months_used": 12,
        "trend_damping": 0.9,
        "min_history_months": 6,
        "liquidity_pressure_low": 1.5,
        "liquidity_pressure_high": 0.75,
    },
    "fairness": {
        "protected_attributes": [
            "enterprise_size",
            "sector",
            "region",
            "digital_adoption_level",
            "credit_history_band",
        ],
        "min_group_size": 30,
        "min_group_positives": 5,
        "disparity_flag_threshold": 0.10,
        "intersections": [["enterprise_size", "region"], ["sector", "digital_adoption_level"]],
        "proxy_correlation_threshold": 0.30,
        "mitigation_method": "threshold_adjustment",
        "mitigation_attribute": "enterprise_size",
    },
    "stress": {
        "sales_decline_pct": [0, -10, -20, -30],
        "input_cost_increase_pct": [0, 10, 20, 30],
        "interest_increase_pct_points": [0, 1, 2, 3],
        "opex_increase_pct": 0,
        "default_scenario": {
            "sales_decline_pct": 0,
            "input_cost_increase_pct": 0,
            "interest_increase_pct_points": 0,
            "opex_increase_pct": 0,
        },
    },
    "decision": {
        "manual_review_pd": 0.35,
        "manual_review_evidence_score": 0.60,
        "manual_review_stress_migration": True,
    },
    "api": {"host": "127.0.0.1", "port": 8000, "cors_origins": ["*"]},
    "logging": {"level": "INFO"},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_settings(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load settings from YAML and merge over defaults."""
    settings_path = Path(path or os.environ.get(SETTINGS_ENV_VAR, DEFAULT_SETTINGS_PATH))
    overrides: Dict[str, Any] = {}
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            logger.error("Could not parse %s (%s); using defaults.", settings_path, exc)
    else:
        logger.warning("Settings file %s not found; using defaults.", settings_path)
    return _deep_merge(DEFAULT_SETTINGS, overrides)


@lru_cache(maxsize=1)
def get_settings() -> Dict[str, Any]:
    """Cached settings accessor. Call ``reload_settings`` to refresh."""
    return load_settings()


def reload_settings() -> Dict[str, Any]:
    """Clear the cache and reload settings from disk."""
    get_settings.cache_clear()
    return get_settings()


@dataclass(frozen=True)
class Paths:
    """Resolved absolute filesystem locations used across the backend."""

    root: Path
    raw_data: Path
    processed_data: Path
    models: Path
    artifacts: Path
    database: Path

    # Model artifacts
    @property
    def logistic_model(self) -> Path:
        return self.models / "logistic_model.joblib"

    @property
    def random_forest_model(self) -> Path:
        return self.models / "random_forest_model.joblib"

    @property
    def lightgbm_model(self) -> Path:
        return self.models / "lightgbm_model.joblib"

    @property
    def preprocessor(self) -> Path:
        return self.models / "preprocessor.joblib"

    # JSON artifacts
    @property
    def model_metrics(self) -> Path:
        return self.artifacts / "model_metrics.json"

    @property
    def fairness_metrics(self) -> Path:
        return self.artifacts / "fairness_metrics.json"

    @property
    def feature_metadata(self) -> Path:
        return self.artifacts / "feature_metadata.json"

    @property
    def evidence_summary(self) -> Path:
        return self.artifacts / "evidence_summary.json"

    def model_path(self, name: str) -> Path:
        mapping = {
            "logistic": self.logistic_model,
            "logistic_regression": self.logistic_model,
            "random_forest": self.random_forest_model,
            "lightgbm": self.lightgbm_model,
        }
        if name not in mapping:
            raise KeyError(f"Unknown model name: {name!r}")
        return mapping[name]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_paths(create: bool = True) -> Paths:
    """Return resolved project paths, optionally creating directories."""
    cfg = get_settings()["paths"]
    paths = Paths(
        root=PROJECT_ROOT,
        raw_data=_resolve(cfg["raw_data"]),
        processed_data=_resolve(cfg["processed_data"]),
        models=_resolve(cfg["models"]),
        artifacts=_resolve(cfg["artifacts"]),
        database=_resolve(cfg["database"]),
    )
    if create:
        for d in (paths.raw_data, paths.processed_data, paths.models, paths.artifacts, paths.database.parent):
            d.mkdir(parents=True, exist_ok=True)
    return paths


def get_seed() -> int:
    """Global deterministic seed."""
    return int(get_settings()["seed"])