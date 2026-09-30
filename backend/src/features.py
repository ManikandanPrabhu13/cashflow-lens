"""Financial health feature engine.

Turns raw bank/UPI-style transactions (and optional loan records) into one
row of numeric financial-health features per borrower.

Design notes
------------
* Only observable behaviour is used. No label / default outcome is read here,
  and ``as_of`` lets the caller cut the observation window so that features
  never see data from the outcome window (prevents target leakage).
* Borrower attributes used for fairness auditing (segment, sector, region,
  digital adoption, credit-history depth) are deliberately NOT produced here.
* ``monthly_cashflow`` is public so forecasting and stress testing can reuse
  exactly the same monthly aggregation as the feature engine.
* Missing inputs never produce NaN: features fall back to documented
  neutral values (0, or ``dscr_cap`` for debt-service coverage without debt).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

INFLOW = "inflow"
OUTFLOW = "outflow"

_DIRECTION_ALIASES = {
    "inflow": INFLOW, "credit": INFLOW, "cr": INFLOW, "in": INFLOW,
    "outflow": OUTFLOW, "debit": OUTFLOW, "dr": OUTFLOW, "out": OUTFLOW,
}


def normalize_direction(series: pd.Series) -> pd.Series:
    """Map direction labels (credit/debit/cr/dr/...) to 'inflow' / 'outflow'."""
    s = series.astype(str).str.strip().str.lower()
    return s.map(_DIRECTION_ALIASES).fillna(s)


@dataclass(frozen=True)
class FeatureConfig:
    """Configurable vocabulary and constants for feature engineering."""

    input_cost_categories: frozenset = frozenset(
        {"input_cost", "purchase", "raw_material", "inventory_purchase",
         "vendor_payment", "cogs", "supplier_payment"}
    )
    opex_categories: frozenset = frozenset(
        {"opex", "salary", "salaries", "rent", "utilities", "marketing",
         "logistics", "operating_expense", "overheads"}
    )
    debt_service_categories: frozenset = frozenset(
        {"loan_repayment", "emi", "debt_service", "interest"}
    )
    dscr_cap: float = 10.0
    min_months_for_trend: int = 3

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]] = None) -> "FeatureConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            if isinstance(getattr(cls, key), frozenset):
                value = frozenset(str(v).lower() for v in value)
            kwargs[key] = value
        return cls(**kwargs)


# (name, label, group, description) -- single source of truth for the feature set.
FEATURES: List[tuple] = [
    ("avg_monthly_inflow", "Average monthly inflow", "cash_flow_level",
     "Mean total monthly inflow across observed months."),
    ("avg_monthly_outflow", "Average monthly outflow", "cash_flow_level",
     "Mean total monthly outflow across observed months."),
    ("avg_monthly_net_cashflow", "Average monthly net cash flow", "cash_flow_level",
     "Mean of monthly inflow minus outflow."),
    ("net_cashflow_margin", "Net cash-flow margin", "cash_flow_level",
     "Total net cash flow divided by total inflow."),
    ("inflow_volatility_cv", "Inflow volatility", "volatility",
     "Coefficient of variation (std / mean) of monthly inflow."),
    ("outflow_volatility_cv", "Outflow volatility", "volatility",
     "Coefficient of variation of monthly outflow."),
    ("net_cashflow_volatility", "Net cash-flow volatility", "volatility",
     "Std of monthly net cash flow divided by mean monthly inflow."),
    ("negative_net_month_share", "Share of negative cash-flow months", "volatility",
     "Fraction of observed months where outflow exceeded inflow."),
    ("inflow_trend_pct_per_month", "Inflow trend", "trend",
     "Linear-trend slope of monthly inflow as a fraction of mean monthly inflow."),
    ("input_cost_ratio", "Input-cost ratio", "cost_structure",
     "Input-cost outflows divided by inflow."),
    ("opex_ratio", "Operating-expense ratio", "cost_structure",
     "Operating-expense outflows divided by inflow."),
    ("debt_outflow_ratio", "Observed debt-service outflow ratio", "debt_service",
     "Debt-service outflows seen in transactions divided by inflow."),
    ("top_counterparty_inflow_share", "Top customer share of inflow", "concentration",
     "Largest single counterparty share of identified inflow."),
    ("inflow_counterparty_hhi", "Inflow concentration index", "concentration",
     "Herfindahl-Hirschman index of identified inflow by counterparty (0-1)."),
    ("active_counterparty_count", "Active counterparties", "concentration",
     "Number of distinct identified counterparties."),
    ("txn_per_month", "Transactions per month", "activity",
     "Transaction count divided by months observed."),
    ("avg_txn_size", "Average transaction size", "activity",
     "Mean absolute transaction amount."),
    ("months_observed", "Months of history observed", "activity",
     "Number of calendar months in the observation window."),
    ("monthly_emi", "Scheduled monthly EMI", "debt_service",
     "Sum of scheduled monthly instalments across loans."),
    ("emi_to_inflow_ratio", "EMI-to-inflow ratio", "debt_service",
     "Scheduled EMI divided by mean monthly inflow."),
    ("debt_service_coverage", "Debt-service coverage", "debt_service",
     "Mean monthly net cash flow before debt service divided by scheduled EMI "
     "(capped; equals the cap when the borrower has no scheduled EMI)."),
    ("outstanding_to_annual_inflow", "Outstanding debt to annualised inflow", "debt_service",
     "Outstanding loan balance divided by 12x mean monthly inflow."),
    ("missed_payment_count", "Missed payments", "repayment_history",
     "Total missed payments recorded across loans."),
    ("max_days_past_due", "Maximum days past due", "repayment_history",
     "Worst days-past-due recorded across loans."),
]

FEATURE_COLUMNS: List[str] = [f[0] for f in FEATURES]
FEATURE_LABELS: Dict[str, str] = {f[0]: f[1] for f in FEATURES}
FEATURE_GROUPS: Dict[str, str] = {f[0]: f[2] for f in FEATURES}
FEATURE_DESCRIPTIONS: Dict[str, str] = {f[0]: f[3] for f in FEATURES}

MONTHLY_COLUMNS = [
    "borrower_id", "month", "inflow", "outflow", "net_cashflow",
    "input_cost", "opex", "debt_service", "other_outflow",
]


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------
def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den and float(den) > 0 else default


def _std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _cv(x: np.ndarray) -> float:
    return _safe_div(_std(x), float(np.mean(x)))


def _trend(x: np.ndarray, min_months: int) -> float:
    mean = float(np.mean(x)) if len(x) else 0.0
    if len(x) < min_months or mean <= 0:
        return 0.0
    slope = np.polyfit(np.arange(len(x)), x, 1)[0]
    return float(slope / mean)


def _vdiv(a: pd.Series, b: pd.Series) -> pd.Series:
    """Vectorised a / b, returning 0 where b <= 0."""
    b = b.astype(float)
    return pd.Series(np.where(b > 0, a.astype(float) / b.where(b > 0, 1.0), 0.0), index=a.index)


# --------------------------------------------------------------------------
# Transaction preparation + monthly aggregation
# --------------------------------------------------------------------------
def prepare_transactions(
    transactions: Optional[pd.DataFrame], as_of: Optional[Any] = None
) -> pd.DataFrame:
    """Normalise a transactions table (dates, positive amounts, direction)."""
    base_cols = ["txn_id", "borrower_id", "date", "amount", "direction", "counterparty_id"]
    if transactions is None or len(transactions) == 0:
        return pd.DataFrame(columns=base_cols)
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["direction"] = normalize_direction(df["direction"])
    if "counterparty_id" not in df.columns:
        df["counterparty_id"] = np.nan
    if "txn_id" not in df.columns:
        df["txn_id"] = np.arange(len(df))
    df = df.dropna(subset=["date", "amount"])
    if as_of is not None:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    return df.reset_index(drop=True)


def monthly_cashflow(
    transactions: Optional[pd.DataFrame],
    config: Optional[FeatureConfig] = None,
    as_of: Optional[Any] = None,
) -> pd.DataFrame:
    """Monthly cash-flow table per borrower with category splits.

    Months between a borrower's first and last observed month with no
    activity are included as zeros. Output columns: ``MONTHLY_COLUMNS``.
    """
    cfg = config or FeatureConfig()
    df = prepare_transactions(transactions, as_of)
    if df.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    if "category" in df.columns:
        category = df["category"].astype(str).str.lower()
    else:
        category = pd.Series("", index=df.index)
    is_in = df["direction"].eq(INFLOW)
    is_out = df["direction"].eq(OUTFLOW)
    amount = df["amount"].to_numpy(float)
    out_amt = np.where(is_out, amount, 0.0)

    df["inflow"] = np.where(is_in, amount, 0.0)
    df["outflow"] = out_amt
    df["input_cost"] = np.where(category.isin(cfg.input_cost_categories), out_amt, 0.0)
    df["opex"] = np.where(category.isin(cfg.opex_categories), out_amt, 0.0)
    df["debt_service"] = np.where(category.isin(cfg.debt_service_categories), out_amt, 0.0)
    df["other_outflow"] = (
        df["outflow"] - df["input_cost"] - df["opex"] - df["debt_service"]
    ).clip(lower=0.0)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    value_cols = ["inflow", "outflow", "input_cost", "opex", "debt_service", "other_outflow"]
    agg = df.groupby(["borrower_id", "month"], as_index=False)[value_cols].sum()

    frames = []
    for borrower_id, g in agg.groupby("borrower_id"):
        g = g.drop(columns="borrower_id").set_index("month")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g = g.reindex(idx, fill_value=0.0)
        g.index.name = "month"
        g.insert(0, "borrower_id", borrower_id)
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["net_cashflow"] = out["inflow"] - out["outflow"]
    return out[MONTHLY_COLUMNS]


# --------------------------------------------------------------------------
# Counterparty / activity / loan features
# --------------------------------------------------------------------------
def _counterparty_features(tx: pd.DataFrame) -> pd.DataFrame:
    cols = ["txn_count", "avg_txn_size", "top_counterparty_inflow_share",
            "inflow_counterparty_hhi", "active_counterparty_count"]
    if tx.empty:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="borrower_id"))
    n = tx.groupby("borrower_id").size()
    total = tx.groupby("borrower_id")["amount"].sum()
    out = pd.DataFrame({"txn_count": n, "avg_txn_size": total / n})

    ident = tx.dropna(subset=["counterparty_id"])
    out["active_counterparty_count"] = ident.groupby("borrower_id")["counterparty_id"].nunique()
    inflow = ident[ident["direction"] == INFLOW]
    if not inflow.empty:
        by = inflow.groupby(["borrower_id", "counterparty_id"])["amount"].sum()
        tot = by.groupby(level=0).transform("sum")
        share = (by / tot).where(tot > 0, 0.0)
        out["top_counterparty_inflow_share"] = share.groupby(level=0).max()
        out["inflow_counterparty_hhi"] = (share ** 2).groupby(level=0).sum()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[cols].fillna(0.0)


def _loan_features(loans: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["monthly_emi", "outstanding", "missed_payment_count", "max_days_past_due"]
    if loans is None or len(loans) == 0:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="borrower_id"))
    df = loans.copy()
    for c in ("monthly_emi", "outstanding_balance", "missed_payments", "max_dpd"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.groupby("borrower_id").agg(
        monthly_emi=("monthly_emi", "sum"),
        outstanding=("outstanding_balance", "sum"),
        missed_payment_count=("missed_payments", "sum"),
        max_days_past_due=("max_dpd", "max"),
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def compute_financial_features(
    transactions: Optional[pd.DataFrame],
    loans: Optional[pd.DataFrame] = None,
    borrower_ids: Optional[Sequence[Any]] = None,
    config: Optional[FeatureConfig] = None,
    as_of: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute ``FEATURE_COLUMNS`` for every borrower.

    Parameters
    ----------
    transactions : normalised or raw transactions table.
    loans        : optional loan table (see module contract).
    borrower_ids : if given, output contains exactly these borrowers in this
                   order; borrowers without transactions get neutral values.
    as_of        : only transactions dated on/before this date are used.
    """
    cfg = config or FeatureConfig()
    monthly = monthly_cashflow(transactions, cfg, as_of)
    tx = prepare_transactions(transactions, as_of)

    rows = []
    for borrower_id, g in monthly.groupby("borrower_id"):
        inflow = g["inflow"].to_numpy(float)
        outflow = g["outflow"].to_numpy(float)
        net = g["net_cashflow"].to_numpy(float)
        total_in = float(inflow.sum())
        mean_in = float(inflow.mean())
        rows.append({
            "borrower_id": borrower_id,
            "avg_monthly_inflow": mean_in,
            "avg_monthly_outflow": float(outflow.mean()),
            "avg_monthly_net_cashflow": float(net.mean()),
            "net_cashflow_margin": _safe_div(net.sum(), total_in),
            "inflow_volatility_cv": _cv(inflow),
            "outflow_volatility_cv": _cv(outflow),
            "net_cashflow_volatility": _safe_div(_std(net), mean_in),
            "negative_net_month_share": float((net < 0).mean()),
            "inflow_trend_pct_per_month": _trend(inflow, cfg.min_months_for_trend),
            "input_cost_ratio": _safe_div(g["input_cost"].sum(), total_in),
            "opex_ratio": _safe_div(g["opex"].sum(), total_in),
            "debt_outflow_ratio": _safe_div(g["debt_service"].sum(), total_in),
            "months_observed": float(len(g)),
            "_avg_debt_outflow": float(g["debt_service"].mean()),
        })
    if rows:
        feat = pd.DataFrame(rows).set_index("borrower_id")
    else:
        feat = pd.DataFrame(index=pd.Index([], name="borrower_id"))

    feat = feat.join(_counterparty_features(tx), how="left").join(
        _loan_features(loans), how="left"
    )
    if borrower_ids is not None:
        feat = feat.reindex(pd.Index(list(borrower_ids), name="borrower_id"))

    needed = [c for c in FEATURE_COLUMNS if c not in
              ("emi_to_inflow_ratio", "debt_service_coverage", "outstanding_to_annual_inflow",
               "txn_per_month")] + ["outstanding", "txn_count", "_avg_debt_outflow"]
    for c in needed:
        if c not in feat.columns:
            feat[c] = 0.0
    feat[needed] = feat[needed].fillna(0.0)

    feat["txn_per_month"] = _vdiv(feat["txn_count"], feat["months_observed"])
    feat["emi_to_inflow_ratio"] = _vdiv(feat["monthly_emi"], feat["avg_monthly_inflow"])
    net_before_debt = feat["avg_monthly_net_cashflow"] + feat["_avg_debt_outflow"]
    dscr = np.where(
        feat["monthly_emi"] > 0,
        net_before_debt / feat["monthly_emi"].where(feat["monthly_emi"] > 0, 1.0),
        cfg.dscr_cap,
    )
    feat["debt_service_coverage"] = np.clip(dscr, -cfg.dscr_cap, cfg.dscr_cap)
    feat["outstanding_to_annual_inflow"] = _vdiv(
        feat["outstanding"], feat["avg_monthly_inflow"] * 12.0
    )

    out = feat.reset_index()
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float)
    return out[["borrower_id"] + FEATURE_COLUMNS]


def build_feature_table(
    tables: Mapping[str, pd.DataFrame],
    config: Optional[FeatureConfig] = None,
    as_of: Optional[Any] = None,
) -> pd.DataFrame:
    """Convenience wrapper: build features from a dict of tables."""
    borrowers = tables.get("borrowers")
    ids = borrowers["borrower_id"].tolist() if borrowers is not None and len(borrowers) else None
    return compute_financial_features(
        tables.get("transactions"), tables.get("loans"), ids, config, as_of
    )


def feature_metadata() -> Dict[str, Any]:
    """Structured description of every feature (for artifacts / dashboard)."""
    return {
        "features": [
            {"name": n, "label": l, "group": g, "description": d} for n, l, g, d in FEATURES
        ],
        "groups": sorted({f[2] for f in FEATURES}),
        "notes": [
            "Features are computed only from observed transactions/loans on or before the "
            "as_of date; no outcome label is used.",
            "Borrower attributes used for fairness auditing are excluded from this set.",
            "Missing inputs fall back to neutral values (0; the DSCR cap for no scheduled EMI).",
        ],
    }


def save_feature_metadata(path: str | Path) -> Path:
    """Write ``feature_metadata()`` to JSON (e.g. artifacts/feature_metadata.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_metadata(), indent=2), encoding="utf-8")
    return path