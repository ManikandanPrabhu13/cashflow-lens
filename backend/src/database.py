"""SQLite persistence layer (SQLAlchemy Core + pandas).

Dates are stored as ISO text (YYYY-MM-DD). Bank and UPI transactions share one
``transactions`` table, distinguished by the ``channel`` column.
All data is SYNTHETIC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import (
    Column, Float, Index, Integer, MetaData, String, Table, Text, create_engine, delete, inspect, select,
)
from sqlalchemy.engine import Engine

from src.config import get_paths
from src.utils import get_logger

logger = get_logger(__name__)

metadata = MetaData()

# (column name, type, is_primary_key)
_SCHEMA: Dict[str, List[tuple]] = {
    "borrowers": [
        ("borrower_id", String, True), ("business_name", String), ("enterprise_size", String),
        ("sector", String), ("region", String), ("credit_history_months", Integer),
        ("credit_history_band", String), ("digital_adoption_level", String),
        ("incorporation_year", Integer), ("inventory_tracked", Integer), ("application_date", String),
        ("requested_amount", Float), ("requested_tenure_months", Integer),
        ("requested_interest_rate", Float), ("requested_emi", Float), ("default_flag", Integer),
    ],
    "transactions": [
        ("txn_id", String, True), ("borrower_id", String), ("channel", String), ("txn_date", String),
        ("txn_hour", Integer), ("amount", Float), ("direction", String), ("txn_type", String),
        ("payment_mode", String), ("counterparty_id", String), ("counterparty_name", String),
        ("reference", String), ("status", String), ("balance_after", Float),
    ],
    "invoices": [
        ("invoice_id", String, True), ("borrower_id", String), ("invoice_type", String),
        ("invoice_number", String), ("counterparty_id", String), ("counterparty_name", String),
        ("invoice_date", String), ("due_date", String), ("amount", Float), ("tax_amount", Float),
        ("total_amount", Float), ("item", String), ("quantity", Float),
    ],
    "vendors": [
        ("vendor_id", String, True), ("borrower_id", String), ("vendor_name", String),
        ("vendor_gstin", String), ("relationship_start_date", String),
        ("avg_payment_terms_days", Integer), ("is_registered", Integer),
    ],
    "inventory": [
        ("inventory_id", String, True), ("borrower_id", String), ("movement_date", String),
        ("item", String), ("movement_type", String), ("quantity", Float), ("unit_cost", Float),
        ("reference", String), ("stock_after", Float),
    ],
    "gst_records": [
        ("gst_record_id", String, True), ("borrower_id", String), ("invoice_number", String),
        ("counterparty_name", String), ("counterparty_gstin", String), ("invoice_date", String),
        ("taxable_value", Float), ("tax_amount", Float), ("direction", String),
        ("filing_period", String), ("filing_date", String),
    ],
    "loans": [
        ("loan_id", String, True), ("borrower_id", String), ("lender_type", String),
        ("disbursal_date", String), ("principal", Float), ("annual_interest_rate", Float),
        ("tenure_months", Integer), ("emi", Float), ("is_application_loan", Integer),
    ],
    "repayments": [
        ("repayment_id", String, True), ("loan_id", String), ("borrower_id", String),
        ("due_date", String), ("paid_date", String), ("due_amount", Float), ("paid_amount", Float),
        ("days_past_due", Float), ("status", String),
    ],
    "demo_scenarios": [
        ("borrower_id", String, True), ("scenario_types", Text), ("n_scenarios", Integer),
    ],
}

# Tables with an auto-increment surrogate key ("id").
_AUTO_SCHEMA: Dict[str, List[tuple]] = {
    "evidence_relationships": [
        ("borrower_id", String), ("claim_id", String), ("source_type", String), ("source_id", String),
        ("target_type", String), ("target_id", String), ("relationship_type", String),
        ("match_status", String), ("amount_difference", Float), ("date_difference_days", Float),
        ("quantity_difference", Float), ("matching_logic", Text), ("confidence", Float),
        ("explanation", Text), ("created_at", String),
    ],
    "evidence_alerts": [
        ("borrower_id", String), ("alert_type", String), ("category", String), ("severity", String),
        ("title", String), ("description", Text), ("related_ids", Text), ("metric_value", Float),
        ("created_at", String),
    ],
    "model_predictions": [
        ("borrower_id", String), ("model_name", String), ("scenario", String), ("pd", Float),
        ("risk_segment", String), ("evidence_score", Float), ("details_json", Text),
        ("created_at", String),
    ],
}

TABLES: Dict[str, Table] = {}
for _name, _cols in _SCHEMA.items():
    TABLES[_name] = Table(
        _name, metadata,
        *[Column(c[0], c[1], primary_key=(len(c) > 2 and c[2])) for c in _cols],
    )
for _name, _cols in _AUTO_SCHEMA.items():
    TABLES[_name] = Table(
        _name, metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        *[Column(c[0], c[1]) for c in _cols],
    )
for _name, _t in TABLES.items():
    if "borrower_id" in _t.c and _name != "borrowers":
        Index(f"ix_{_name}_borrower", _t.c.borrower_id)

DB_DATE_COLUMNS = {
    "txn_date", "invoice_date", "due_date", "movement_date", "filing_date",
    "disbursal_date", "paid_date", "application_date", "relationship_start_date",
}
_BOOL_INT_COLUMNS = {"inventory_tracked", "is_registered", "is_application_loan"}


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Create a SQLite engine (thread-safe for FastAPI usage)."""
    path = Path(db_path) if db_path else get_paths().database
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{path.as_posix()}", future=True, connect_args={"check_same_thread": False}
    )


def init_db(engine: Optional[Engine] = None, drop_existing: bool = False) -> Engine:
    """Create all tables programmatically (optionally dropping existing ones)."""
    engine = engine or get_engine()
    if drop_existing:
        metadata.drop_all(engine)
    metadata.create_all(engine)
    logger.info("Database initialised at %s", engine.url)
    return engine


def _prepare(df: pd.DataFrame, table: Table) -> pd.DataFrame:
    """Align a DataFrame to a table's columns and convert to storable types."""
    out = df.copy()
    cols = [c.name for c in table.columns if c.name != "id"]
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols]
    for c in cols:
        if c in DB_DATE_COLUMNS:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
        elif c in _BOOL_INT_COLUMNS:
            out[c] = out[c].astype("float").astype("Int64")
    return out


def write_table(engine: Engine, name: str, df: pd.DataFrame, replace: bool = True) -> int:
    """Write a DataFrame to a table; ``replace`` clears existing rows first."""
    table = TABLES[name]
    metadata.create_all(engine, tables=[table])
    if replace:
        with engine.begin() as conn:
            conn.execute(delete(table))
    if df is None or df.empty:
        return 0
    _prepare(df, table).to_sql(name, engine, if_exists="append", index=False, chunksize=5000)
    return len(df)


def replace_borrower_rows(engine: Engine, name: str, borrower_id: str, df: pd.DataFrame) -> int:
    """Delete then insert all rows of one borrower in a table (e.g. refreshed predictions)."""
    table = TABLES[name]
    with engine.begin() as conn:
        conn.execute(delete(table).where(table.c.borrower_id == borrower_id))
    if df is None or df.empty:
        return 0
    _prepare(df, table).to_sql(name, engine, if_exists="append", index=False, chunksize=5000)
    return len(df)


def read_table(engine: Engine, name: str, borrower_id: Optional[str] = None) -> pd.DataFrame:
    """Read a table (optionally for one borrower) with date columns parsed."""
    table = TABLES[name]
    stmt = select(table)
    if borrower_id is not None and "borrower_id" in table.c:
        stmt = stmt.where(table.c.borrower_id == borrower_id)
    with engine.connect() as conn:
        df = pd.read_sql_query(stmt, conn)
    for c in df.columns:
        if c in DB_DATE_COLUMNS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in _BOOL_INT_COLUMNS & set(df.columns):
        df[c] = df[c].fillna(0).astype(int).astype(bool)
    return df


def save_dataset(engine: Engine, data: Dict[str, pd.DataFrame], replace: bool = True) -> Dict[str, int]:
    """Persist a generated/cleaned dataset dictionary. Returns row counts per table."""
    init_db(engine)
    counts: Dict[str, int] = {}
    for name in ["borrowers", "invoices", "vendors", "inventory", "gst_records", "loans", "repayments"]:
        counts[name] = write_table(engine, name, data[name], replace)
    tx = pd.concat(
        [data["bank_transactions"], data["upi_transactions"]], ignore_index=True
    )
    counts["transactions"] = write_table(engine, "transactions", tx, replace)
    if "scenarios" in data:
        counts["demo_scenarios"] = write_table(engine, "demo_scenarios", data["scenarios"], replace)
    logger.info("Saved dataset to database: %s", counts)
    return counts


def load_dataset(engine: Engine) -> Dict[str, pd.DataFrame]:
    """Load the dataset back in the same dictionary layout the generator produces."""
    data = {n: read_table(engine, n) for n in
            ["borrowers", "invoices", "vendors", "inventory", "gst_records", "loans", "repayments"]}
    tx = read_table(engine, "transactions")
    data["bank_transactions"] = tx[tx["channel"] == "BANK"].reset_index(drop=True)
    data["upi_transactions"] = tx[tx["channel"] == "UPI"].reset_index(drop=True)
    data["scenarios"] = read_table(engine, "demo_scenarios")
    return data


def load_borrower_dataset(engine: Engine, borrower_id: str) -> Dict[str, pd.DataFrame]:
    """Load every table restricted to one borrower."""
    data = {n: read_table(engine, n, borrower_id) for n in
            ["borrowers", "invoices", "vendors", "inventory", "gst_records", "loans", "repayments"]}
    tx = read_table(engine, "transactions", borrower_id)
    data["bank_transactions"] = tx[tx["channel"] == "BANK"].reset_index(drop=True)
    data["upi_transactions"] = tx[tx["channel"] == "UPI"].reset_index(drop=True)
    return data


def list_borrower_ids(engine: Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(select(TABLES["borrowers"].c.borrower_id)).fetchall()
    return [r[0] for r in rows]


def table_exists(engine: Engine, name: str) -> bool:
    return name in inspect(engine).get_table_names()


def row_counts(engine: Engine) -> Dict[str, int]:
    from sqlalchemy import func
    out: Dict[str, int] = {}
    with engine.connect() as conn:
        for name, t in TABLES.items():
            if table_exists(engine, name):
                out[name] = int(conn.execute(select(func.count()).select_from(t)).scalar() or 0)
    return out