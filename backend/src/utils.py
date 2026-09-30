"""Shared helpers: logging, seeding, JSON IO, safe math, date utilities.

No business logic and no presentation code lives here.
"""
from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import get_seed, get_settings

_LOGGING_CONFIGURED = False


# --------------------------------------------------------------------------- #
# Logging / seeding
# --------------------------------------------------------------------------- #
def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    lvl = (level or get_settings()["logging"]["level"]).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def set_global_seed(seed: Optional[int] = None) -> int:
    """Seed python and numpy RNGs. Returns the seed used."""
    s = get_seed() if seed is None else int(seed)
    random.seed(s)
    np.random.seed(s)
    return s


def get_rng(seed: Optional[int] = None, offset: int = 0) -> np.random.Generator:
    """Return a numpy Generator derived from the global seed."""
    base = get_seed() if seed is None else int(seed)
    return np.random.default_rng(base + offset)


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log elapsed wall time for a block."""
    log = logger or get_logger("timer")
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s finished in %.2fs", label, time.perf_counter() - start)


# --------------------------------------------------------------------------- #
# JSON IO (numpy / pandas safe)
# --------------------------------------------------------------------------- #
def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas / date objects to JSON-safe types.

    NaN and infinity become ``None`` so output is strict JSON.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return to_serializable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_serializable(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(to_serializable(data), fh, indent=indent, ensure_ascii=False)
    return p


def load_json(path: str | Path, default: Any = None) -> Any:
    """Read JSON; return ``default`` if the file does not exist."""
    p = Path(path)
    if not p.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> Any:
    """Divide while returning ``default`` where the denominator is 0/NaN.

    Works on scalars, numpy arrays and pandas Series.
    """
    if np.isscalar(numerator) and np.isscalar(denominator):
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(num, den).shape, default, dtype=float)
    mask = (den != 0) & ~np.isnan(den) & ~np.isnan(num)
    np.divide(num, den, out=out, where=mask)
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return out


def clip01(x: Any) -> Any:
    return np.clip(x, 0.0, 1.0)


def pct_diff(a: float, b: float) -> float:
    """Relative difference |a-b| / max(|a|,|b|); 0 when both are zero."""
    denom = max(abs(a), abs(b))
    return 0.0 if denom == 0 else abs(a - b) / denom


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    mean = arr.mean()
    return 0.0 if mean == 0 else float(arr.std(ddof=0) / abs(mean))


def gini_concentration(values: Iterable[float]) -> float:
    """Herfindahl-Hirschman index of shares (0..1); higher = more concentrated."""
    arr = np.asarray(list(values), dtype=float)
    total = arr.sum()
    if arr.size == 0 or total <= 0:
        return 0.0
    shares = arr / total
    return float((shares**2).sum())


def growth_rate(series: Sequence[float]) -> float:
    """Relative growth between the mean of the first and last third of a series."""
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return 0.0
    k = max(1, arr.size // 3)
    early, late = arr[:k].mean(), arr[-k:].mean()
    return 0.0 if early == 0 else float((late - early) / abs(early))


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
def to_datetime(series: pd.Series | Sequence) -> pd.Series:
    """Parse to datetime, coercing errors to NaT."""
    return pd.to_datetime(pd.Series(series), errors="coerce")


def month_start(ts: pd.Timestamp | str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return pd.Timestamp(year=t.year, month=t.month, day=1)


def add_months(ts: pd.Timestamp | str, months: int) -> pd.Timestamp:
    return month_start(ts) + pd.DateOffset(months=months)


def days_between(a: pd.Timestamp | str, b: pd.Timestamp | str) -> int:
    return int((pd.Timestamp(b) - pd.Timestamp(a)).days)


def to_month_period(series: pd.Series) -> pd.Series:
    """Convert a datetime series to monthly periods (string ``YYYY-MM``)."""
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype(str)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalise_name(name: Any) -> str:
    """Lower-case, strip punctuation/extra whitespace for entity matching."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(name))
    return " ".join(s.split())


def name_similarity(a: Any, b: Any) -> float:
    """Token-based Jaccard similarity of two names (0..1)."""
    ta, tb = set(normalise_name(a).split()), set(normalise_name(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def band_from_thresholds(value: float, low_max: float, medium_max: float) -> str:
    """Map a value to Low / Medium / High using configurable cut-offs."""
    if value <= low_max:
        return "Low"
    if value <= medium_max:
        return "Medium"
    return "High"