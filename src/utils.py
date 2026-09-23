"""Small shared utilities: config loading and path helpers."""
from __future__ import annotations

import builtins
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = PROJECT_ROOT / "config.yaml") -> Dict[str, Any]:
    """Load the project's YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_path(relative: str) -> Path:
    """Resolve a path relative to the project root, creating parent dirs if needed."""
    path = PROJECT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def exclude_date_range(data, start: str, end: str):
    """
    Drop rows whose datetime index falls within [start, end] (inclusive), for use in
    shock-exclusion robustness checks (e.g. masking the 2020:03-2020:06 COVID shock out
    of both training and evaluation). Works on a DataFrame or Series with a DatetimeIndex.

    Note this does not insert a gap placeholder -- the remaining rows become contiguous,
    so any "last N rows" expanding-window logic downstream will treat the row immediately
    after the excluded range as following the row immediately before it. This is the
    intended behavior for this kind of exclusion check.
    """
    mask = (data.index < pd.Timestamp(start)) | (data.index > pd.Timestamp(end))
    return data.loc[mask].copy()


def get_fred_api_key() -> str | None:
    """Return the FRED API key from the environment, if set. Not required for CSV pulls."""
    return os.environ.get("FRED_API_KEY")


@contextmanager
def shap_xgboost_base_score_compat():
    """
    Work around a shap/xgboost incompatibility: xgboost>=2.1 serializes
    ``base_score`` as a bracketed array string (e.g. ``"[4.96E0]"``) even for
    single-output regressors, but shap's TreeExplainer parses it with a bare
    ``float()`` call and raises ValueError. Temporarily patch ``float`` to
    strip the brackets for the duration of the ``with`` block.
    """
    original_float = builtins.float

    def _patched_float(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            return original_float(x[1:-1])
        return original_float(x)

    builtins.float = _patched_float
    try:
        yield
    finally:
        builtins.float = original_float
