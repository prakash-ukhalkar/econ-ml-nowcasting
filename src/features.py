"""Feature engineering for the nowcasting panel: lags, rolling stats, seasonal dummies."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def add_lags(df: pd.DataFrame, columns: List[str], n_lags: int) -> pd.DataFrame:
    """Add lag_1..lag_n columns for each specified column."""
    out = df.copy()
    for col in columns:
        for lag in range(1, n_lags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def add_rolling_stats(df: pd.DataFrame, columns: List[str], windows: List[int] = (3, 6)) -> pd.DataFrame:
    """Add rolling mean/std features computed on lag_1 (to avoid leakage) for each column."""
    out = df.copy()
    for col in columns:
        base = out[col].shift(1)
        for w in windows:
            out[f"{col}_rollmean{w}"] = base.rolling(w).mean()
            out[f"{col}_rollstd{w}"] = base.rolling(w).std()
    return out


def add_month_over_month_change(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Add month-over-month percentage change, computed on lagged values to avoid leakage."""
    out = df.copy()
    for col in columns:
        pct = out[col].shift(1).pct_change()
        out[f"{col}_mom_pct"] = pct.replace([np.inf, -np.inf], np.nan)
    return out


def add_seasonal_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar month dummies (11 dummies, December is the reference category)."""
    out = df.copy()
    month_dummies = pd.get_dummies(out.index.month, prefix="month", drop_first=True)
    month_dummies.index = out.index
    return pd.concat([out, month_dummies], axis=1)


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    n_lags: int = 3,
    include_mom_pct: bool = True,
) -> pd.DataFrame:
    """
    Build the full leakage-safe feature matrix for one-step-ahead nowcasting of target_col.

    All predictors (including the target's own lags) use only information available at
    time t-1 or earlier, consistent with a genuine real-time nowcasting exercise.

    ``include_mom_pct`` can be set to False to skip the month-over-month percentage-change
    features. This is useful for series with an extended zero-search-volume history (e.g.
    a Google Trends query that had no measurable search interest until years into the
    sample): a 0-to-0 percentage change is undefined, so ``dropna()`` would otherwise erase
    every row until the series' first stable nonzero stretch, needlessly shrinking the
    usable sample for analyses that don't require this specific feature.
    """
    feat = add_lags(df, [target_col] + predictor_cols, n_lags)
    feat = add_rolling_stats(feat, [target_col] + predictor_cols)
    if include_mom_pct:
        feat = add_month_over_month_change(feat, predictor_cols)
    feat = add_seasonal_dummies(feat)

    # Drop contemporaneous predictor columns (keep target_col as the label, drop raw
    # contemporaneous predictors since only their lagged versions are legitimate features).
    feat = feat.drop(columns=predictor_cols)
    feat = feat.dropna().copy()
    return feat
