"""Benchmark (AR/ARIMA) and challenger (gradient-boosted trees) nowcasting models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


@dataclass
class ExpandingWindowResult:
    dates: List[pd.Timestamp]
    y_true: List[float]
    y_pred: List[float]

    def to_frame(self, model_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"date": self.dates, "y_true": self.y_true, f"y_pred_{model_name}": self.y_pred}
        ).set_index("date")


def run_arima_benchmark(
    series: pd.Series, test_window_months: int, order: Tuple[int, int, int] = (1, 0, 1)
) -> ExpandingWindowResult:
    """
    Expanding-window, one-step-ahead ARIMA forecasts on the target series alone.

    At each step, refit ARIMA(order) on all data up to t-1 and forecast t. This mirrors
    a real-time nowcasting exercise: only information available at the forecast origin
    is used.
    """
    dates, y_true, y_pred = [], [], []
    n = len(series)
    start_idx = n - test_window_months

    for t in range(start_idx, n):
        train = series.iloc[:t]
        model = ARIMA(train, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=1).iloc[0]
        dates.append(series.index[t])
        y_true.append(series.iloc[t])
        y_pred.append(forecast)

    return ExpandingWindowResult(dates, y_true, y_pred)


def run_ml_challenger(
    feature_df: pd.DataFrame,
    target_col: str,
    test_window_months: int,
    xgb_params: dict,
    random_state: int = 42,
) -> Tuple[ExpandingWindowResult, XGBRegressor]:
    """
    Expanding-window, one-step-ahead XGBoost forecasts using lagged FRED + Trends features.

    Refits at each step on all data up to t-1, consistent with the ARIMA benchmark's
    real-time evaluation protocol so RMSE/MAE are directly comparable.
    """
    feature_cols = [c for c in feature_df.columns if c != target_col]
    X, y = feature_df[feature_cols], feature_df[target_col]

    dates, y_true, y_pred = [], [], []
    n = len(feature_df)
    start_idx = n - test_window_months

    last_model = None
    for t in range(start_idx, n):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test = X.iloc[[t]]

        model = XGBRegressor(random_state=random_state, **xgb_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        dates.append(feature_df.index[t])
        y_true.append(y.iloc[t])
        y_pred.append(pred)
        last_model = model

    return ExpandingWindowResult(dates, y_true, y_pred), last_model


def run_ridge_challenger(
    feature_df: pd.DataFrame,
    target_col: str,
    test_window_months: int,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Tuple[ExpandingWindowResult, Ridge]:
    """
    Expanding-window, one-step-ahead Ridge regression forecasts on the same lagged
    feature set used by the XGBoost challenger.

    This isolates the effect of the *feature set* (lagged FRED + Google Trends
    predictors) from the effect of the *model class* (linear vs. nonlinear/tree-based):
    Ridge uses identical inputs and the identical expanding-window protocol as
    ``run_ml_challenger``, so any remaining XGBoost-vs-Ridge gap is attributable to
    nonlinearity/interactions rather than to having more predictors than ARIMA.

    Features are standardized (mean/std fit on the training fold only, to avoid
    look-ahead leakage) before each refit, since Ridge penalizes coefficients on
    their raw scale.
    """
    feature_cols = [c for c in feature_df.columns if c != target_col]
    X, y = feature_df[feature_cols], feature_df[target_col]

    dates, y_true, y_pred = [], [], []
    n = len(feature_df)
    start_idx = n - test_window_months

    last_model = None
    for t in range(start_idx, n):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test = X.iloc[[t]]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]

        dates.append(feature_df.index[t])
        y_true.append(y.iloc[t])
        y_pred.append(pred)
        last_model = model

    return ExpandingWindowResult(dates, y_true, y_pred), last_model
