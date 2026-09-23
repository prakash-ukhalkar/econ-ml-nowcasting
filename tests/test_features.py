"""Unit tests for feature engineering leakage-safety and shape correctness."""
import numpy as np
import pandas as pd
import pytest

from src.features import add_lags, add_rolling_stats, build_feature_matrix


@pytest.fixture
def toy_df():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"y": rng.normal(5, 1, len(idx)), "x1": rng.normal(0, 1, len(idx))}, index=idx
    )


def test_add_lags_shifts_correctly(toy_df):
    out = add_lags(toy_df, ["y"], n_lags=2)
    assert (out["y_lag1"].iloc[1:].values == toy_df["y"].iloc[:-1].values).all()
    assert np.isnan(out["y_lag1"].iloc[0])


def test_rolling_stats_use_lagged_base_no_leakage(toy_df):
    out = add_rolling_stats(toy_df, ["y"], windows=[3])
    # rollmean at time t must not depend on y[t]
    manual = toy_df["y"].shift(1).rolling(3).mean()
    pd.testing.assert_series_equal(out["y_rollmean3"], manual, check_names=False)


def test_build_feature_matrix_has_no_contemporaneous_predictor(toy_df):
    feat = build_feature_matrix(toy_df, target_col="y", predictor_cols=["x1"], n_lags=2)
    assert "x1" not in feat.columns
    assert "y" in feat.columns
    assert feat.isna().sum().sum() == 0
