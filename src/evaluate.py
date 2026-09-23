"""
Evaluation metrics and out-of-sample predictive-accuracy tests:
  * Diebold & Mariano (1995) -- general test for equal predictive accuracy.
  * Clark & West (2007) -- MSPE-adjusted test for equal predictive accuracy between
    *nested* models (the restricted model's predictors are a subset of the
    unrestricted model's), where the standard DM test is known to be undersized.
  * A moving-block bootstrap confidence interval on the RMSE improvement of one
    model over another, to complement the point estimate with an uncertainty band.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy between two forecast error series
    (e1, e2 = y_true - y_pred for model 1 and model 2), using a squared-error loss
    differential. Returns (DM statistic, two-sided p-value).

    Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy," JBES.
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_bar = d.mean()

    # Newey-West style long-run variance with h-1 lags (h=1 for one-step-ahead forecasts).
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma = np.cov(d[lag:], d[:-lag])[0, 1]
        var_d += 2 * gamma
    var_d /= n

    dm_stat = d_bar / np.sqrt(var_d) if var_d > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)


def clark_west_test(
    y_true: np.ndarray, pred_restricted: np.ndarray, pred_unrestricted: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """
    Clark & West (2007) MSPE-adjusted test for equal predictive accuracy between two
    *nested* forecasting models, where ``pred_restricted`` comes from a model whose
    predictor set is a strict subset of the one that produced ``pred_unrestricted``
    (e.g. an AR benchmark nested inside an ML model that also uses the AR lags).

    Under nesting, the unrestricted model's extra parameters add noise to its
    point forecasts even when their true coefficients are zero, which biases the
    standard Diebold-Mariano test toward *not* rejecting the null even when the
    larger model is truly better. The CW adjustment removes this bias by
    subtracting the unrestricted model's own excess variance from the loss
    differential before testing.

    The test is one-sided: a large positive CW statistic is evidence that the
    unrestricted (larger) model improves on the restricted (nested) benchmark.

    Returns (CW statistic, one-sided p-value for H1: unrestricted model is better).

    Reference: Clark, T. E., and West, K. D. (2007). "Approximately Normal Tests for
    Equal Predictive Accuracy in Nested Models." Journal of Econometrics, 138(1), 291-311.
    """
    y_true = np.asarray(y_true)
    pred_restricted = np.asarray(pred_restricted)
    pred_unrestricted = np.asarray(pred_unrestricted)

    e_restricted = y_true - pred_restricted
    e_unrestricted = y_true - pred_unrestricted
    adjustment = (pred_restricted - pred_unrestricted) ** 2

    f = e_restricted ** 2 - (e_unrestricted ** 2 - adjustment)
    n = len(f)
    f_bar = f.mean()

    gamma0 = np.var(f, ddof=0)
    var_f = gamma0
    for lag in range(1, h):
        gamma = np.cov(f[lag:], f[:-lag])[0, 1]
        var_f += 2 * gamma
    var_f /= n

    cw_stat = f_bar / np.sqrt(var_f) if var_f > 0 else np.nan
    p_value = float(1 - stats.norm.cdf(cw_stat))
    return float(cw_stat), p_value


def moving_block_bootstrap_rmse_ci(
    y_true: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_challenger: np.ndarray,
    block_size: int = 6,
    n_boot: int = 2000,
    ci: float = 0.90,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Moving-block bootstrap confidence interval for the percentage RMSE improvement of
    ``pred_challenger`` over ``pred_benchmark``. Blocking (rather than iid resampling)
    preserves the serial correlation present in monthly macro forecast errors.

    Returns a dict with the point estimate, the bootstrap CI bounds, and the
    bootstrap standard error.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_benchmark = np.asarray(pred_benchmark, dtype=float)
    pred_challenger = np.asarray(pred_challenger, dtype=float)
    n = len(y_true)

    e_benchmark = y_true - pred_benchmark
    e_challenger = y_true - pred_challenger

    point_rmse_benchmark = np.sqrt(np.mean(e_benchmark ** 2))
    point_rmse_challenger = np.sqrt(np.mean(e_challenger ** 2))
    point_estimate = (point_rmse_benchmark - point_rmse_challenger) / point_rmse_benchmark * 100

    rng = np.random.default_rng(random_state)
    n_blocks = int(np.ceil(n / block_size))
    boot_improvements = np.empty(n_boot)

    for b in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])[:n]

        rmse_benchmark_b = np.sqrt(np.mean(e_benchmark[idx] ** 2))
        rmse_challenger_b = np.sqrt(np.mean(e_challenger[idx] ** 2))
        boot_improvements[b] = (
            (rmse_benchmark_b - rmse_challenger_b) / rmse_benchmark_b * 100
            if rmse_benchmark_b > 0
            else np.nan
        )

    alpha = 1 - ci
    lower, upper = np.nanpercentile(boot_improvements, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate_pct": float(point_estimate),
        "ci_lower_pct": float(lower),
        "ci_upper_pct": float(upper),
        "ci_level": ci,
        "boot_se_pct": float(np.nanstd(boot_improvements, ddof=1)),
        "n_boot": n_boot,
        "block_size": block_size,
    }
