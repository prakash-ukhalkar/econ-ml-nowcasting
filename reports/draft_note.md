# Signal or Shock Robustness? Disentangling a Gradient-Boosted Nowcasting Edge for U.S. Unemployment

**JEL classification:** C53, C55, E24, E27
**Keywords:** nowcasting, machine learning, gradient boosting, Google Trends, unemployment rate,
model robustness, distributional shift

> *Draft note (letters format, ≤7 printed pages exclusive of references, tables, figures,
> and appendices).*

---

## Abstract

A single-window RMSE comparison showing a machine learning model beating a classical
benchmark is a common building block of applied nowcasting notes — and, we show here, an
unreliable one. Using U.S. unemployment nowcasting with FRED and Google Trends predictors
as a concrete test case, we document a 16.3% RMSE improvement of XGBoost over an ARIMA
benchmark on a standard 60-month holdout, then apply a battery of checks a referee should
demand before accepting it: the improvement is not statistically significant
(Diebold-Mariano p = 0.31; Clark-West p = 0.12), reverses sign in shorter holdout windows
(40-80% *underperformance* pre-2020-shock), and a Ridge model on XGBoost's identical feature
set collapses catastrophically during COVID (predicting 37.7% unemployment in April 2020
against an actual 14.8%) — all consistent with the improvement being an artifact of
tree-based robustness to a single rare shock rather than genuine information in the Google
Trends predictors. We then test that shock-robustness explanation directly, and it does not
survive testing either: excluding the four acute COVID-shock months from training and
evaluation *increases* XGBoost's edge (to +53%) rather than eliminating it; under a feature
specification held fixed across both episodes, XGBoost significantly *underperforms* ARIMA
in an independent shock (the 2008-2009 financial crisis: -20.5%, p < 0.001) while
outperforming it in 2020 (+10.5%) — the opposite ranking a generic shock-robustness account
would predict; and SHAP attribution shows a Google Trends feature ranks 3rd of 66 by
importance, undermining the "the model ignores search data" reading. **The applied
contribution is a negative result on search-trend nowcasting; the methodological
contribution — the one we emphasize — is a demonstration that a single-window comparison,
even a large and clean-looking one, is not sufficient evidence for either "the new predictor
works" or "it's just robustness," and a compact, reusable checklist (matched linear
baseline, shock-exclusion re-estimation, second-shock replication, SHAP attribution) for
telling the difference.**

---

## 1. Introduction

Timely assessment of labor-market conditions is central to monetary and fiscal
policymaking, yet official unemployment statistics are released with a lag and are
subsequently revised. A growing "nowcasting" literature (Bok et al., 2018; Choi and Varian,
2012) asks whether high-frequency, freely available proxies — search engine query volumes
foremost among them — can sharpen short-horizon forecasts of macroeconomic aggregates ahead
of, or in place of, official releases. A parallel and more recent literature has documented
that machine learning forecasters, and tree-based ensembles in particular, behave very
differently from linear benchmarks during the extreme, short-lived distributional shift of
the 2020 pandemic: Goulet Coulombe et al. (2022) decompose *why* machine learning helps
macroeconomic forecasting into four distinct mechanisms (nonlinearity, regularization,
cross-validation, and alternative loss functions) rather than treating "ML" as a monolith,
and Goulet Coulombe (2024) develops a random-forest-based macro model whose bounded,
nonparametric response functions are argued to be structurally more robust to outlier
observations than linear autoregressive or factor models — directly relevant to a setting,
like ours, where a single extreme episode (2020) dominates a short evaluation window.
Medeiros et al. (2021) more generally document accuracy gains from regularized/nonlinear ML
methods in macro forecasting outside crisis periods, providing a useful contrast case where
the mechanism is closer to genuine signal extraction than shock robustness.

This note sits at the intersection of these two literatures, and its contribution is
methodological rather than substantive: we show, in a single concrete case, how easily a
"machine learning beats the benchmark" result driven by shock robustness can be mistaken for
a result driven by a novel predictor (here, Google Trends), and we develop a small set of
diagnostic checks — feature-set-matched linear baselines, shock-exclusion re-estimation,
second-shock replication, and SHAP-based attribution — that a researcher can run before
drawing that conclusion. In our case, running these checks does **not** cleanly resolve the
mechanism; we report the resulting ambiguity honestly, which we regard as more useful to the
literature than a falsely clean story in either direction.

## 2. Data

All series are public and freely accessible (see `README.md` §2 and
`notebooks/01_data_acquisition.ipynb` for the full extraction pipeline):

- **Target:** U.S. civilian unemployment rate, `UNRATE`, monthly, seasonally adjusted (FRED).
- **Predictors:** initial jobless claims (`ICSA`, FRED, weekly, resampled to monthly mean),
  industrial production index (`INDPRO`, FRED), nonfarm payrolls (`PAYEMS`, FRED), and three
  Google Trends search-interest indices ("file for unemployment", "unemployment office",
  "jobs near me"; weekly, resampled to monthly mean).
- **Sample period:** raw series span 2004:01-2024:12 (monthly frequency); after constructing
  lags, 6-month rolling statistics, and month-over-month changes, the leakage-safe feature
  matrix spans 2013:03-2024:12 (141 monthly observations). The primary evaluation holds out
  the final 60 months (2020:01-2024:12); §4.4 also considers 36-, 48-, and 84-month holdouts,
  and §4.5 considers a COVID-excluded 60-month window and an independent 2008-2009 window.
- **Google Trends reproducibility.** Google Trends indices are relative, re-normalized to
  the specific keyword set and pull date — a fresh pull will not reproduce our exact values.
  Our series were pulled on **2026-09-23**; the raw pull is committed at
  `data/raw/google_trends.csv` (see `README.md` §2) so the results in this note are exactly
  reproducible without a live API call.
- **A specific data limitation, load-bearing for §4.5.2:** the query "jobs near me" recorded
  *zero* measurable search interest from 2004 through 2012 (Figure, `notebooks/07`). This is
  why the main feature matrix only starts in 2013:03 (a 0-to-0 percentage change is
  undefined) and why the 2008-2009 mechanism check in §4.5.2 necessarily uses a reduced
  feature set.

## 3. Empirical strategy

### 3.1 Feature construction

To respect the real-time nature of nowcasting, every feature used to predict month *t* is
constructed from information dated *t*-1 or earlier: lags 1-3 of the target and of each
predictor, rolling 3- and 6-month means/standard deviations (computed on lagged values),
month-over-month percentage changes of each predictor, and calendar-month dummies. This
construction is implemented and unit-tested in `src/features.py` / `tests/test_features.py`
and verified for leakage in `notebooks/03_feature_engineering.ipynb`.

### 3.2 Models

- **Benchmark:** ARIMA(1,0,1) fit on the target series alone (`src/models.py:
  run_arima_benchmark`).
- **Linear baseline:** Ridge regression ($\alpha=1$, sklearn default) trained on the
  *identical* lagged feature set as XGBoost, with features standardized on the training fold
  only at each refit (`src/models.py: run_ridge_challenger`). This baseline isolates the
  effect of the richer feature set (more predictors) from the effect of the nonlinear
  tree-ensemble model class.
- **Challenger:** XGBoost regressor trained on the full lagged feature set (`src/models.py:
  run_ml_challenger`).

All three models are evaluated under an identical **expanding-window, one-step-ahead**
protocol: at each date in the held-out evaluation window, each model is refit using only
data strictly before that date and produces a single one-step-ahead forecast. We repeat the
full evaluation for four holdout lengths (36/48/60/84 months, §4.4) and for two shock-focused
re-specifications (§4.5).

### 3.3 Evaluation

We report RMSE and MAE over each held-out window, and test the null of equal predictive
accuracy two ways: **Diebold-Mariano (1995)** (`src/evaluate.py: diebold_mariano_test`), and
**Clark-West (2007)** (`src/evaluate.py: clark_west_test`), an MSPE-adjusted, one-sided test
for *nested* model comparisons, appropriate here because ARIMA's information set is a strict
subset of what Ridge and XGBoost see (plain DM is known to be undersized in this case). We
additionally report a **moving-block bootstrap** 90% confidence interval (`src/evaluate.py:
moving_block_bootstrap_rmse_ci`; 2,000 resamples, 6-month blocks) around the RMSE
improvement, and SHAP (Lundberg and Lee, 2017) feature attributions for the XGBoost model.

Beyond the primary specification, three mechanism checks (`notebooks/07_shock_robustness_checks.ipynb`)
probe *why* any XGBoost-ARIMA gap exists: (i) re-estimating with the 2020:03-2020:06 COVID
shock months masked out of training and evaluation; (ii) replicating the comparison around
the independent 2008-2009 financial-crisis shock; and (iii) checking whether Google Trends
features rank meaningfully in SHAP importance at all.

## 4. Results

All figures/tables in this section are generated by `notebooks/06_evaluation_and_results.ipynb`
and `notebooks/07_shock_robustness_checks.ipynb`, and reproduced exactly in `reports/table_*.csv`.

### 4.1 Head-to-head accuracy (60-month holdout)

**Table 1. Out-of-sample forecast accuracy, 60-month evaluation window (2020:01-2024:12)**

| Model | RMSE | MAE | RMSE improvement vs. ARIMA |
|---|---|---|---|
| ARIMA(1,0,1) (benchmark) | 1.695 | 0.547 | — |
| Ridge (linear, XGBoost feature set) | 3.176 | 0.903 | -87.4% |
| XGBoost (challenger) | 1.418 | 0.470 | +16.3% |

Ridge — trained on the identical feature set as XGBoost, differing only in being linear
rather than tree-based — is *dramatically worse* than both alternatives. Inspecting
individual errors shows this is driven almost entirely by two months: Ridge predicts 37.7%
and -5.0% unemployment for April-May 2020 (actual: 14.8% and 13.2%), a catastrophic
extrapolation failure at the onset of the COVID-19 shock that gradient-boosted trees cannot
exhibit (tree predictions are bounded to the range of leaf values seen in training). This is
the first piece of evidence for a shock-robustness explanation of XGBoost's edge.

### 4.2 Significance tests

**Table 2. Diebold-Mariano test (two-sided)**

| Comparison | DM statistic | p-value |
|---|---|---|
| ARIMA vs. XGBoost | 1.007 | 0.314 |
| ARIMA vs. Ridge | -1.530 | 0.126 |
| Ridge vs. XGBoost | 1.567 | 0.117 |

**Table 3. Clark-West test (one-sided, nested models)**

| Comparison | CW statistic | p-value (one-sided) |
|---|---|---|
| ARIMA (restricted) vs. XGBoost (unrestricted) | 1.171 | 0.121 |
| ARIMA (restricted) vs. Ridge (unrestricted) | -1.436 | 0.924 |

None of the pairwise comparisons reach conventional significance at the 5% level.

### 4.3 Bootstrap confidence interval on the RMSE improvement

**Table 4. Moving-block bootstrap, 90% CI (2,000 resamples, 6-month blocks)**

| Comparison | Point estimate | 90% CI |
|---|---|---|
| XGBoost vs. ARIMA | +16.3% | [-61.8%, +44.9%] |
| Ridge vs. ARIMA | -87.4% | [-147.4%, +13.2%] |

The confidence interval on the headline XGBoost improvement is wide and straddles zero.

### 4.4 Robustness across holdout window length

**Table 5. RMSE improvement vs. ARIMA by test-window length**

| Window (months) | ARIMA RMSE | Ridge RMSE | XGBoost RMSE | XGBoost improvement | DM p-value |
|---|---|---|---|---|---|
| 36 | 0.182 | 0.204 | 0.328 | -80.4% | 0.169 |
| 48 | 0.225 | 0.213 | 0.317 | -40.7% | 0.232 |
| 60 | 1.695 | 3.176 | 1.418 | +16.3% | 0.314 |
| 84 | 1.434 | 2.686 | 1.200 | +16.3% | 0.315 |

The sign of the XGBoost-vs-ARIMA comparison flips with the holdout window: XGBoost
underperforms by 40-80% in the two shorter, calmer windows (2021-2024) and only outperforms
in the two longer windows that include the 2020 shock. No window length reaches significance.

**Figure 1.** One-step-ahead nowcasts, actual vs. ARIMA vs. Ridge vs. XGBoost, 60-month
window (`reports/fig_forecast_comparison.png`).

**Figure 2.** SHAP summary plot, XGBoost feature attributions (`reports/fig_shap_summary.png`).

**Figure 3.** RMSE improvement and DM significance across holdout window lengths
(`reports/fig_robustness_windows.png`).

### 4.5 Mechanism checks: is this really about shock robustness?

Sections 4.1 and 4.4 are consistent with a shock-robustness story — but consistency is not
confirmation. We ran three targeted checks (`notebooks/07_shock_robustness_checks.ipynb`).
**None of the three cleanly confirms the story, and one directly contradicts it.**

#### 4.5.1 COVID-exclusion test

We drop 2020:03-2020:06 (the four acute shock months) from both training and evaluation and
re-run the identical 60-month-holdout protocol (the resulting window shifts back to
2019:09-2024:12 to hold the evaluated sample size fixed at 60 months).

**Table 6. With vs. without the acute COVID-shock months**

| Condition | ARIMA RMSE | XGBoost RMSE | XGBoost improvement | DM p-value | CW p-value (one-sided) |
|---|---|---|---|---|---|
| With COVID (original, 2020:01-2024:12) | 1.695 | 1.418 | +16.3% | 0.314 | 0.121 |
| COVID excluded (2019:09-2024:12 minus 4 months) | 0.938 | 0.440 | **+53.1%** | 0.257 | 0.094 |

**This is the opposite of what the shock-robustness hypothesis predicted.** If XGBoost's
edge were driven specifically by the four acute spike months, removing them should shrink
the improvement toward zero. Instead it *nearly quadruples*, to +53%. The likely explanation
is that the excluded-COVID window still retains the broader, elevated-volatility 2020-2022
labor-market recovery (unemployment swung from double digits back to pre-pandemic levels
over roughly two years), against which ARIMA's single-order linear structure still performs
comparatively poorly relative to XGBoost — i.e., if a shock-robustness mechanism is at work
at all, it concerns robustness to a *prolonged elevated-volatility regime*, not narrowly to
the four most extreme months. This materially weakens the narrow version of the working
hypothesis this check was designed to test.

#### 4.5.2 Second-shock check: 2008-2009 financial crisis

Because the Google Trends feature "jobs near me" has no measurable search volume before
2013 (§2), reaching back to 2008-2009 requires a feature matrix without month-over-month
Trends changes (`build_feature_matrix(..., include_mom_pct=False)`) — a genuinely different,
slightly poorer feature set than §4.1-4.4. A naive comparison of this reduced-feature-set
2008-2009 result against the main-specification 2020 result would confound the shock-period
effect with the feature-set change, so **we instead re-estimate the 2020 window under the
identical reduced feature set** used for 2008-2009. This keeps the two shock-window rows in
Table 7 on a single, matched specification; the main-specification 2020 result is retained
only as a reference. The pre-2009 training sample is also considerably shorter (~39 months
of lead-in vs. ~80 for the main results), which should still be kept in mind.

**Table 7. 2008-2009 crisis vs. 2020 COVID, identical (reduced) feature specification**

| Window | ARIMA RMSE | Ridge RMSE | XGBoost RMSE | XGBoost improvement | DM statistic | DM p-value |
|---|---|---|---|---|---|---|
| 2008-2009 crisis (2008:01-2009:12) | 0.274 | 0.261 | 0.330 | **-20.5%** | -3.558 | **<0.001** |
| 2020 COVID, matched spec (2020:01-2024:12) | 1.695 | 2.481 | 1.517 | +10.5% | 0.712 | 0.476 |
| *2020 COVID, main spec (reference only)* | *1.695* | *3.176* | *1.418* | *+16.3%* | *1.007* | *0.314* |

**Under a feature specification held fixed across both shock windows, the two episodes still
rank the models in opposite order.** XGBoost outperforms ARIMA in the matched 2020 window
(+10.5%, not significant) but *significantly underperforms* both ARIMA and Ridge in the
2008-2009 window (-20.5%, DM p < 0.001, favoring ARIMA). Because the feature-set confound
from the earlier, unmatched version of this check is now removed, this is a cleaner result
than before, and it still contradicts a generic "gradient-boosted trees are robust to macro
shocks" claim. The remaining, harder-to-remove confound is training-sample size: XGBoost
in the 2008-2009 window has roughly half the training history available to the 2020 window,
which independently handicaps a 300-tree ensemble regardless of any shock-robustness
property. We cannot fully separate "2020 was COVID-specific" from "2008-2009 was
data-starved" with the data available, and we report the contradiction as-is.

#### 4.5.3 SHAP re-check: does the model actually use the Trends features?

We re-ranked all 66 features by mean |SHAP value| on the full main-specification model.

**Table 8. Top 10 features by SHAP importance**

| Rank | Feature | Mean \|SHAP\| | Source |
|---|---|---|---|
| 1 | `ICSA_lag1` | 0.252 | FRED |
| 2 | `UNRATE_lag1` | 0.238 | FRED (target) |
| 3 | `unemployment office_rollmean6` | 0.206 | **Google Trends** |
| 4 | `PAYEMS_lag1` | 0.118 | FRED |
| 5 | `unemployment office_rollmean3` | 0.109 | **Google Trends** |
| 6 | `UNRATE_rollmean3` | 0.098 | FRED (target) |
| 7 | `PAYEMS_lag2` | 0.078 | FRED |
| 8 | `UNRATE_lag2` | 0.061 | FRED (target) |
| 9 | `PAYEMS_lag3` | 0.060 | FRED |
| 10 | `file for unemployment_lag1` | 0.048 | **Google Trends** |

Full ranking: `reports/table_shap_feature_ranking.csv`. Figure:
`reports/fig_shap_rank_by_source.png`.

**This also complicates the "it's not the search data" reading.** The best-ranked Google
Trends feature (`unemployment office_rollmean6`) places 3rd of 66 overall, ahead of all but
two FRED/target-derived features, and Trends-derived features occupy 3 of the top 10 slots.
If XGBoost's forecasts were driven purely by lagged `UNRATE`/`ICSA` with Trends carried along
inertly, we would expect Trends features to cluster near the bottom of the ranking; instead
they are interspersed near the top. SHAP importance reflects how much a feature moves
predictions across the sample, not necessarily genuine *incremental* out-of-sample
information content (a feature can be important within-sample while still adding forecast
noise out-of-sample, which is one way to reconcile this with §4.1-4.4) — but taken at face
value, this finding is evidence *against* a clean "the model ignores search trends" story.

### 4.6 Synthesis of the mechanism checks

The three checks in §4.5 do not converge on a single explanation:

- **Consistent with shock robustness:** Ridge's catastrophic COVID collapse (§4.1); the
  general pattern that XGBoost's edge is concentrated in shock-inclusive windows (§4.4).
- **Inconsistent with a narrow shock-robustness story:** excluding the acute COVID months
  *increases* rather than eliminates the XGBoost edge (§4.5.1); under a feature
  specification matched across both episodes, the 2008-2009 crisis shows the *opposite*
  ranking, with XGBoost significantly underperforming ARIMA and Ridge (§4.5.2).
- **Inconsistent with a "the search data doesn't matter" story:** Google Trends features
  rank highly by SHAP importance, not negligibly (§4.5.3).

We take this as genuine evidence that **the mechanism behind the original 16.3% point
estimate is not resolved by these checks**, and we are not willing to assert either "it's
shock robustness, not signal" or "it's genuine Trends signal" as a conclusion the data
support. What the checks *do* establish is that the original single-window comparison
substantially understated how fragile and context-dependent the result is — which is itself
the paper's main contribution.

## 5. Discussion and conclusion

The evidence in this note does **not** support a firm conclusion that Google Trends
search-interest data improve short-horizon nowcasts of the U.S. unemployment rate, nor does
it support a firm alternative conclusion that XGBoost's apparent edge is *simply* an
artifact of shock robustness. What it supports is a narrower, more defensible claim: **a
single-window ARIMA-vs-XGBoost RMSE comparison in this setting is not a reliable basis for
either conclusion.** The headline 16.3% improvement is not statistically significant, is not
stable across holdout window lengths, reverses when the specific shock months are excluded
(in the "wrong" direction for the shock-robustness hypothesis), does not replicate in an
earlier, independent shock under a matched feature specification, and coexists with
genuinely non-trivial SHAP importance for the search-trend features. Any one of these
findings alone might have been dismissed as noise; together, they indicate the underlying
data-generating process here is not well characterized by either candidate story at this
sample size — which is itself the point: **a researcher who stopped at Table 1 would have
reported a false positive with unusual confidence**, since 16.3% on a clean 60-month
out-of-sample comparison looks, on its face, like a solid result.

**Limitations.** Three are worth stating explicitly. First, **statistical power**: even the
longest evaluation window used here has only 84 test-set months, and the 2008-2009 mechanism
check uses only 24 — samples this small give DM/CW tests limited power to distinguish "no
effect" from "effect too small to detect," and the wide bootstrap CI in §4.3 reflects exactly
this. Second, **training-sample size in the 2008-2009 check**: even after matching the
feature specification across both shock windows (§4.5.2), XGBoost's 2008-2009 training fold
is roughly half the length of its 2020 counterpart, which independently disadvantages a
300-tree ensemble; we cannot fully separate this from a genuine COVID-specific effect. Third,
**Google Trends reproducibility**: because Trends indices are re-normalized per pull, results
reported here rest on a Trends pull with a specific, documented pull date (2026-09-23) and
are only exactly reproducible through the raw file committed to this repository
(`data/raw/google_trends.csv`); an independent re-pull, even for identical historical months,
would not produce byte-identical input data (see §2).

We view this as a preliminary, honestly unresolved empirical result attached to a
methodological point we are confident in: single-window ML-vs-benchmark comparisons in
applied macro nowcasting should be treated as a starting hypothesis, not a finding, until
subjected to checks of this kind. Natural extensions include a longer sample spanning more
shock episodes with a consistent training-sample size (which would require either a longer
Google Trends history or a data source without the "jobs near me"-style zero-search-volume
problem), additional targets and countries, and model classes that could separate genuine
predictor information from shock robustness more directly — for instance, explicit shock
indicators or quantile/robust loss functions that decouple a model's crisis-period behavior
from its ordinary-times behavior.

## References

1. Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., and Tambalotti, A. (2018).
   "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics*,
   10, 615-643.
2. Choi, H., and Varian, H. (2012). "Predicting the Present with Google Trends."
   *Economic Record*, 88(s1), 2-9.
3. Clark, T. E., and West, K. D. (2007). "Approximately Normal Tests for Equal Predictive
   Accuracy in Nested Models." *Journal of Econometrics*, 138(1), 291-311.
4. Diebold, F. X., and Mariano, R. S. (1995). "Comparing Predictive Accuracy."
   *Journal of Business & Economic Statistics*, 13(3), 253-263.
5. Goulet Coulombe, P., Leroux, M., Stevanovic, D., and Surprenant, S. (2022). "How Is
   Machine Learning Useful for Macroeconomic Forecasting?" *Journal of Applied Econometrics*,
   37(5), 920-964.
6. Goulet Coulombe, P. (2024). "The Macroeconomy as a Random Forest." *Journal of Applied
   Econometrics*, 39(3), 401-421.
7. Lundberg, S. M., and Lee, S.-I. (2017). "A Unified Approach to Interpreting Model
   Predictions." *Advances in Neural Information Processing Systems*, 30.
8. Medeiros, M. C., Vasconcelos, G. F. R., Veiga, A., and Zilberman, E. (2021). "Forecasting
   Inflation in a Data-Rich Environment: The Benefits of Machine Learning Methods." *Journal
   of Business & Economic Statistics*, 39(1), 98-119.

---

*All data are public (FRED, Google Trends); no proprietary or restricted-access data are
used. Google Trends pull date: 2026-09-23 (see §2 and `README.md` §2 for reproducibility
details).*
