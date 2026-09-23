# Signal or Shock Robustness? Disentangling a Gradient-Boosted Nowcasting Edge for U.S. Unemployment

[![Author](https://img.shields.io/badge/author-prakash--ukhalkar-181717?logo=github)](https://github.com/prakash-ukhalkar)
[![Repo](https://img.shields.io/badge/repo-econ--ml--nowcasting-blue?logo=github)](https://github.com/prakash-ukhalkar/econ-ml-nowcasting)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/interpretability-SHAP-9cf)](https://shap.readthedocs.io/)
[![Data: FRED](https://img.shields.io/badge/data-FRED-013243)](https://fred.stlouisfed.org/)
[![Data: Google Trends](https://img.shields.io/badge/data-Google%20Trends-4285F4?logo=google)](https://trends.google.com/)

## 1. Motivation

Standard macroeconomic nowcasting relies on autoregressive (AR) and dynamic-factor
models built on officially released, low-frequency series (e.g., monthly unemployment,
quarterly GDP). These releases are lagged and revised. High-frequency behavioral proxies,
such as Google Trends search-interest indices, are available in near real time and may
contain information about the state of the economy before official data are released.

This project set out to ask a narrow question: **does adding Google Trends search-trend
features to a gradient-boosted tree model improve out-of-sample nowcasting accuracy for
the U.S. unemployment rate relative to a standard ARIMA benchmark?** An XGBoost model
does show a 16.3% RMSE improvement over ARIMA on a standard 60-month holdout, but that
result does not survive scrutiny. It is not statistically significant, it reverses sign in
shorter holdout windows, and four targeted follow-up checks (a linear model matched to the
same feature set, exclusion of the 2020 COVID-19 shock months, replication in the
2008-2009 financial crisis, and SHAP feature attribution) give a mixed verdict that
confirms neither "the search-trend data carries real signal" nor "the improvement is pure
shock robustness."

The project's actual contribution ended up being methodological rather than substantive: a
concrete demonstration of how fragile a single-window machine-learning-beats-benchmark
nowcasting comparison is, plus a reusable four-check diagnostic protocol for catching that
kind of overclaiming before publication. See §6 below for a summary of the results.

## 2. Data (100% publicly available, no paid licenses)

| Series | Source | Frequency | Access |
|---|---|---|---|
| Civilian Unemployment Rate (`UNRATE`), target | FRED (Federal Reserve Bank of St. Louis) | Monthly | Free API, no key required for CSV; free API key for `fredapi` |
| Initial Jobless Claims (`ICSA`) | FRED | Weekly (resampled monthly) | Free |
| Industrial Production Index (`INDPRO`) | FRED | Monthly | Free |
| All Employees, Total Nonfarm (`PAYEMS`) | FRED | Monthly | Free |
| Search interest: "unemployment office", "file for unemployment", "jobs near me" | Google Trends (`pytrends`) | Weekly (resampled monthly) | Free, no key |

All data are pulled programmatically in `notebooks/01_data_acquisition.ipynb` /
`src/data_acquisition.py` and cached to `data/raw/` and `data/processed/` so the full
pipeline is reproducible offline after the first run. Both directories are committed to
this repository (not gitignored) for exact reproducibility of the reported results.

**Google Trends reproducibility note.** Unlike FRED's stable, versioned historical CSVs,
`pytrends` returns *relative* search-interest indices that are re-normalized to the
specific keyword set and query date of each pull. A re-pull on a different day will not
reproduce the exact index values used in this note, even for the same historical months.
The Google Trends series used for all reported results were pulled on **2026-09-23**, and
the raw output is committed at `data/raw/google_trends.csv` so the manuscript's numbers
remain exactly reproducible from this repository alone, without depending on a fresh API
pull.

## 3. Method

1. **Data acquisition**: pull FRED + Google Trends series, align to monthly frequency.
2. **EDA**: stationarity checks (ADF/KPSS), seasonality, missingness, correlation with target.
3. **Feature engineering**: lags, rolling statistics, month-over-month deltas, seasonal
   dummies, all leakage-safe (every feature for month *t* uses only information dated
   *t*-1 or earlier).
4. **Benchmark model**: ARIMA(1,0,1) on `UNRATE` alone (classical nowcasting baseline).
5. **Linear baseline**: Ridge regression on the identical lagged feature set as the
   XGBoost challenger, to separate "more predictors" from "nonlinear model" effects.
6. **Challenger model**: XGBoost gradient-boosted trees using lagged FRED + Google Trends features.
7. **Evaluation**: expanding-window, one-step-ahead out-of-sample forecasts; RMSE/MAE,
   Diebold-Mariano and Clark-West tests for equal predictive accuracy, a moving-block
   bootstrap confidence interval on the RMSE improvement, and a robustness sweep across
   four holdout window lengths (36/48/60/84 months).
8. **Mechanism checks**: three post hoc diagnostics run after observing the headline
   result, none pre-registered. Re-estimation with the COVID-19 shock months excluded,
   replication in the independent 2008-2009 financial crisis under a matched feature
   specification, and a SHAP re-check of whether Google Trends features actually rank
   meaningfully in feature importance.

## 4. Repository structure

```
econ-ml-nowcasting/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # untouched pulls from FRED / Google Trends (committed)
│   └── processed/      # cleaned, merged, feature-engineered panel (committed)
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_benchmark.ipynb
│   ├── 05_modeling_ml.ipynb
│   ├── 06_evaluation_and_results.ipynb
│   └── 07_shock_robustness_checks.ipynb
├── src/
│   ├── __init__.py
│   ├── data_acquisition.py
│   ├── features.py       # feature engineering, incl. include_mom_pct flag for
│   │                      #   coverage back to 2008-09 (see notebook 07)
│   ├── models.py          # ARIMA benchmark, Ridge baseline, XGBoost challenger
│   ├── evaluate.py        # RMSE/MAE, Diebold-Mariano, Clark-West, bootstrap CI
│   └── utils.py           # config/path helpers, exclude_date_range, shap/xgboost compat shim
├── tests/
│   └── test_features.py
└── reports/
    ├── table_*.csv        # every reported table, machine-generated
    └── fig_*.png          # every reported figure, machine-generated
```

`submission_files/` (a journal-formatted manuscript, title page, and cover letter for
Economics Bulletin) is prepared locally but excluded from version control via
`.gitignore`, since it contains submission-workflow drafts rather than analysis code.

## 5. Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numeric order (`01` to `07`). Each notebook can also be executed
headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb
```

An optional free FRED API key (https://fredapi.stlouisfed.org, instant signup) can be
set as the environment variable `FRED_API_KEY` for the `fredapi` client; a no-key CSV
fallback (`pandas_datareader` / direct FRED CSV endpoints) is used automatically if the
key is absent, so the pipeline runs with zero configuration.

## 6. Key findings

- XGBoost beats ARIMA by 16.3% RMSE on the primary 60-month holdout (2020:01-2024:12),
  but the improvement is not significant (Diebold-Mariano p = 0.31; Clark-West p = 0.12)
  and a 90% moving-block bootstrap CI on the improvement spans -61.8% to +44.9%.
- The result is not robust to holdout window length: XGBoost *underperforms* ARIMA by
  40-80% in shorter, calmer post-2020 windows, and only outperforms in windows that
  include the 2020 shock.
- A Ridge model on XGBoost's identical feature set collapses catastrophically during
  COVID-19 (predicting 37.7% unemployment in April 2020 against an actual 14.8%),
  initial evidence for a shock-robustness explanation.
- That explanation does not survive further testing: excluding the acute COVID-shock
  months *increases* XGBoost's edge (to +53%) rather than removing it, and under a
  feature specification held fixed across two shock episodes, XGBoost significantly
  *underperforms* ARIMA in 2008-2009 (-20.5%, p < 0.001) while outperforming it in 2020
  (+10.5%), the opposite ranking a general shock-robustness account would predict.
- SHAP attribution shows a Google Trends feature ranks 3rd of 66 by importance,
  undermining a "the model ignores search data" reading.

Full result tables and figures underlying these findings are in `reports/`.

## License

MIT, see `LICENSE`. Data are redistributed under FRED's and Google Trends' respective
public-use terms; no proprietary data are included in this repository.
