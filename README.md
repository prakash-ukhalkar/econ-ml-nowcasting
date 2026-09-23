# Can Search-Trend Data Improve Short-Horizon Macro Nowcasts? A Gradient-Boosting Approach

## 1. Motivation

Standard macroeconomic nowcasting relies on autoregressive (AR) and dynamic-factor
models built on officially released, low-frequency series (e.g., monthly unemployment,
quarterly GDP). These releases are lagged and revised. High-frequency behavioral proxies
— such as Google Trends search-interest indices — are available in near real time and may
contain information about the state of the economy before official data are released.

This project asks: **does adding search-trend features to a gradient-boosted tree model
improve out-of-sample nowcasting accuracy for the U.S. unemployment rate relative to a
standard AR benchmark?**

This is framed as a compact, falsifiable empirical note suitable for a letters-format
journal: one target variable, one benchmark, one challenger model, a clean rolling-origin
out-of-sample evaluation, and an interpretability check (SHAP) — extendable later into a
full-length paper (multiple targets, countries, or model classes).

## 2. Data (100% publicly available, no paid licenses)

| Series | Source | Frequency | Access |
|---|---|---|---|
| Civilian Unemployment Rate (`UNRATE`) | FRED (Federal Reserve Bank of St. Louis) | Monthly | Free API, no key required for CSV; free API key for `fredapi` |
| Initial Jobless Claims (`ICSA`) | FRED | Weekly (resampled monthly) | Free |
| Industrial Production Index (`INDPRO`) | FRED | Monthly | Free |
| ISM-proxy / ancillary series (optional) | FRED | Monthly | Free |
| Search interest: "unemployment office", "file for unemployment", "jobs near me" | Google Trends (`pytrends`) | Weekly (resampled monthly) | Free, no key |

All data are pulled programmatically in `notebooks/01_data_acquisition.ipynb` /
`src/data_acquisition.py` and cached to `data/raw/` and `data/processed/` so the full
pipeline is reproducible offline after the first run.

**Google Trends reproducibility note.** Unlike FRED's stable, versioned historical CSVs,
`pytrends` returns *relative* search-interest indices that are re-normalized to the
specific keyword set and query date of each pull — a re-pull on a different day will not
reproduce the exact index values used in this note, even for the same historical months.
The Google Trends series used for all reported results were pulled on **2026-09-23** and
the raw output is committed at `data/raw/google_trends.csv` (an explicit exception to the
`data/raw/` gitignore rule — see `.gitignore`) so the manuscript's numbers remain exactly
reproducible from this repository alone, without depending on a fresh API pull.

## 3. Method

1. **Data acquisition** — pull FRED + Google Trends series, align to monthly frequency.
2. **EDA** — stationarity checks (ADF/KPSS), seasonality, missingness, correlation with target.
3. **Feature engineering** — lags, rolling statistics, month-over-month deltas, seasonal dummies.
4. **Benchmark model** — AR(p) / ARIMA on `UNRATE` alone (classical nowcasting baseline).
5. **Challenger model** — gradient-boosted trees (XGBoost/LightGBM) using lagged FRED +
   Google Trends features.
6. **Evaluation** — expanding-window, one-step-ahead out-of-sample forecasts; compare
   RMSE / MAE / Diebold-Mariano test for equal predictive accuracy.
7. **Interpretability** — SHAP values to identify which features (and which lags) drive
   the ML model's edge, addressing the "black box" objection referees commonly raise.

## 4. Repository structure

```
econ-ml-nowcasting/
├── README.md
├── LICENSE
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/            # untouched pulls from FRED / Google Trends (FRED CSVs regenerable and
│   │                   #   gitignored; google_trends.csv is committed -- see §2 note below)
│   └── processed/      # cleaned, merged, feature-engineered panel (gitignored, regenerable)
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
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   └── test_features.py
└── reports/
    └── draft_note.md   # manuscript draft
```

## 5. Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numeric order (`01` → `06`). Each notebook can also be executed
headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_acquisition.ipynb
```

An optional free FRED API key (https://fredapi.stlouisfed.org — instant signup) can be
set as the environment variable `FRED_API_KEY` for the `fredapi` client; a no-key CSV
fallback (`pandas_datareader` / direct FRED CSV endpoints) is used automatically if the
key is absent, so the pipeline runs with zero configuration.

## 6. Manuscript status

Draft manuscript text (abstract, results, JEL codes) lives in
`reports/draft_note.md`. Results tables/figures are generated by
`notebooks/06_evaluation_and_results.ipynb` and referenced directly from the draft.

## License

MIT — see `LICENSE`. Data are redistributed under FRED's and Google Trends' respective
public-use terms; no proprietary data are included in this repository.
