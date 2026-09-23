"""
Data acquisition for the macro nowcasting project.

All sources are free and public:
  * FRED (https://fred.stlouisfed.org) -- via `fredapi` if FRED_API_KEY is set in the
    environment, otherwise via the no-key CSV download endpoint
    (https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID).
  * Google Trends -- via `pytrends`, no key required.

Every function caches its raw pull to `data/raw/` so the pipeline is reproducible
offline after the first successful run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .utils import get_fred_api_key, get_path

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id: str, start_date: str = "2000-01-01") -> pd.Series:
    """Fetch a single FRED series, caching the raw CSV to data/raw/fred_<series_id>.csv."""
    cache_path = get_path(f"data/raw/fred_{series_id}.csv")

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        return df[series_id]

    api_key = get_fred_api_key()
    if api_key:
        from fredapi import Fred

        fred = Fred(api_key=api_key)
        series = fred.get_series(series_id, observation_start=start_date)
        series.name = series_id
        series.to_frame().rename_axis("date").to_csv(cache_path)
        return series

    # No-key fallback: FRED's public CSV endpoint.
    df = pd.read_csv(FRED_CSV_URL.format(series_id=series_id))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df[df.index >= pd.Timestamp(start_date)]
    df.to_csv(cache_path)
    return df[series_id]


def fetch_all_fred_series(target_series: str, predictor_configs: List[dict],
                           start_date: str) -> pd.DataFrame:
    """Fetch the target plus all FRED predictor series and return a merged daily-index frame."""
    series_list = [fetch_fred_series(target_series, start_date).rename(target_series)]
    for cfg in predictor_configs:
        s = fetch_fred_series(cfg["series_id"], start_date).rename(cfg["series_id"])
        series_list.append(s)
        time.sleep(0.2)  # be polite to the no-key CSV endpoint
    return pd.concat(series_list, axis=1)


def fetch_google_trends(keywords: Iterable[str], timeframe: str, geo: str = "US") -> pd.DataFrame:
    """
    Fetch weekly Google Trends search-interest indices for the given keywords.

    Caches to data/raw/google_trends.csv. pytrends occasionally rate-limits; on failure
    this raises so the caller/notebook can retry or fall back to the cached copy.
    """
    cache_path = get_path("data/raw/google_trends.csv")
    if cache_path.exists():
        return pd.read_csv(cache_path, parse_dates=["date"], index_col="date")

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    frames = []
    for kw in keywords:
        pytrends.build_payload([kw], timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        frames.append(df)
        time.sleep(1.0)  # avoid Google Trends rate limiting

    merged = pd.concat(frames, axis=1)
    merged.index.name = "date"
    merged.to_csv(cache_path)
    return merged


def resample_to_monthly(df: pd.DataFrame, resample_rules: dict) -> pd.DataFrame:
    """Resample a mixed-frequency frame to month-end using per-column rules ('mean'/'last')."""
    out = {}
    for col in df.columns:
        rule = resample_rules.get(col, "last")
        resampled = df[col].resample("ME")
        out[col] = resampled.mean() if rule == "mean" else resampled.last()
    return pd.DataFrame(out)
