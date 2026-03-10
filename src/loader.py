"""
loader.py
---------
Handles loading and merging of the three SuperLender dataset tables:
- performance data (trainperf / testperf)
- demographic data (traindemographics / testdemographics)
- previous loans data (trainprevloans / testprevloans)

All merging logic lives here so that notebooks stay clean and the
pipeline can be reproduced with a single function call.
"""

import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_raw_tables(split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw tables for a given split.

    Parameters
    ----------
    split : str
        Either 'train' or 'test'.

    Returns
    -------
    tuple of (perf, demographics, prevloans) DataFrames
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    perf = pd.read_csv(RAW_DIR / f"{split}perf.csv")
    demographics = pd.read_csv(RAW_DIR / f"{split}demographics.csv")
    prevloans = pd.read_csv(RAW_DIR / f"{split}prevloans.csv")

    return perf, demographics, prevloans


def merge_tables(
    perf: pd.DataFrame,
    demographics: pd.DataFrame,
    prevloans_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge performance, demographic, and engineered previous loans features
    into a single modelling-ready DataFrame.

    Previous loans features are passed in already aggregated per customer
    (aggregation happens in features.py, not here).

    Parameters
    ----------
    perf : pd.DataFrame
        Performance table.
    demographics : pd.DataFrame
        Demographics table.
    prevloans_features : pd.DataFrame
        Aggregated previous loans features, one row per customerid.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame ready for modelling.
    """
    df = perf.merge(demographics, on="customerid", how="left")
    df = df.merge(prevloans_features, on="customerid", how="left")
    return df


def load_and_merge(split: str = "train") -> pd.DataFrame:
    """
    Convenience function that loads all raw tables and returns the merged
    DataFrame. Feature engineering on prevloans is handled separately in
    features.py — this function handles structural merging only.

    Parameters
    ----------
    split : str
        Either 'train' or 'test'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    from src.features import engineer_prevloans_features

    perf, demographics, prevloans = load_raw_tables(split)
    prevloans_features = engineer_prevloans_features(prevloans)
    return merge_tables(perf, demographics, prevloans_features)
