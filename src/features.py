"""
features.py
-----------
All feature engineering logic for the SuperLender loan default pipeline.

Two categories of features are built here:
1. Behavioural features derived from previous loans history (per customer aggregates)
2. Demographic and performance features derived from the merged table

Iteration 2 additions:
- Recency-based features: lateness metrics computed on the most recent 1, 2,
  and 3 prior loans rather than the full history
- Most recent loan outcome: binary flag and days late for the single most
  recent prior loan, capturing the customer's latest repayment behaviour
- Lateness trend: slope of days late across prior loans ordered by date,
  capturing whether a customer is improving or deteriorating over time
"""

import pandas as pd
import numpy as np


def _compute_lateness_trend(days_late_series: pd.Series) -> float:
    """
    Compute the slope of days late across prior loans ordered by date.
    A positive slope means the customer is getting progressively later.
    A negative slope means they are improving.
    Returns 0 if fewer than 2 loans exist.
    """
    vals = days_late_series.values
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals))
    slope = np.polyfit(x, vals, 1)[0]
    return slope


def engineer_prevloans_features(prevloans: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous loans data into one row per customer, producing
    behavioural features that capture repayment history.

    Iteration 2 adds recency-based features, most recent loan outcome,
    and a lateness trend feature to the baseline aggregations.

    Parameters
    ----------
    prevloans : pd.DataFrame
        Raw previous loans table.

    Returns
    -------
    pd.DataFrame
        One row per customerid with aggregated behavioural features.
    """
    df = prevloans.copy()

    date_cols = ["approveddate", "creationdate", "closeddate", "firstduedate", "firstrepaiddate"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["days_late_first_payment"] = (df["firstrepaiddate"] - df["firstduedate"]).dt.days
    df["paid_late"] = (df["days_late_first_payment"] > 0).astype(int)
    df["actual_loan_duration"] = (df["closeddate"] - df["approveddate"]).dt.days
    df["interest_ratio"] = (df["totaldue"] - df["loanamount"]) / df["loanamount"]

    # Sort by customer and approval date so recency-based features are computed
    # on the correct temporal order
    df = df.sort_values(["customerid", "approveddate"]).reset_index(drop=True)

    # --- Baseline aggregations (unchanged from iteration 1) ---
    agg = df.groupby("customerid").agg(
        prev_loan_count=("systemloanid", "count"),
        prev_avg_loanamount=("loanamount", "mean"),
        prev_max_loanamount=("loanamount", "max"),
        prev_avg_totaldue=("totaldue", "mean"),
        prev_avg_termdays=("termdays", "mean"),
        prev_avg_days_late=("days_late_first_payment", "mean"),
        prev_max_days_late=("days_late_first_payment", "max"),
        prev_late_payment_rate=("paid_late", "mean"),
        prev_total_late_payments=("paid_late", "sum"),
        prev_avg_interest_ratio=("interest_ratio", "mean"),
        prev_avg_loan_duration=("actual_loan_duration", "mean"),
        prev_was_referred=("referredby", lambda x: x.notna().any().astype(int)),
    ).reset_index()

    # --- Iteration 2: Most recent loan outcome ---
    # The single most recent prior loan is particularly informative.
    # Someone who just defaulted is a higher risk than their lifetime average suggests.
    most_recent = df.groupby("customerid").last().reset_index()
    most_recent = most_recent[["customerid", "days_late_first_payment", "paid_late"]].rename(columns={
        "days_late_first_payment": "recent_1_days_late",
        "paid_late": "recent_1_paid_late",
    })
    agg = agg.merge(most_recent, on="customerid", how="left")

    # --- Iteration 2: Recency-based late payment rate (last 3 loans) ---
    # Customers with only 1 or 2 prior loans will get NaN for the 3-loan window
    # which the imputer handles. This is intentional — a short history should
    # not be filled with the overall average.
    def recent_late_rate(group, n):
        recent = group.tail(n)
        return recent["paid_late"].mean()

    def recent_avg_days_late(group, n):
        recent = group.tail(n)
        return recent["days_late_first_payment"].mean()

    recency_features = df.groupby("customerid").apply(
        lambda g: pd.Series({
            "recent_3_late_rate": recent_late_rate(g, 3),
            "recent_3_avg_days_late": recent_avg_days_late(g, 3),
            "recent_2_late_rate": recent_late_rate(g, 2),
            "recent_2_avg_days_late": recent_avg_days_late(g, 2),
        })
    ).reset_index()

    agg = agg.merge(recency_features, on="customerid", how="left")

    # --- Iteration 2: Lateness trend ---
    # Slope of days late across prior loans ordered by date.
    # Positive = getting worse over time. Negative = improving.
    trend = df.groupby("customerid")["days_late_first_payment"].apply(
        _compute_lateness_trend
    ).reset_index()
    trend.columns = ["customerid", "lateness_trend"]

    agg = agg.merge(trend, on="customerid", how="left")

    return agg


def engineer_model_features(
    df: pd.DataFrame,
    is_train: bool = True,
    train_columns: list = None
) -> tuple:
    """
    Apply feature engineering to the merged DataFrame and return
    features (X) and optionally the target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from loader.py.
    is_train : bool
        If True, extracts and encodes the target variable.
    train_columns : list, optional
        List of feature columns from training. When provided during test
        processing, the output is reindexed to match these columns exactly.
        Columns missing from the test set are filled with 0, preventing
        one-hot encoding mismatches caused by categories present in training
        but absent in the test set.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or None
        Target variable (None if is_train=False).
    """
    data = df.copy()

    data["approveddate"] = pd.to_datetime(data["approveddate"], errors="coerce")
    data["creationdate"] = pd.to_datetime(data["creationdate"], errors="coerce")
    data["birthdate"] = pd.to_datetime(data["birthdate"], errors="coerce")

    data["age_at_application"] = (
        (data["approveddate"] - data["birthdate"]).dt.days / 365.25
    ).round(1)

    data["days_to_approval"] = (data["approveddate"] - data["creationdate"]).dt.days

    data["current_interest_ratio"] = (
        (data["totaldue"] - data["loanamount"]) / data["loanamount"]
    )

    data["is_referred"] = data["referredby"].notna().astype(int)

    y = None
    if is_train:
        y = (data["good_bad_flag"].str.strip().str.lower() == "good").astype(int)

    categorical_cols = [
        "bank_account_type",
        "employment_status_clients",
        "level_of_education_clients",
        "bank_name_clients",
    ]

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    drop_cols = [
        "customerid", "systemloanid", "approveddate", "creationdate",
        "birthdate", "referredby", "bank_branch_clients",
        "longitude_gps", "latitude_gps"
    ]
    if is_train:
        drop_cols.append("good_bad_flag")

    feature_cols = [c for c in data.columns if c not in drop_cols]
    X = data[feature_cols]

    # Align test features to match training columns exactly.
    # Categories present in training but absent in test produce missing columns
    # after get_dummies. We add them back as zeros so the pipeline does not
    # throw a feature mismatch error at prediction time.
    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)

    return X, y