"""
features.py
-----------
All feature engineering logic for the SuperLender loan default pipeline.

Two categories of features are built here:
1. Behavioural features derived from previous loans history (per customer aggregates)
2. Demographic and performance features derived from the merged table

The key insight is that virtually all customers in this dataset are repeat
customers, meaning prior loan behaviour is available for almost everyone
and is likely to be among the strongest predictors of future default risk.
"""

import pandas as pd
import numpy as np


def engineer_prevloans_features(prevloans: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous loans data into one row per customer, producing
    behavioural features that capture repayment history.

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

    # Parse date columns
    date_cols = ["approveddate", "creationdate", "closeddate", "firstduedate", "firstrepaiddate"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Days between first due date and actual first repayment
    # Positive = paid late, negative = paid early, zero = paid on time
    df["days_late_first_payment"] = (df["firstrepaiddate"] - df["firstduedate"]).dt.days

    # Whether the loan was paid late (binary flag per loan)
    df["paid_late"] = (df["days_late_first_payment"] > 0).astype(int)

    # Loan duration actually taken (closed - approved)
    df["actual_loan_duration"] = (df["closeddate"] - df["approveddate"]).dt.days

    # Interest ratio — how much extra the customer paid relative to principal
    df["interest_ratio"] = (df["totaldue"] - df["loanamount"]) / df["loanamount"]

    # Aggregate per customer
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

    return agg


def engineer_model_features(df: pd.DataFrame, is_train: bool = True) -> tuple:
    """
    Apply feature engineering to the merged DataFrame and return
    features (X) and optionally the target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame from loader.py.
    is_train : bool
        If True, also extracts and encodes the target variable.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or None
        Target variable (None if is_train=False).
    """
    data = df.copy()

    # Parse date columns
    data["approveddate"] = pd.to_datetime(data["approveddate"], errors="coerce")
    data["creationdate"] = pd.to_datetime(data["creationdate"], errors="coerce")
    data["birthdate"] = pd.to_datetime(data["birthdate"], errors="coerce")

    # Age at time of loan application
    data["age_at_application"] = (
        (data["approveddate"] - data["birthdate"]).dt.days / 365.25
    ).round(1)

    # Days between loan creation and approval
    data["days_to_approval"] = (data["approveddate"] - data["creationdate"]).dt.days

    # Interest ratio on the current loan
    data["current_interest_ratio"] = (
        (data["totaldue"] - data["loanamount"]) / data["loanamount"]
    )

    # Whether the customer was referred for this loan
    data["is_referred"] = data["referredby"].notna().astype(int)

    # Encode target variable
    y = None
    if is_train:
        y = (data["good_bad_flag"].str.strip().str.lower() == "good").astype(int)

    # Select and encode categorical features
    categorical_cols = [
        "bank_account_type",
        "employment_status_clients",
        "level_of_education_clients",
        "bank_name_clients",
    ]

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Define final feature columns — drop identifiers, raw dates, and target
    drop_cols = [
        "customerid", "systemloanid", "approveddate", "creationdate",
        "birthdate", "referredby", "bank_branch_clients",
        "longitude_gps", "latitude_gps"
    ]
    if is_train:
        drop_cols.append("good_bad_flag")

    feature_cols = [c for c in data.columns if c not in drop_cols]
    X = data[feature_cols]

    return X, y
