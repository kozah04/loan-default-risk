"""
features.py
-----------
All feature engineering logic for the SuperLender loan default pipeline.

Two categories of features are built here:
1. Behavioural features derived from previous loans history (per customer aggregates)
2. Demographic and performance features derived from the merged table

Feature engineering history:
- Baseline: lifetime aggregations of prior loan behaviour
- Iteration 2: added recency features (recent_1_paid_late, recent_2_late_rate,
  recent_3 variants, avg_days_late variants, lateness_trend)
- Iteration 3: pruned redundant lateness features. Kept recent_1_paid_late,
  recent_2_late_rate, prev_late_payment_rate, prev_max_days_late
- Iteration 9: added sequential loan history features
- Iteration 10: added engineer_pltr_features() for tree-derived interaction rules capturing streaks and
  recovery patterns that flat averages cannot represent
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _consecutive_ontime_streak(paid_late_series: pd.Series) -> int:
    """
    Count consecutive on-time payments at the END of the series.
    Walks backwards from the most recent loan and stops at the first
    late payment. Returns 0 if the most recent loan was late.
    """
    vals = list(paid_late_series)
    streak = 0
    for v in reversed(vals):
        if v == 0:
            streak += 1
        else:
            break
    return streak


def _consecutive_late_streak(paid_late_series: pd.Series) -> int:
    """
    Count consecutive late payments at the END of the series.
    Walks backwards from the most recent loan and stops at the first
    on-time payment. Returns 0 if the most recent loan was on time.
    """
    vals = list(paid_late_series)
    streak = 0
    for v in reversed(vals):
        if v == 1:
            streak += 1
        else:
            break
    return streak


def _ever_recovered(paid_late_series: pd.Series) -> int:
    """
    Returns 1 if the customer ever had a late payment followed by
    an on-time payment. This distinguishes chronic late payers from
    customers who had a rough patch but recovered.
    Returns 0 if the customer never had a late payment, or never
    recovered after one.
    """
    vals = list(paid_late_series)
    for i in range(len(vals) - 1):
        if vals[i] == 1 and vals[i + 1] == 0:
            return 1
    return 0


def _last3_trend(paid_late_series: pd.Series) -> int:
    """
    Categorical signal based on the last 3 loans:
     1 = all on time (improving / stable good)
    -1 = all late (deteriorating / stable bad)
     0 = mixed (no clear trend)
    Returns 0 if fewer than 3 prior loans exist.
    """
    vals = list(paid_late_series.tail(3))
    if len(vals) < 3:
        return 0
    if all(v == 0 for v in vals):
        return 1
    if all(v == 1 for v in vals):
        return -1
    return 0


def engineer_prevloans_features(prevloans: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous loans data into one row per customer, producing
    behavioural features that capture repayment history.

    Iteration 9 adds sequential features that capture streak patterns
    and recovery behaviour that flat averages cannot represent:
    - consecutive_ontime_streak: how many recent loans in a row were on time
    - consecutive_late_streak: how many recent loans in a row were late
    - ever_recovered: did the customer ever recover after a late payment
    - last3_trend: are the last 3 loans all good, all bad, or mixed

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

    # Sort chronologically per customer so tail() gives the most recent loans
    df = df.sort_values(["customerid", "approveddate"]).reset_index(drop=True)

    # --- Lifetime aggregations ---
    agg = df.groupby("customerid").agg(
        prev_loan_count=("systemloanid", "count"),
        prev_avg_loanamount=("loanamount", "mean"),
        prev_max_loanamount=("loanamount", "max"),
        prev_avg_totaldue=("totaldue", "mean"),
        prev_avg_termdays=("termdays", "mean"),
        prev_max_days_late=("days_late_first_payment", "max"),
        prev_late_payment_rate=("paid_late", "mean"),
        prev_total_late_payments=("paid_late", "sum"),
        prev_avg_interest_ratio=("interest_ratio", "mean"),
        prev_avg_loan_duration=("actual_loan_duration", "mean"),
        prev_was_referred=("referredby", lambda x: x.notna().any().astype(int)),
    ).reset_index()

    # --- Recency: most recent loan outcome ---
    most_recent = (
        df.groupby("customerid")
        .last()
        .reset_index()[["customerid", "paid_late"]]
        .rename(columns={"paid_late": "recent_1_paid_late"})
    )
    agg = agg.merge(most_recent, on="customerid", how="left")

    # --- Recency: late rate over last 2 loans ---
    recent_2 = (
        df.groupby("customerid")
        .apply(lambda g: g.tail(2)["paid_late"].mean())
        .reset_index()
    )
    recent_2.columns = ["customerid", "recent_2_late_rate"]
    agg = agg.merge(recent_2, on="customerid", how="left")

    # --- Iteration 9: Sequential features ---
    seq = df.groupby("customerid")["paid_late"].apply(
        lambda s: pd.Series({
            "consecutive_ontime_streak": _consecutive_ontime_streak(s),
            "consecutive_late_streak":   _consecutive_late_streak(s),
            "ever_recovered":            _ever_recovered(s),
            "last3_trend":               _last3_trend(s),
        })
    ).reset_index()

    # Pivot from long to wide
    seq = seq.pivot(index="customerid", columns="level_1", values="paid_late").reset_index()
    seq.columns.name = None

    agg = agg.merge(seq, on="customerid", how="left")

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
    data["creationdate"]  = pd.to_datetime(data["creationdate"],  errors="coerce")
    data["birthdate"]     = pd.to_datetime(data["birthdate"],     errors="coerce")

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

    if train_columns is not None:
        X = X.reindex(columns=train_columns, fill_value=0)

    return X, y


def engineer_pltr_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_apply: pd.DataFrame,
    max_depth: int = 2,
    min_samples_leaf: int = 30,
) -> pd.DataFrame:
    """
    Penalised Logistic Tree Regression (PLTR) interaction features.

    Fits a shallow decision tree on training data to discover high-value
    feature interactions (splits), then encodes each leaf condition as a
    binary feature. These rule features are added to the feature matrix
    before passing to Logistic Regression.

    For example, a depth-2 tree might discover:
    "prev_late_payment_rate > 0.5 AND loanamount > 20000" as a leaf
    condition that is especially predictive of default. This interaction
    cannot be captured by logistic regression alone.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features — used to fit the tree.
    y_train : pd.Series
        Training labels — used to fit the tree.
    X_apply : pd.DataFrame
        Features to apply the learned rules to (train or test).
    max_depth : int
        Maximum tree depth. Depth 2 produces at most 4 leaf conditions.
        Keep shallow to avoid overfitting on small data.
    min_samples_leaf : int
        Minimum samples per leaf. Higher values prevent noisy splits.

    Returns
    -------
    pd.DataFrame
        X_apply with additional binary rule columns appended.
    """
    # Fill NaN before fitting the tree — DecisionTree cannot handle NaN
    X_train_filled = X_train.fillna(X_train.median(numeric_only=True))
    X_apply_filled = X_apply.fillna(X_train.median(numeric_only=True))

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_train_filled, y_train)

    # Apply tree to get leaf node assignments
    train_leaves = tree.apply(X_train_filled)
    apply_leaves = tree.apply(X_apply_filled)

    # Encode each unique leaf as a binary indicator column
    unique_leaves = np.unique(train_leaves)
    rule_cols = {}
    for leaf_id in unique_leaves:
        col_name = f"rule_leaf_{leaf_id}"
        rule_cols[col_name] = (apply_leaves == leaf_id).astype(int)

    rule_df = pd.DataFrame(rule_cols, index=X_apply.index)

    return pd.concat([X_apply.reset_index(drop=True), rule_df.reset_index(drop=True)], axis=1)