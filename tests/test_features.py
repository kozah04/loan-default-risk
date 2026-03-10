"""
test_features.py
----------------
Tests for src/features.py
"""

import pytest
import pandas as pd
import numpy as np
from src.features import engineer_prevloans_features, engineer_model_features


def make_prevloans_df():
    return pd.DataFrame({
        "customerid": ["A", "A", "B"],
        "systemloanid": [1, 2, 3],
        "loannumber": [1, 2, 1],
        "approveddate": ["2020-01-01", "2020-06-01", "2019-03-01"],
        "creationdate": ["2019-12-28", "2020-05-28", "2019-02-26"],
        "loanamount": [5000.0, 7000.0, 4000.0],
        "totaldue": [5500.0, 7700.0, 4400.0],
        "termdays": [30, 30, 30],
        "closeddate": ["2020-02-01", "2020-07-01", "2019-04-01"],
        "referredby": [None, "C", None],
        "firstduedate": ["2020-02-01", "2020-07-01", "2019-04-01"],
        "firstrepaiddate": ["2020-02-03", "2020-07-01", "2019-04-05"],
    })


class TestEngineerPrevloansFeatures:
    def test_returns_one_row_per_customer(self):
        df = make_prevloans_df()
        result = engineer_prevloans_features(df)
        assert len(result) == 2

    def test_contains_expected_columns(self):
        df = make_prevloans_df()
        result = engineer_prevloans_features(df)
        expected = [
            "customerid", "prev_loan_count", "prev_avg_loanamount",
            "prev_late_payment_rate", "prev_avg_days_late"
        ]
        for col in expected:
            assert col in result.columns

    def test_loan_count_is_correct(self):
        df = make_prevloans_df()
        result = engineer_prevloans_features(df)
        customer_a = result[result["customerid"] == "A"].iloc[0]
        assert customer_a["prev_loan_count"] == 2

    def test_days_late_calculation(self):
        df = make_prevloans_df()
        result = engineer_prevloans_features(df)
        # Customer A: loan 1 was 2 days late, loan 2 was on time (0 days)
        customer_a = result[result["customerid"] == "A"].iloc[0]
        assert customer_a["prev_avg_days_late"] == 1.0

    def test_late_payment_rate_range(self):
        df = make_prevloans_df()
        result = engineer_prevloans_features(df)
        assert result["prev_late_payment_rate"].between(0, 1).all()


class TestEngineerModelFeatures:
    def make_merged_df(self):
        return pd.DataFrame({
            "customerid": ["A"],
            "systemloanid": [10],
            "loannumber": [3],
            "approveddate": ["2021-01-15"],
            "creationdate": ["2021-01-12"],
            "loanamount": [8000.0],
            "totaldue": [8800.0],
            "termdays": [30],
            "referredby": [None],
            "good_bad_flag": ["Good"],
            "birthdate": ["1990-05-20"],
            "bank_account_type": ["Savings"],
            "longitude_gps": [3.3792],
            "latitude_gps": [6.5244],
            "bank_name_clients": ["GTBank"],
            "bank_branch_clients": [None],
            "employment_status_clients": ["Permanent"],
            "level_of_education_clients": [None],
            "prev_loan_count": [2],
            "prev_avg_loanamount": [6000.0],
            "prev_late_payment_rate": [0.5],
            "prev_avg_days_late": [1.0],
            "prev_max_days_late": [2.0],
            "prev_total_late_payments": [1],
            "prev_max_loanamount": [7000.0],
            "prev_avg_totaldue": [6600.0],
            "prev_avg_termdays": [30.0],
            "prev_avg_interest_ratio": [0.1],
            "prev_avg_loan_duration": [31.0],
            "prev_was_referred": [0],
        })

    def test_returns_tuple_of_x_and_y(self):
        df = self.make_merged_df()
        result = engineer_model_features(df, is_train=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_target_is_binary(self):
        df = self.make_merged_df()
        _, y = engineer_model_features(df, is_train=True)
        assert set(y.unique()).issubset({0, 1})

    def test_good_flag_encodes_to_one(self):
        df = self.make_merged_df()
        _, y = engineer_model_features(df, is_train=True)
        assert y.iloc[0] == 1

    def test_customerid_not_in_features(self):
        df = self.make_merged_df()
        X, _ = engineer_model_features(df, is_train=True)
        assert "customerid" not in X.columns

    def test_no_target_returned_when_not_train(self):
        df = self.make_merged_df().drop(columns=["good_bad_flag"])
        X, y = engineer_model_features(df, is_train=False)
        assert y is None
