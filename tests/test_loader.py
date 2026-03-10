"""
test_loader.py
--------------
Tests for src/loader.py

Note: These tests require the raw data files to be present in data/raw/.
The data is gitignored due to Zindi licensing restrictions.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.loader import load_raw_tables, merge_tables


class TestLoadRawTables:
    def test_invalid_split_raises_value_error(self):
        with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
            load_raw_tables(split="validation")

    def test_returns_three_dataframes(self, tmp_path, monkeypatch):
        # Create minimal mock CSVs
        perf = pd.DataFrame({"customerid": ["A"], "loanamount": [1000]})
        demo = pd.DataFrame({"customerid": ["A"], "birthdate": ["1990-01-01"]})
        prev = pd.DataFrame({"customerid": ["A"], "systemloanid": [1]})

        perf.to_csv(tmp_path / "trainperf.csv", index=False)
        demo.to_csv(tmp_path / "traindemographics.csv", index=False)
        prev.to_csv(tmp_path / "trainprevloans.csv", index=False)

        import src.loader as loader_module
        monkeypatch.setattr(loader_module, "RAW_DIR", tmp_path)

        result = load_raw_tables("train")
        assert len(result) == 3
        assert all(isinstance(df, pd.DataFrame) for df in result)


class TestMergeTables:
    def test_merge_produces_single_dataframe(self):
        perf = pd.DataFrame({"customerid": ["A", "B"], "loanamount": [1000, 2000]})
        demo = pd.DataFrame({"customerid": ["A", "B"], "birthdate": ["1990-01-01", "1985-06-15"]})
        prev_features = pd.DataFrame({"customerid": ["A", "B"], "prev_loan_count": [3, 5]})

        result = merge_tables(perf, demo, prev_features)
        assert isinstance(result, pd.DataFrame)

    def test_merge_retains_all_perf_rows(self):
        perf = pd.DataFrame({"customerid": ["A", "B", "C"], "loanamount": [1000, 2000, 3000]})
        demo = pd.DataFrame({"customerid": ["A", "B"], "birthdate": ["1990-01-01", "1985-06-15"]})
        prev_features = pd.DataFrame({"customerid": ["A"], "prev_loan_count": [3]})

        result = merge_tables(perf, demo, prev_features)
        assert len(result) == 3

    def test_merge_adds_demographic_columns(self):
        perf = pd.DataFrame({"customerid": ["A"], "loanamount": [1000]})
        demo = pd.DataFrame({"customerid": ["A"], "birthdate": ["1990-01-01"]})
        prev_features = pd.DataFrame({"customerid": ["A"], "prev_loan_count": [3]})

        result = merge_tables(perf, demo, prev_features)
        assert "birthdate" in result.columns

    def test_merge_adds_prevloans_columns(self):
        perf = pd.DataFrame({"customerid": ["A"], "loanamount": [1000]})
        demo = pd.DataFrame({"customerid": ["A"], "birthdate": ["1990-01-01"]})
        prev_features = pd.DataFrame({"customerid": ["A"], "prev_loan_count": [3]})

        result = merge_tables(perf, demo, prev_features)
        assert "prev_loan_count" in result.columns
