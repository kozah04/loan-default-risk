"""
test_model.py
-------------
Tests for src/model.py

Full model training tests are skipped by default to avoid long runtimes.
Enable with: RUN_MODEL_TESTS=1 python -m pytest tests/test_model.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model import build_pipeline, get_models, get_param_grids, predict_proba
from sklearn.linear_model import LogisticRegression


RUN_MODEL_TESTS = os.environ.get("RUN_MODEL_TESTS", "0") == "1"


class TestBuildPipeline:
    def test_returns_pipeline(self):
        model = LogisticRegression()
        pipeline = build_pipeline(model)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_imputer(self):
        model = LogisticRegression()
        pipeline = build_pipeline(model)
        assert "imputer" in pipeline.named_steps

    def test_pipeline_has_scaler_when_requested(self):
        model = LogisticRegression()
        pipeline = build_pipeline(model, scale=True)
        assert "scaler" in pipeline.named_steps

    def test_pipeline_has_no_scaler_by_default(self):
        model = LogisticRegression()
        pipeline = build_pipeline(model)
        assert "scaler" not in pipeline.named_steps


class TestGetModels:
    def test_returns_three_models(self):
        models = get_models()
        assert len(models) == 3

    def test_expected_model_names(self):
        models = get_models()
        expected = {"logistic_regression", "random_forest", "xgboost"}
        assert set(models.keys()) == expected

    def test_all_models_are_pipelines(self):
        models = get_models()
        for name, model in models.items():
            assert isinstance(model, Pipeline), f"{name} is not a Pipeline"


class TestGetParamGrids:
    def test_returns_three_grids(self):
        grids = get_param_grids()
        assert len(grids) == 3

    def test_param_keys_have_model_prefix(self):
        grids = get_param_grids()
        for model_name, grid in grids.items():
            for key in grid.keys():
                assert key.startswith("model__"), (
                    f"Param '{key}' in {model_name} grid is missing 'model__' prefix"
                )


class TestPredictProba:
    def make_data(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    @pytest.mark.skipif(not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to enable")
    def test_predict_proba_returns_array(self):
        X, y = self.make_data()
        model = build_pipeline(LogisticRegression(max_iter=1000), scale=True)
        model.fit(X, y)
        proba = predict_proba(model, X)
        assert isinstance(proba, np.ndarray)
        assert len(proba) == len(X)

    @pytest.mark.skipif(not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to enable")
    def test_probabilities_between_zero_and_one(self):
        X, y = self.make_data()
        model = build_pipeline(LogisticRegression(max_iter=1000), scale=True)
        model.fit(X, y)
        proba = predict_proba(model, X)
        assert (proba >= 0).all() and (proba <= 1).all()
