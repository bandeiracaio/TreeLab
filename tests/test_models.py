"""Tests for model actions."""

import pytest
import pandas as pd
import numpy as np
from treelab.actions.modeling.tree_models import (
    DecisionTreeClassifierAction,
    RandomForestClassifierAction,
    DecisionTreeRegressorAction,
    RandomForestRegressorAction,
    GradientBoostingRegressorAction,
)
from treelab.core.action_registry import ActionRegistry


@pytest.fixture
def numeric_dataframe():
    """Create a numeric-only DataFrame for model testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "score": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def numeric_train_test_split(numeric_dataframe):
    """Create train/test split DataFrames for model testing."""
    train_df = numeric_dataframe.iloc[:6].copy()
    test_df = numeric_dataframe.iloc[6:].copy()
    return train_df, test_df


@pytest.fixture(autouse=True)
def setup_actions():
    """Register actions before tests."""
    from treelab.actions.transformations import TrainTestSplitAction
    from treelab.actions.modeling import (
        DecisionTreeClassifierAction,
        RandomForestClassifierAction,
        DecisionTreeRegressorAction,
        RandomForestRegressorAction,
        GradientBoostingRegressorAction,
    )

    ActionRegistry.clear()
    ActionRegistry.register_transformation(TrainTestSplitAction)
    ActionRegistry.register_modeling(DecisionTreeClassifierAction)
    ActionRegistry.register_modeling(RandomForestClassifierAction)
    ActionRegistry.register_modeling(DecisionTreeRegressorAction)
    ActionRegistry.register_modeling(RandomForestRegressorAction)
    ActionRegistry.register_modeling(GradientBoostingRegressorAction)
    yield
    ActionRegistry.clear()


class TestDecisionTreeClassifierAction:
    def test_get_parameters(self):
        action = DecisionTreeClassifierAction()
        params = action.get_parameters()
        assert len(params) > 0

    def test_validate_no_split(self, numeric_dataframe):
        action = DecisionTreeClassifierAction()
        is_valid, msg = action.validate(numeric_dataframe, {})
        assert is_valid is False

    def test_validate_invalid_max_depth(
        self, numeric_dataframe, numeric_train_test_split
    ):
        action = DecisionTreeClassifierAction()
        train_df, test_df = numeric_train_test_split
        is_valid, msg = action.validate(
            numeric_dataframe, {"max_depth": -1}, train_df, test_df
        )
        assert is_valid is False

    def test_execute(self, numeric_dataframe, numeric_train_test_split):
        action = DecisionTreeClassifierAction()
        train_df, test_df = numeric_train_test_split
        result = action.execute(
            numeric_dataframe, {"_target_column": "score"}, train_df, test_df
        )
        assert result["model"] is not None
        assert "train_accuracy" in result["metadata"]
        assert "test_accuracy" in result["metadata"]

    def test_to_python_code(self):
        action = DecisionTreeClassifierAction()
        code = action.to_python_code({"max_depth": 5})
        assert "DecisionTreeClassifier" in code
        assert "max_depth=5" in code


class TestRandomForestClassifierAction:
    def test_get_parameters(self):
        action = RandomForestClassifierAction()
        params = action.get_parameters()
        assert len(params) > 0
        assert any(p.name == "n_estimators" for p in params)

    def test_execute(self, numeric_dataframe, numeric_train_test_split):
        action = RandomForestClassifierAction()
        train_df, test_df = numeric_train_test_split
        result = action.execute(
            numeric_dataframe, {"_target_column": "score"}, train_df, test_df
        )
        assert result["model"] is not None
        assert "test_accuracy" in result["metadata"]


class TestDecisionTreeRegressorAction:
    def test_execute(self, numeric_dataframe, numeric_train_test_split):
        action = DecisionTreeRegressorAction()
        train_df, test_df = numeric_train_test_split
        result = action.execute(
            numeric_dataframe, {"_target_column": "salary"}, train_df, test_df
        )
        assert result["model"] is not None
        assert "train_r2" in result["metadata"]
        assert "test_r2" in result["metadata"]


class TestRandomForestRegressorAction:
    def test_execute(self, numeric_dataframe, numeric_train_test_split):
        action = RandomForestRegressorAction()
        train_df, test_df = numeric_train_test_split
        result = action.execute(
            numeric_dataframe, {"_target_column": "salary"}, train_df, test_df
        )
        assert result["model"] is not None
        assert "test_r2" in result["metadata"]
        assert "test_mae" in result["metadata"]
        assert "test_rmse" in result["metadata"]


class TestGradientBoostingRegressorAction:
    def test_get_parameters(self):
        action = GradientBoostingRegressorAction()
        params = action.get_parameters()
        assert len(params) > 0
        assert any(p.name == "n_estimators" for p in params)

    def test_execute(self, numeric_dataframe, numeric_train_test_split):
        action = GradientBoostingRegressorAction()
        train_df, test_df = numeric_train_test_split
        result = action.execute(
            numeric_dataframe, {"_target_column": "salary"}, train_df, test_df
        )
        assert result["model"] is not None
        assert "test_r2" in result["metadata"]

    def test_to_python_code(self):
        action = GradientBoostingRegressorAction()
        code = action.to_python_code({"n_estimators": 100})
        assert "GradientBoostingRegressor" in code
