"""Tests for transformation actions."""

import pytest
import pandas as pd
import numpy as np
from treelab.actions.transformations import (
    DropColumnsAction,
    SimpleImputerAction,
    TrainTestSplitAction,
)


class TestDropColumnsAction:
    """Tests for DropColumnsAction."""

    def test_get_parameters(self):
        """Test getting parameters."""
        action = DropColumnsAction()
        params = action.get_parameters()

        assert len(params) > 0
        assert any(p.name == "columns" for p in params)

    def test_validate_no_columns(self, sample_dataframe):
        """Test validation fails with no columns selected."""
        action = DropColumnsAction()

        is_valid, msg = action.validate(sample_dataframe, {"columns": []})

        assert is_valid is False

    def test_validate_invalid_column(self, sample_dataframe):
        """Test validation fails with non-existent column."""
        action = DropColumnsAction()

        is_valid, msg = action.validate(sample_dataframe, {"columns": ["nonexistent"]})

        assert is_valid is False

    def test_execute(self, sample_dataframe):
        """Test executing drop columns action."""
        action = DropColumnsAction()

        result = action.execute(sample_dataframe, {"columns": ["department"]})

        assert result["df"] is not None
        assert "department" not in result["df"].columns
        assert len(result["df"].columns) == len(sample_dataframe.columns) - 1

    def test_suggest_columns(self, sample_dataframe):
        """Test suggesting columns."""
        action = DropColumnsAction()

        cols = action.suggest_columns(sample_dataframe)

        assert isinstance(cols, list)


class TestSimpleImputerAction:
    """Tests for SimpleImputerAction."""

    def test_get_parameters(self):
        """Test getting parameters."""
        action = SimpleImputerAction()
        params = action.get_parameters()

        assert len(params) > 0

    def test_validate_no_columns(self, dataframe_with_missing):
        """Test validation fails with no columns."""
        action = SimpleImputerAction()

        is_valid, msg = action.validate(dataframe_with_missing, {"columns": []})

        assert is_valid is False

    def test_validate_non_numeric_for_mean(self, sample_dataframe):
        """Test validation fails for non-numeric with mean strategy."""
        action = SimpleImputerAction()

        is_valid, msg = action.validate(
            sample_dataframe, {"columns": ["department"], "strategy": "mean"}
        )

        assert is_valid is False

    def test_execute_mean(self, dataframe_with_missing):
        """Test executing imputation with mean strategy."""
        action = SimpleImputerAction()

        result = action.execute(
            dataframe_with_missing, {"columns": ["a"], "strategy": "mean"}
        )

        assert result["df"]["a"].isnull().sum() == 0

    def test_execute_constant(self, dataframe_with_missing):
        """Test executing imputation with constant strategy."""
        action = SimpleImputerAction()

        result = action.execute(
            dataframe_with_missing,
            {"columns": ["a"], "strategy": "constant", "fill_value": 999},
        )

        assert result["df"]["a"].isnull().sum() == 0
        assert (result["df"]["a"] == 999).sum() > 0

    def test_suggest_columns(self, dataframe_with_missing):
        """Test suggesting columns for imputation."""
        action = SimpleImputerAction()

        cols = action.suggest_columns(dataframe_with_missing)

        assert "a" in cols
        assert "d" in cols


class TestTrainTestSplitAction:
    """Tests for TrainTestSplitAction."""

    def test_get_parameters(self):
        """Test getting parameters."""
        action = TrainTestSplitAction()
        params = action.get_parameters()

        assert len(params) > 0
        assert any(p.name == "target_column" for p in params)

    def test_validate_no_target(self, sample_dataframe):
        """Test validation fails without target column."""
        action = TrainTestSplitAction()

        is_valid, msg = action.validate(sample_dataframe, {"target_column": ""})

        assert is_valid is False

    def test_validate_invalid_target(self, sample_dataframe):
        """Test validation fails with non-existent target."""
        action = TrainTestSplitAction()

        is_valid, msg = action.validate(
            sample_dataframe, {"target_column": "nonexistent"}
        )

        assert is_valid is False

    def test_validate_test_size_out_of_range(self, sample_dataframe):
        """Test validation fails with invalid test_size."""
        action = TrainTestSplitAction()

        is_valid, msg = action.validate(
            sample_dataframe, {"target_column": "age", "test_size": 1.5}
        )

        assert is_valid is False

    def test_execute(self, sample_dataframe):
        """Test executing train/test split."""
        action = TrainTestSplitAction()

        result = action.execute(
            sample_dataframe,
            {"target_column": "target", "test_size": 0.3, "random_state": 42},
        )

        assert result["train_df"] is not None
        assert result["test_df"] is not None
        assert len(result["train_df"]) + len(result["test_df"]) == len(sample_dataframe)
        assert "target" in result["train_df"].columns

    def test_suggest_columns(self, sample_dataframe):
        """Test suggesting target columns."""
        action = TrainTestSplitAction()

        cols = action.suggest_columns(sample_dataframe)

        assert "target" in cols
