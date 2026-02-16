"""Tests for ColumnAnalyzer utility."""

import pytest
import pandas as pd
import numpy as np
from treelab.utils.column_analyzer import ColumnAnalyzer


class TestColumnAnalyzer:
    """Tests for ColumnAnalyzer class."""

    def test_get_numeric_columns(self, sample_dataframe):
        """Test getting numeric columns."""
        numeric_cols = ColumnAnalyzer.get_numeric_columns(sample_dataframe)

        assert "age" in numeric_cols
        assert "salary" in numeric_cols
        assert "department" not in numeric_cols

    def test_get_categorical_columns(self, sample_dataframe):
        """Test getting categorical columns."""
        cat_cols = ColumnAnalyzer.get_categorical_columns(sample_dataframe)

        assert "department" in cat_cols
        assert "age" not in cat_cols

    def test_get_columns_with_missing(self, dataframe_with_missing):
        """Test getting columns with missing values."""
        cols = ColumnAnalyzer.get_columns_with_missing(dataframe_with_missing)

        assert "a" in cols
        assert "c" in cols
        assert "d" in cols

    def test_get_low_cardinality_categorical(self, sample_dataframe):
        """Test getting low cardinality categorical columns."""
        low_card = ColumnAnalyzer.get_low_cardinality_categorical(
            sample_dataframe, max_unique=5
        )

        assert "department" in low_card

    def test_get_column_info(self, sample_dataframe):
        """Test getting column information."""
        info = ColumnAnalyzer.get_column_info(sample_dataframe, "age")

        assert info["name"] == "age"
        assert info["dtype"] == "int64"
        assert info["is_numeric"] is True
        assert "min" in info
        assert "max" in info
        assert "mean" in info

    def test_get_column_info_categorical(self, sample_dataframe):
        """Test getting column info for categorical."""
        info = ColumnAnalyzer.get_column_info(sample_dataframe, "department")

        assert info["is_numeric"] is False
        assert "top_values" in info

    def test_suggest_scaling_columns(self, sample_dataframe):
        """Test suggesting columns for scaling."""
        cols = ColumnAnalyzer.suggest_scaling_columns(sample_dataframe)

        assert "age" in cols
        assert "salary" in cols

    def test_suggest_encoding_columns(self, sample_dataframe):
        """Test suggesting columns for encoding."""
        cols = ColumnAnalyzer.suggest_encoding_columns(sample_dataframe)

        assert "department" in cols

    def test_suggest_imputation_columns_numeric(self, dataframe_with_missing):
        """Test suggesting columns for imputation (numeric only)."""
        cols = ColumnAnalyzer.suggest_imputation_columns(
            dataframe_with_missing, numeric_only=True
        )

        assert "a" in cols
        assert "d" in cols
        assert "c" not in cols

    def test_get_all_columns_info(self, sample_dataframe):
        """Test getting info for all columns."""
        all_info = ColumnAnalyzer.get_all_columns_info(sample_dataframe)

        assert len(all_info) == 4
        assert all(isinstance(info, dict) for info in all_info)
