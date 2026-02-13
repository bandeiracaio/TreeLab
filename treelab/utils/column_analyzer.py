"""Utilities for analyzing DataFrame columns and making smart suggestions."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class ColumnAnalyzer:
    """Analyzes DataFrame columns to provide smart suggestions for transformations."""

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Get all numeric columns."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> List[str]:
        """Get all categorical/object columns."""
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    @staticmethod
    def get_columns_with_missing(df: pd.DataFrame) -> List[str]:
        """Get columns that have missing values."""
        return df.columns[df.isnull().any()].tolist()

    @staticmethod
    def get_low_cardinality_categorical(
        df: pd.DataFrame, max_unique: int = 20
    ) -> List[str]:
        """
        Get categorical columns with low cardinality (good for one-hot encoding).

        Args:
            df: DataFrame to analyze
            max_unique: Maximum number of unique values

        Returns:
            List of column names
        """
        cat_cols = ColumnAnalyzer.get_categorical_columns(df)
        return [col for col in cat_cols if df[col].nunique() <= max_unique]

    @staticmethod
    def get_column_info(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Get detailed information about a column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Dictionary with column information
        """
        col_data = df[column]

        info = {
            "name": column,
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_pct": float(col_data.isnull().sum() / len(df) * 100),
            "unique_count": int(col_data.nunique()),
            "unique_pct": float(col_data.nunique() / len(df) * 100),
        }

        # Add type-specific info
        if pd.api.types.is_numeric_dtype(col_data):
            info["is_numeric"] = True
            info["min"] = float(col_data.min()) if not col_data.isna().all() else None
            info["max"] = float(col_data.max()) if not col_data.isna().all() else None
            info["mean"] = float(col_data.mean()) if not col_data.isna().all() else None
            info["std"] = float(col_data.std()) if not col_data.isna().all() else None
        else:
            info["is_numeric"] = False
            info["top_values"] = (
                col_data.value_counts().head(5).to_dict() if len(col_data) > 0 else {}
            )

        # Sample values
        sample_values = col_data.dropna().head(3).tolist()
        info["sample_values"] = [str(v) for v in sample_values]

        return info

    @staticmethod
    def suggest_scaling_columns(df: pd.DataFrame) -> List[str]:
        """Suggest columns that should be scaled (numeric columns)."""
        return ColumnAnalyzer.get_numeric_columns(df)

    @staticmethod
    def suggest_encoding_columns(df: pd.DataFrame) -> List[str]:
        """Suggest columns for one-hot encoding (low cardinality categorical)."""
        return ColumnAnalyzer.get_low_cardinality_categorical(df)

    @staticmethod
    def suggest_imputation_columns(
        df: pd.DataFrame, numeric_only: bool = True
    ) -> List[str]:
        """
        Suggest columns that need imputation.

        Args:
            df: DataFrame
            numeric_only: If True, only suggest numeric columns

        Returns:
            List of column names with missing values
        """
        missing_cols = ColumnAnalyzer.get_columns_with_missing(df)

        if numeric_only:
            numeric_cols = ColumnAnalyzer.get_numeric_columns(df)
            return [col for col in missing_cols if col in numeric_cols]

        return missing_cols

    @staticmethod
    def get_all_columns_info(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get information for all columns in DataFrame."""
        return [ColumnAnalyzer.get_column_info(df, col) for col in df.columns]
