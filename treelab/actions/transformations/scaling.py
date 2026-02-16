"""Scaling actions for numeric features."""

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import (
    StandardScaler as SKLearnStandardScaler,
    MinMaxScaler as SKLearnMinMaxScaler,
)
from treelab.actions.base import Action, Parameter
from treelab.utils.column_analyzer import ColumnAnalyzer


class StandardScalerAction(Action):
    """Standardize features by removing mean and scaling to unit variance (z-score normalization)."""

    name = "StandardScaler"
    description = (
        "Standardize numeric features to have mean=0 and std=1 (z-score normalization)"
    )
    mode = "transformation"
    category = "Scaling"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Scale",
                type="columns",
                required=True,
                description="Select numeric columns to standardize",
            )
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate scaling parameters."""
        columns = params.get("columns", [])

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        # Check columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        # Check columns are numeric
        numeric_cols = ColumnAnalyzer.get_numeric_columns(df)
        non_numeric = [col for col in columns if col not in numeric_cols]
        if non_numeric:
            return False, f"Columns must be numeric: {non_numeric}"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute scaling."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        # Create scaler
        scaler = SKLearnStandardScaler()

        # Scale main DataFrame
        df_new = df.copy()
        df_new[columns] = scaler.fit_transform(df[columns])

        # Scale train/test if they exist
        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            train_new[columns] = scaler.transform(train_df[columns])

        if test_df is not None:
            test_new = test_df.copy()
            test_new[columns] = scaler.transform(test_df[columns])

        # Calculate statistics
        means_before = {col: float(df[col].mean()) for col in columns}
        stds_before = {col: float(df[col].std()) for col in columns}
        means_after = {col: float(df_new[col].mean()) for col in columns}
        stds_after = {col: float(df_new[col].std()) for col in columns}

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "scaled_columns": columns,
                "means_before": means_before,
                "stds_before": stds_before,
                "means_after": means_after,
                "stds_after": stds_after,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest all numeric columns."""
        return ColumnAnalyzer.suggest_scaling_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for scaling."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# Standardize columns: {columns}\n"
        code += f"from sklearn.preprocessing import StandardScaler\n"
        code += f"scaler = StandardScaler()\n"
        code += f"df[{columns}] = scaler.fit_transform(df[{columns}])\n"
        code += f"print(f'Scaled columns: {columns}')\n"
        code += f"print(f'New means: {{df[{columns}].mean().round(3).to_dict()}}')\n"
        code += f"print(f'New stds: {{df[{columns}].std().round(3).to_dict()}}')"

        return code


class MinMaxScalerAction(Action):
    """Scale features to a [0, 1] range using MinMax scaling."""

    name = "MinMaxScaler"
    description = "Scale numeric features to a [0, 1] range"
    mode = "transformation"
    category = "Scaling"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Scale",
                type="columns",
                required=True,
                description="Select numeric columns to scale to [0, 1]",
            )
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate scaling parameters."""
        columns = params.get("columns", [])

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        numeric_cols = ColumnAnalyzer.get_numeric_columns(df)
        non_numeric = [col for col in columns if col not in numeric_cols]
        if non_numeric:
            return False, f"Columns must be numeric: {non_numeric}"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute MinMax scaling."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        scaler = SKLearnMinMaxScaler()

        df_new = df.copy()
        df_new[columns] = scaler.fit_transform(df[columns])

        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            train_new[columns] = scaler.transform(train_df[columns])

        if test_df is not None:
            test_new = test_df.copy()
            test_new[columns] = scaler.transform(test_df[columns])

        mins_before = {col: float(df[col].min()) for col in columns}
        maxs_before = {col: float(df[col].max()) for col in columns}
        mins_after = {col: float(df_new[col].min()) for col in columns}
        maxs_after = {col: float(df_new[col].max()) for col in columns}

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "scaled_columns": columns,
                "mins_before": mins_before,
                "maxs_before": maxs_before,
                "mins_after": mins_after,
                "maxs_after": maxs_after,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest all numeric columns."""
        return ColumnAnalyzer.suggest_scaling_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for MinMax scaling."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# MinMax scale columns: {columns}\n"
        code += "from sklearn.preprocessing import MinMaxScaler\n"
        code += "scaler = MinMaxScaler()\n"
        code += f"df[{columns}] = scaler.fit_transform(df[{columns}])\n"
        code += f"print(f'Scaled columns: {columns}')\n"
        code += f"print(f'New mins: {{df[{columns}].min().round(3).to_dict()}}')\n"
        code += f"print(f'New maxs: {{df[{columns}].max().round(3).to_dict()}}')"

        return code
