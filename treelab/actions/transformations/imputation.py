"""Imputation actions for handling missing values."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.impute import SimpleImputer as SKLearnSimpleImputer
from treelab.actions.base import Action, Parameter
from treelab.utils.column_analyzer import ColumnAnalyzer


class SimpleImputerAction(Action):
    """Fill missing values using simple strategies (mean, median, mode, constant)."""

    name = "SimpleImputer"
    description = "Fill missing values using mean, median, mode, or a constant value"
    mode = "transformation"
    category = "Data Cleaning"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Impute",
                type="columns",
                required=True,
                description="Select numeric columns with missing values to impute",
            ),
            Parameter(
                name="strategy",
                label="Imputation Strategy",
                type="select",
                required=True,
                default="mean",
                options=["mean", "median", "most_frequent", "constant"],
                description="Strategy for imputing missing values",
            ),
            Parameter(
                name="fill_value",
                label="Fill Value (for constant strategy)",
                type="numeric",
                required=False,
                default=0,
                description="Value to use when strategy is 'constant'",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate imputation parameters."""
        columns = params.get("columns", [])
        strategy = params.get("strategy", "mean")

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        # Check columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        # Check columns have missing values
        cols_without_missing = [col for col in columns if not df[col].isnull().any()]
        if cols_without_missing:
            return False, f"Columns have no missing values: {cols_without_missing}"

        # For mean/median, check columns are numeric
        if strategy in ["mean", "median"]:
            numeric_cols = ColumnAnalyzer.get_numeric_columns(df)
            non_numeric = [col for col in columns if col not in numeric_cols]
            if non_numeric:
                return (
                    False,
                    f"Columns must be numeric for {strategy} strategy: {non_numeric}",
                )

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute imputation."""
        columns = params.get("columns", [])
        strategy = params.get("strategy", "mean")
        fill_value = params.get("fill_value", 0)

        if not isinstance(columns, list):
            columns = [columns]

        # Create imputer
        if strategy == "constant":
            imputer = SKLearnSimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SKLearnSimpleImputer(strategy=strategy)

        # Impute main DataFrame
        df_new = df.copy()
        df_new[columns] = imputer.fit_transform(df[columns])

        # Impute train/test if they exist
        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            train_new[columns] = imputer.transform(train_df[columns])

        if test_df is not None:
            test_new = test_df.copy()
            test_new[columns] = imputer.transform(test_df[columns])

        # Calculate missing values before and after
        missing_before = {col: int(df[col].isnull().sum()) for col in columns}
        missing_after = {col: int(df_new[col].isnull().sum()) for col in columns}

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "imputed_columns": columns,
                "strategy": strategy,
                "missing_before": missing_before,
                "missing_after": missing_after,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest numeric columns with missing values."""
        return ColumnAnalyzer.suggest_imputation_columns(df, numeric_only=True)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for imputation."""
        columns = params.get("columns", [])
        strategy = params.get("strategy", "mean")
        fill_value = params.get("fill_value", 0)

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# Impute missing values: {columns}\n"
        code += f"from sklearn.impute import SimpleImputer\n"

        if strategy == "constant":
            code += f"imputer = SimpleImputer(strategy='{strategy}', fill_value={fill_value})\n"
        else:
            code += f"imputer = SimpleImputer(strategy='{strategy}')\n"

        code += f"df[{columns}] = imputer.fit_transform(df[{columns}])\n"
        code += f"print(f'Imputed columns: {columns} using {strategy} strategy')\n"
        code += f"print(f'Missing values after imputation: {{df[{columns}].isnull().sum().sum()}}')"

        return code

    def to_bigquery_sql(
        self, params: Dict[str, Any], table_name: str = "input_table"
    ) -> str:
        """Generate BigQuery SQL for imputation."""
        columns = params.get("columns", [])
        strategy = params.get("strategy", "mean")
        fill_value = params.get("fill_value", 0)

        if not isinstance(columns, list):
            columns = [columns]

        if not columns:
            return f"SELECT * FROM {table_name}"

        select_parts = []
        all_columns = []  # Would need to know original columns - this is a simplification

        for col in columns:
            if strategy == "mean":
                select_parts.append(f"AVG(`{col}`) OVER() AS `{col}`")
            elif strategy == "median":
                select_parts.append(
                    f"APPROX_QUANTILES(`{col}`, 100)[OFFSET(50)] AS `{col}`"
                )
            elif strategy == "most_frequent":
                select_parts.append(f"MODE_SIMPLE(`{col}`) AS `{col}`")
            elif strategy == "constant":
                select_parts.append(f"IFNULL(`{col}`, {fill_value}) AS `{col}`")
            else:
                select_parts.append(f"IFNULL(`{col}`, 0) AS `{col}`")

        # This is a simplified version - in practice would need CTE for mean calculation
        sql = f"""-- Impute missing values using {strategy}
SELECT * FROM (
  SELECT 
    {", ".join([f"IFNULL(`{col}`, {fill_value if strategy == 'constant' else '0'}) AS `{col}`" for col in columns])}
  FROM `{table_name}`
)"""

        return sql
