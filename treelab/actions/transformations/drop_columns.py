"""Drop columns action."""

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from treelab.actions.base import Action, Parameter


class DropColumnsAction(Action):
    """Drop specified columns from the DataFrame."""

    name = "DropColumns"
    description = "Remove specified columns from the dataset"
    mode = "transformation"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Drop",
                type="columns",
                required=True,
                description="Select columns to remove from the dataset",
            )
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate that columns exist in DataFrame."""
        columns = params.get("columns", [])

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        # Check all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found in DataFrame: {missing_cols}"

        # Check we're not dropping all columns
        if len(columns) >= len(df.columns):
            return False, "Cannot drop all columns"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Drop the specified columns."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        # Drop from main DataFrame
        df_new = df.drop(columns=columns)

        # Drop from train/test if they exist
        train_new = train_df.drop(columns=columns) if train_df is not None else None
        test_new = test_df.drop(columns=columns) if test_df is not None else None

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {"dropped_columns": columns, "new_shape": df_new.shape},
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest columns that might be good to drop.

        Common candidates: ID columns, high-cardinality text columns, columns with all nulls.
        """
        suggestions = []

        for col in df.columns:
            # Suggest ID-like columns
            if "id" in col.lower() or col.lower().endswith("_id"):
                suggestions.append(col)

            # Suggest columns with all null values
            elif df[col].isnull().all():
                suggestions.append(col)

            # Suggest very high cardinality text columns (likely not useful)
            elif df[col].dtype == "object" and df[col].nunique() / len(df) > 0.95:
                suggestions.append(col)

        return suggestions

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for dropping columns."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# Drop columns: {columns}\n"
        code += f"df = df.drop(columns={columns})\n"
        code += f"print(f'Dropped columns: {columns}')\n"
        code += f"print(f'New shape: {{df.shape}}')"

        return code

    def to_bigquery_sql(
        self, params: Dict[str, Any], table_name: str = "input_table"
    ) -> str:
        """Generate BigQuery SQL for dropping columns."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        if not columns:
            return f"SELECT * FROM {table_name}"

        cols_to_select = [f"`{col}`" for col in columns]
        sql = f"SELECT {', '.join(cols_to_select)}\nFROM `{table_name}`"

        return sql
