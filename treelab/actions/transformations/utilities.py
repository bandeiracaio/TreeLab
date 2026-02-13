"""Utility transformation actions."""

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from treelab.actions.base import Action, Parameter


class TrainTestSplitAction(Action):
    """Split data into training and test sets."""

    name = "TrainTestSplit"
    description = "Split DataFrame into training and test sets for model evaluation"
    mode = "transformation"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="target_column",
                label="Target Column",
                type="column",
                required=True,
                description="Column to use as the target variable (y)",
            ),
            Parameter(
                name="test_size",
                label="Test Size (0-1 or integer)",
                type="numeric",
                required=False,
                default=0.2,
                description="Proportion of dataset for test set (e.g., 0.2 = 20%)",
            ),
            Parameter(
                name="random_state",
                label="Random State (seed)",
                type="numeric",
                required=False,
                default=42,
                description="Random seed for reproducibility",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate split parameters."""
        target_column = params.get("target_column")
        test_size = params.get("test_size", 0.2)

        if not target_column:
            return False, "Target column not specified"

        if target_column not in df.columns:
            return False, f"Target column '{target_column}' not found in DataFrame"

        # Validate test_size
        if isinstance(test_size, float):
            if not (0.0 < test_size < 1.0):
                return False, "Test size must be between 0 and 1 when using proportion"
        elif isinstance(test_size, int):
            if test_size >= len(df):
                return False, "Test size cannot be >= number of samples"
        else:
            return False, "Test size must be a number"

        # Check if we have enough samples
        if len(df) < 10:
            return False, "Not enough samples for train/test split (need at least 10)"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute train/test split."""
        target_column = params.get("target_column")
        test_size = params.get("test_size", 0.2)
        random_state = params.get("random_state", 42)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Perform split
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Reconstruct train and test DataFrames
        train_new = X_train.copy()
        train_new[target_column] = y_train

        test_new = X_test.copy()
        test_new[target_column] = y_test

        return {
            "df": df,  # Keep original df unchanged
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "target_column": target_column,
            "metadata": {
                "target_column": target_column,
                "test_size": test_size,
                "train_shape": train_new.shape,
                "test_shape": test_new.shape,
                "train_samples": len(train_new),
                "test_samples": len(test_new),
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest potential target columns.

        Look for columns that might be targets (binary, categorical with few classes).
        """
        suggestions = []

        for col in df.columns:
            # Binary columns
            if df[col].nunique() == 2:
                suggestions.append(col)
            # Low cardinality categorical (possible classification target)
            elif df[col].nunique() <= 10 and df[col].dtype == "object":
                suggestions.append(col)
            # Columns with 'target', 'label', 'class' in name
            elif any(
                keyword in col.lower() for keyword in ["target", "label", "class", "y"]
            ):
                suggestions.append(col)

        return suggestions

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for train/test split."""
        target_column = params.get("target_column")
        test_size = params.get("test_size", 0.2)
        random_state = params.get("random_state", 42)

        code = f"# Train/Test Split\n"
        code += f"from sklearn.model_selection import train_test_split\n"
        code += f"X = df.drop(columns=['{target_column}'])\n"
        code += f"y = df['{target_column}']\n"
        code += f"X_train, X_test, y_train, y_test = train_test_split(\n"
        code += f"    X, y, test_size={test_size}, random_state={random_state}\n"
        code += f")\n"
        code += f"print(f'Train size: {{len(X_train)}} samples')\n"
        code += f"print(f'Test size: {{len(X_test)}} samples')\n"
        code += f"print(f'Target column: {target_column}')"

        return code
