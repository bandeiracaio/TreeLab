"""Encoding actions for categorical features."""

import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import (
    OneHotEncoder as SKLearnOneHotEncoder,
    LabelEncoder as SKLearnLabelEncoder,
)
from treelab.actions.base import Action, Parameter
from treelab.utils.column_analyzer import ColumnAnalyzer


class OneHotEncoderAction(Action):
    """Convert categorical variables to one-hot encoded columns."""

    name = "OneHotEncoder"
    description = (
        "Convert categorical columns into multiple binary columns (one-hot encoding)"
    )
    mode = "transformation"
    category = "Encoding"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Encode",
                type="columns",
                required=True,
                description="Select categorical columns to one-hot encode",
            ),
            Parameter(
                name="drop_first",
                label="Drop First Category",
                type="boolean",
                required=False,
                default=True,
                description="Drop first category to avoid multicollinearity",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate encoding parameters."""
        columns = params.get("columns", [])

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        # Check columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        # Check columns are categorical/object
        categorical_cols = ColumnAnalyzer.get_categorical_columns(df)
        non_categorical = [col for col in columns if col not in categorical_cols]
        if non_categorical:
            return False, f"Columns must be categorical/object type: {non_categorical}"

        # Warn about high cardinality
        high_cardinality = [col for col in columns if df[col].nunique() > 50]
        if high_cardinality:
            return (
                False,
                f"Columns have too many unique values (>50): {high_cardinality}. Consider using LabelEncoder instead.",
            )

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute one-hot encoding."""
        columns = params.get("columns", [])
        drop_first = params.get("drop_first", True)

        if not isinstance(columns, list):
            columns = [columns]

        # Create encoder
        encoder = SKLearnOneHotEncoder(
            drop="first" if drop_first else None,
            sparse_output=False,
            handle_unknown="ignore",
        )

        # Encode main DataFrame
        df_new = df.copy()

        # Fit encoder and transform
        encoded = encoder.fit_transform(df[columns])
        feature_names = encoder.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

        # Drop original columns and add encoded ones
        df_new = df_new.drop(columns=columns)
        df_new = pd.concat([df_new, encoded_df], axis=1)

        # Encode train/test if they exist
        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            encoded_train = encoder.transform(train_df[columns])
            encoded_train_df = pd.DataFrame(
                encoded_train, columns=feature_names, index=train_df.index
            )
            train_new = train_new.drop(columns=columns)
            train_new = pd.concat([train_new, encoded_train_df], axis=1)

        if test_df is not None:
            test_new = test_df.copy()
            encoded_test = encoder.transform(test_df[columns])
            encoded_test_df = pd.DataFrame(
                encoded_test, columns=feature_names, index=test_df.index
            )
            test_new = test_new.drop(columns=columns)
            test_new = pd.concat([test_new, encoded_test_df], axis=1)

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "encoded_columns": columns,
                "new_columns": list(feature_names),
                "num_new_columns": len(feature_names),
                "original_shape": df.shape,
                "new_shape": df_new.shape,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest categorical columns with reasonable cardinality."""
        return ColumnAnalyzer.suggest_encoding_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for one-hot encoding."""
        columns = params.get("columns", [])
        drop_first = params.get("drop_first", True)

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# One-hot encode columns: {columns}\n"
        code += f"from sklearn.preprocessing import OneHotEncoder\n"
        code += f"encoder = OneHotEncoder(drop={'first' if drop_first else None}, sparse_output=False, handle_unknown='ignore')\n"
        code += f"encoded = encoder.fit_transform(df[{columns}])\n"
        code += f"feature_names = encoder.get_feature_names_out({columns})\n"
        code += f"encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)\n"
        code += f"df = df.drop(columns={columns})\n"
        code += f"df = pd.concat([df, encoded_df], axis=1)\n"
        code += f"print(f'Encoded columns: {columns}')\n"
        code += f"print(f'Created {{len(feature_names)}} new columns')\n"
        code += f"print(f'New shape: {{df.shape}}')"

        return code


class LabelEncoderAction(Action):
    """Encode categorical variables as integer labels."""

    name = "LabelEncoder"
    description = "Encode categorical columns as integer labels (ordinal encoding)"
    mode = "transformation"
    category = "Encoding"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns to Encode",
                type="columns",
                required=True,
                description="Select categorical columns to label-encode",
            )
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate label encoding parameters."""
        columns = params.get("columns", [])

        if not columns:
            return False, "No columns selected"

        if not isinstance(columns, list):
            columns = [columns]

        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        categorical_cols = ColumnAnalyzer.get_categorical_columns(df)
        non_categorical = [col for col in columns if col not in categorical_cols]
        if non_categorical:
            return False, f"Columns must be categorical/object type: {non_categorical}"

        missing_mask = df[columns].isnull().any(axis=0)
        if isinstance(missing_mask, pd.Series):
            cols_with_missing = [
                col for col, has_missing in missing_mask.items() if bool(has_missing)
            ]
        else:
            cols_with_missing = []
        if cols_with_missing:
            return False, (
                "Columns contain missing values. Impute or drop missing values before "
                f"label encoding: {cols_with_missing}"
            )

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute label encoding."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        df_new = df.copy()
        train_new = train_df.copy() if train_df is not None else None
        test_new = test_df.copy() if test_df is not None else None

        class_mappings = {}

        for col in columns:
            encoder = SKLearnLabelEncoder()
            df_new[col] = encoder.fit_transform(df[col])

            if train_new is not None:
                train_new[col] = encoder.transform(train_new[col])

            if test_new is not None:
                test_new[col] = encoder.transform(test_new[col])

            classes = []
            if encoder.classes_ is not None:
                classes = list(encoder.classes_)
            class_mappings[col] = {
                str(label): int(idx) for idx, label in enumerate(classes)
            }

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "encoded_columns": columns,
                "class_mappings": class_mappings,
                "num_classes": {
                    col: len(mapping) for col, mapping in class_mappings.items()
                },
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest categorical columns."""
        return ColumnAnalyzer.get_categorical_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for label encoding."""
        columns = params.get("columns", [])

        if not isinstance(columns, list):
            columns = [columns]

        code = f"# Label encode columns: {columns}\n"
        code += "from sklearn.preprocessing import LabelEncoder\n"
        code += "label_encoders = {}\n"
        code += "for col in " + f"{columns}:\n"
        code += "    le = LabelEncoder()\n"
        code += "    df[col] = le.fit_transform(df[col])\n"
        code += "    label_encoders[col] = le\n"
        code += "    print(f'{col}: {len(le.classes_)} classes')"

        return code
