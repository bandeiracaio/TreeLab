"""Feature engineering transformation actions."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.preprocessing import PolynomialFeatures as SKLearnPolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.ensemble import (
    RandomForestClassifier as SKRandomForestClassifier,
    RandomForestRegressor as SKRandomForestRegressor,
)
from treelab.actions.base import Action, Parameter
from treelab.utils.column_analyzer import ColumnAnalyzer


class PCAAction(Action):
    """Reduce dimensionality with PCA."""

    name = "PCA"
    description = "Apply PCA to numeric columns"
    mode = "transformation"
    category = "Feature Engineering"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns",
                type="columns",
                required=True,
                description="Numeric columns to include in PCA",
            ),
            Parameter(
                name="n_components",
                label="Components",
                type="numeric",
                required=False,
                default=2,
                description="Number of components (int) or variance ratio (0-1)",
            ),
            Parameter(
                name="whiten",
                label="Whiten",
                type="boolean",
                required=False,
                default=False,
                description="Whiten components",
            ),
            Parameter(
                name="drop_original",
                label="Drop Original Columns",
                type="boolean",
                required=False,
                default=True,
                description="Remove original columns after PCA",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
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

        if df[columns].isnull().any().any():
            return False, "Selected columns contain missing values"

        n_components = params.get("n_components", 2)
        if n_components is None:
            return False, "n_components is required"

        if isinstance(n_components, (int, float)):
            if isinstance(n_components, float) and not (0 < n_components <= 1):
                return False, "n_components ratio must be between 0 and 1"
            if isinstance(n_components, int) and n_components < 1:
                return False, "n_components must be >= 1"
        else:
            return False, "n_components must be numeric"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        columns = params.get("columns", [])
        if not isinstance(columns, list):
            columns = [columns]

        n_components = params.get("n_components", 2)
        if isinstance(n_components, int):
            n_components = int(n_components)
        else:
            n_components = float(n_components)

        whiten = bool(params.get("whiten", False))
        drop_original = bool(params.get("drop_original", True))

        pca = SKLearnPCA(n_components=n_components, whiten=whiten, random_state=42)
        transformed = pca.fit_transform(df[columns])

        component_count = transformed.shape[1]
        pc_columns = [f"PC{i + 1}" for i in range(component_count)]

        df_new = df.copy()
        if drop_original:
            df_new = df_new.drop(columns=columns)
        df_new[pc_columns] = transformed

        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            train_transformed = pca.transform(train_df[columns])
            if drop_original:
                train_new = train_new.drop(columns=columns)
            train_new[pc_columns] = train_transformed

        if test_df is not None:
            test_new = test_df.copy()
            test_transformed = pca.transform(test_df[columns])
            if drop_original:
                test_new = test_new.drop(columns=columns)
            test_new[pc_columns] = test_transformed

        explained_ratio = pca.explained_variance_ratio_.tolist()
        cumulative = np.cumsum(pca.explained_variance_ratio_).tolist()

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "transform_type": "PCA",
                "columns": columns,
                "pc_columns": pc_columns,
                "n_components": component_count,
                "explained_variance_ratio": explained_ratio,
                "cumulative_variance": cumulative,
                "drop_original": drop_original,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return ColumnAnalyzer.get_numeric_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        columns = params.get("columns", [])
        if not isinstance(columns, list):
            columns = [columns]
        n_components = params.get("n_components", 2)
        whiten = bool(params.get("whiten", False))

        code = "# PCA transformation\n"
        code += "from sklearn.decomposition import PCA\n"
        code += f"pca = PCA(n_components={n_components}, whiten={whiten}, random_state=42)\n"
        code += f"pc_values = pca.fit_transform(df[{columns}])\n"
        code += "print('Explained variance:', pca.explained_variance_ratio_)"

        return code


class PolynomialFeaturesAction(Action):
    """Generate polynomial features."""

    name = "PolynomialFeatures"
    description = "Expand numeric features with polynomial terms"
    mode = "transformation"
    category = "Feature Engineering"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="columns",
                label="Columns",
                type="columns",
                required=True,
                description="Numeric columns to expand",
            ),
            Parameter(
                name="degree",
                label="Degree",
                type="numeric",
                required=False,
                default=2,
                description="Polynomial degree",
            ),
            Parameter(
                name="interaction_only",
                label="Interaction Only",
                type="boolean",
                required=False,
                default=False,
                description="Only interaction features",
            ),
            Parameter(
                name="drop_original",
                label="Drop Original Columns",
                type="boolean",
                required=False,
                default=True,
                description="Remove original columns after expansion",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
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

        degree = params.get("degree", 2)
        if degree < 2:
            return False, "Degree must be >= 2"

        if df[columns].isnull().any().any():
            return False, "Selected columns contain missing values"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        columns = params.get("columns", [])
        if not isinstance(columns, list):
            columns = [columns]

        degree = int(params.get("degree", 2))
        interaction_only = bool(params.get("interaction_only", False))
        drop_original = bool(params.get("drop_original", True))

        poly = SKLearnPolynomialFeatures(
            degree=degree,
            include_bias=False,
            interaction_only=interaction_only,
        )

        transformed = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns).tolist()

        if drop_original:
            cols_to_keep = [col for col in df.columns if col not in columns]
            df_new = pd.concat(
                [
                    df[cols_to_keep].reset_index(drop=True),
                    pd.DataFrame(transformed, columns=feature_names, index=df.index),
                ],
                axis=1,
            )
        else:
            df_new = pd.concat(
                [
                    df.reset_index(drop=True),
                    pd.DataFrame(transformed, columns=feature_names, index=df.index),
                ],
                axis=1,
            )

        train_new = None
        test_new = None

        if train_df is not None:
            train_transformed = poly.transform(train_df[columns])
            if drop_original:
                train_cols = [col for col in train_df.columns if col not in columns]
                train_new = pd.concat(
                    [
                        train_df[train_cols].reset_index(drop=True),
                        pd.DataFrame(
                            train_transformed,
                            columns=feature_names,
                            index=train_df.index,
                        ),
                    ],
                    axis=1,
                )
            else:
                train_new = pd.concat(
                    [
                        train_df.reset_index(drop=True),
                        pd.DataFrame(
                            train_transformed,
                            columns=feature_names,
                            index=train_df.index,
                        ),
                    ],
                    axis=1,
                )

        if test_df is not None:
            test_transformed = poly.transform(test_df[columns])
            if drop_original:
                test_cols = [col for col in test_df.columns if col not in columns]
                test_new = pd.concat(
                    [
                        test_df[test_cols].reset_index(drop=True),
                        pd.DataFrame(
                            test_transformed, columns=feature_names, index=test_df.index
                        ),
                    ],
                    axis=1,
                )
            else:
                test_new = pd.concat(
                    [
                        test_df.reset_index(drop=True),
                        pd.DataFrame(
                            test_transformed, columns=feature_names, index=test_df.index
                        ),
                    ],
                    axis=1,
                )

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "transform_type": "PolynomialFeatures",
                "columns": columns,
                "degree": degree,
                "interaction_only": interaction_only,
                "drop_original": drop_original,
                "new_feature_count": len(feature_names),
                "feature_names_sample": feature_names[:15],
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return ColumnAnalyzer.get_numeric_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        columns = params.get("columns", [])
        if not isinstance(columns, list):
            columns = [columns]
        degree = params.get("degree", 2)
        interaction_only = bool(params.get("interaction_only", False))

        code = "# Polynomial feature expansion\n"
        code += "from sklearn.preprocessing import PolynomialFeatures\n"
        code += f"poly = PolynomialFeatures(degree={degree}, include_bias=False, interaction_only={interaction_only})\n"
        code += f"expanded = poly.fit_transform(df[{columns}])\n"
        code += "print('New feature count:', expanded.shape[1])"

        return code


class RFEAction(Action):
    """Select features with Recursive Feature Elimination."""

    name = "RFE"
    description = "Select top features using RFE"
    mode = "transformation"
    category = "Feature Engineering"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="feature_columns",
                label="Feature Columns",
                type="columns",
                required=True,
                description="Numeric features to evaluate",
            ),
            Parameter(
                name="target_column",
                label="Target Column",
                type="column",
                required=True,
                description="Target column for RFE",
            ),
            Parameter(
                name="n_features_to_select",
                label="Features to Select",
                type="numeric",
                required=False,
                default=5,
                description="Number of features to keep",
            ),
            Parameter(
                name="drop_unselected",
                label="Drop Unselected",
                type="boolean",
                required=False,
                default=True,
                description="Remove unselected feature columns",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        feature_columns = params.get("feature_columns", [])
        target_column = params.get("target_column")

        if not feature_columns:
            return False, "Feature columns are required"

        if not isinstance(feature_columns, list):
            feature_columns = [feature_columns]

        if not target_column:
            return False, "Target column is required"

        missing_cols = [
            col for col in feature_columns + [target_column] if col not in df.columns
        ]
        if missing_cols:
            return False, f"Columns not found: {missing_cols}"

        numeric_cols = ColumnAnalyzer.get_numeric_columns(df)
        non_numeric = [col for col in feature_columns if col not in numeric_cols]
        if non_numeric:
            return False, f"Feature columns must be numeric: {non_numeric}"

        if df[feature_columns + [target_column]].isnull().any().any():
            return False, "Selected columns contain missing values"

        n_features = int(params.get("n_features_to_select", 5))
        if n_features < 1 or n_features > len(feature_columns):
            return (
                False,
                "n_features_to_select must be between 1 and number of features",
            )

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        feature_columns = params.get("feature_columns", [])
        if not isinstance(feature_columns, list):
            feature_columns = [feature_columns]

        target_column = params.get("target_column")
        n_features = int(params.get("n_features_to_select", 5))
        drop_unselected = bool(params.get("drop_unselected", True))

        X = df[feature_columns]
        y = df[target_column]

        unique_count = y.nunique()
        is_classification = y.dtype == "object" or unique_count <= 20

        if is_classification:
            estimator = SKRandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )
        else:
            estimator = SKRandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            )

        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        selector.fit(X, y)

        ranking = selector.ranking_.tolist()
        support = selector.support_.tolist()
        selected_features = [col for col, keep in zip(feature_columns, support) if keep]

        drop_cols = []
        if drop_unselected:
            drop_cols = [col for col in feature_columns if col not in selected_features]

        df_new = df.copy()
        if drop_unselected and drop_cols:
            df_new = df_new.drop(columns=drop_cols)

        train_new = None
        test_new = None

        if train_df is not None:
            train_new = train_df.copy()
            if drop_unselected and drop_cols:
                train_new = train_new.drop(columns=drop_cols)

        if test_df is not None:
            test_new = test_df.copy()
            if drop_unselected and drop_cols:
                test_new = test_new.drop(columns=drop_cols)

        rankings = {col: int(rank) for col, rank in zip(feature_columns, ranking)}

        return {
            "df": df_new,
            "train_df": train_new,
            "test_df": test_new,
            "model": None,
            "metadata": {
                "transform_type": "RFE",
                "feature_columns": feature_columns,
                "target_column": target_column,
                "n_features_to_select": n_features,
                "selected_features": selected_features,
                "rankings": rankings,
                "drop_unselected": drop_unselected,
            },
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return ColumnAnalyzer.get_numeric_columns(df)

    def to_python_code(self, params: Dict[str, Any]) -> str:
        code = "# RFE feature selection\n"
        code += "from sklearn.feature_selection import RFE\n"
        code += "from sklearn.ensemble import RandomForestClassifier\n"
        code += "selector = RFE(RandomForestClassifier(n_estimators=200, random_state=42), n_features_to_select=5)\n"
        code += "selector.fit(X, y)\n"
        code += "print('Selected:', selector.support_)"

        return code
