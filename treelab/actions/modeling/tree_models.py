"""Tree-based model actions."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast, Hashable
from sklearn.tree import (
    DecisionTreeClassifier as SKDecisionTreeClassifier,
    DecisionTreeRegressor as SKDecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier as SKRandomForestClassifier,
    RandomForestRegressor as SKRandomForestRegressor,
    GradientBoostingClassifier as SKGradientBoostingClassifier,
    ExtraTreesClassifier as SKExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from treelab.actions.base import Action, Parameter


class DecisionTreeClassifierAction(Action):
    """Fit a Decision Tree Classifier."""

    name = "DecisionTreeClassifier"
    description = "Fit a decision tree for classification"
    mode = "modeling"
    category = "Classification"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=5,
                description="Maximum depth of the tree (None for unlimited)",
            ),
            Parameter(
                name="min_samples_split",
                label="Min Samples to Split",
                type="numeric",
                required=False,
                default=2,
                description="Minimum samples required to split an internal node",
            ),
            Parameter(
                name="min_samples_leaf",
                label="Min Samples in Leaf",
                type="numeric",
                required=False,
                default=1,
                description="Minimum samples required in a leaf node",
            ),
            Parameter(
                name="criterion",
                label="Split Criterion",
                type="select",
                required=False,
                default="gini",
                options=["gini", "entropy"],
                description="Function to measure split quality",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate model parameters."""
        # Check train/test split has been done
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        # Check parameters
        max_depth = params.get("max_depth")
        if max_depth is not None and (
            not isinstance(max_depth, (int, float)) or max_depth < 1
        ):
            return False, "max_depth must be a positive number or None"

        min_samples_split = params.get("min_samples_split", 2)
        if min_samples_split < 2:
            return False, "min_samples_split must be >= 2"

        min_samples_leaf = params.get("min_samples_leaf", 1)
        if min_samples_leaf < 1:
            return False, "min_samples_leaf must be >= 1"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit Decision Tree Classifier."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        # Extract parameters
        max_depth = params.get("max_depth", 5)
        if max_depth == 0:  # Allow unlimited depth with 0 or None
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        criterion = params.get("criterion", "gini")

        # Get target column from state (should be stored from TrainTestSplit)
        # For now, we'll assume last column is target
        target_col = cast(Hashable, train_df.columns[-1])

        # Prepare data
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Create and fit model
        model = SKDecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # For binary classification, calculate additional metrics
        average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"

        test_precision = precision_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_recall = recall_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_f1 = f1_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)

        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "DecisionTreeClassifier",
            "task": "classification",
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "classes": list(model.classes_),
            "tree_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Not applicable for models."""
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for Decision Tree."""
        max_depth = params.get("max_depth", 5)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)
        criterion = params.get("criterion", "gini")

        code = f"# Fit Decision Tree Classifier\n"
        code += f"from sklearn.tree import DecisionTreeClassifier\n"
        code += f"from sklearn.metrics import accuracy_score, classification_report\n\n"
        code += f"model = DecisionTreeClassifier(\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += f"    criterion='{criterion}',\n"
        code += f"    random_state=42\n"
        code += f")\n\n"
        code += f"model.fit(X_train, y_train)\n"
        code += f"y_pred = model.predict(X_test)\n\n"
        code += f"print(f'Train Accuracy: {{accuracy_score(y_train, model.predict(X_train)):.4f}}')\n"
        code += f"print(f'Test Accuracy: {{accuracy_score(y_test, y_pred):.4f}}')\n"
        code += f"print('\\nClassification Report:')\n"
        code += f"print(classification_report(y_test, y_pred))"

        return code


class RandomForestClassifierAction(Action):
    """Fit a Random Forest Classifier."""

    name = "RandomForestClassifier"
    description = "Fit a random forest ensemble for classification"
    mode = "modeling"
    category = "Classification"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_estimators",
                label="Number of Trees",
                type="numeric",
                required=False,
                default=100,
                description="Number of trees in the forest",
            ),
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=10,
                description="Maximum depth of each tree (None for unlimited)",
            ),
            Parameter(
                name="min_samples_split",
                label="Min Samples to Split",
                type="numeric",
                required=False,
                default=2,
                description="Minimum samples required to split an internal node",
            ),
            Parameter(
                name="min_samples_leaf",
                label="Min Samples in Leaf",
                type="numeric",
                required=False,
                default=1,
                description="Minimum samples required in a leaf node",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate model parameters."""
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        n_estimators = params.get("n_estimators", 100)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit Random Forest Classifier."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        # Extract parameters
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        # Get target column
        target_col = cast(Hashable, train_df.columns[-1])

        # Prepare data
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Create and fit model
        model = SKRandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )

        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"

        test_precision = precision_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_recall = recall_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_f1 = f1_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)

        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "RandomForestClassifier",
            "task": "classification",
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "classes": list(model.classes_),
            "n_estimators": n_estimators,
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Not applicable for models."""
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for Random Forest."""
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)

        code = f"# Fit Random Forest Classifier\n"
        code += f"from sklearn.ensemble import RandomForestClassifier\n"
        code += f"from sklearn.metrics import accuracy_score, classification_report\n\n"
        code += f"model = RandomForestClassifier(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += f"    random_state=42,\n"
        code += f"    n_jobs=-1\n"
        code += f")\n\n"
        code += f"model.fit(X_train, y_train)\n"
        code += f"y_pred = model.predict(X_test)\n\n"
        code += f"print(f'Train Accuracy: {{accuracy_score(y_train, model.predict(X_train)):.4f}}')\n"
        code += f"print(f'Test Accuracy: {{accuracy_score(y_test, y_pred):.4f}}')\n"
        code += f"print('\\nClassification Report:')\n"
        code += f"print(classification_report(y_test, y_pred))"

        return code


class DecisionTreeRegressorAction(Action):
    """Fit a Decision Tree Regressor."""

    name = "DecisionTreeRegressor"
    description = "Fit a decision tree for regression"
    mode = "modeling"
    category = "Regression"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=5,
                description="Maximum depth of the tree (None for unlimited)",
            ),
            Parameter(
                name="min_samples_split",
                label="Min Samples to Split",
                type="numeric",
                required=False,
                default=2,
                description="Minimum samples required to split an internal node",
            ),
            Parameter(
                name="min_samples_leaf",
                label="Min Samples in Leaf",
                type="numeric",
                required=False,
                default=1,
                description="Minimum samples required in a leaf node",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate model parameters."""
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        max_depth = params.get("max_depth")
        if max_depth is not None and (
            not isinstance(max_depth, (int, float)) or max_depth < 1
        ):
            return False, "max_depth must be a positive number or None"

        min_samples_split = params.get("min_samples_split", 2)
        if min_samples_split < 2:
            return False, "min_samples_split must be >= 2"

        min_samples_leaf = params.get("min_samples_leaf", 1)
        if min_samples_leaf < 1:
            return False, "min_samples_leaf must be >= 1"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit Decision Tree Regressor."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        max_depth = params.get("max_depth", 5)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        target_col = cast(Hashable, train_df.columns[-1])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = SKDecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5

        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "DecisionTreeRegressor",
            "task": "regression",
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_test.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "tree_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves(),
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Not applicable for models."""
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for Decision Tree Regressor."""
        max_depth = params.get("max_depth", 5)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)

        code = "# Fit Decision Tree Regressor\n"
        code += "from sklearn.tree import DecisionTreeRegressor\n"
        code += "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n\n"
        code += "model = DecisionTreeRegressor(\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += "    random_state=42\n"
        code += ")\n\n"
        code += "model.fit(X_train, y_train)\n"
        code += "y_pred = model.predict(X_test)\n\n"
        code += "print(f'Test R2: {r2_score(y_test, y_pred):.4f}')\n"
        code += "print(f'Test MAE: {mean_absolute_error(y_test, y_pred):.4f}')\n"
        code += "print(f'Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}')"

        return code


class RandomForestRegressorAction(Action):
    """Fit a Random Forest Regressor."""

    name = "RandomForestRegressor"
    description = "Fit a random forest ensemble for regression"
    mode = "modeling"
    category = "Regression"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_estimators",
                label="Number of Trees",
                type="numeric",
                required=False,
                default=200,
                description="Number of trees in the forest",
            ),
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=10,
                description="Maximum depth of each tree (None for unlimited)",
            ),
            Parameter(
                name="min_samples_split",
                label="Min Samples to Split",
                type="numeric",
                required=False,
                default=2,
                description="Minimum samples required to split an internal node",
            ),
            Parameter(
                name="min_samples_leaf",
                label="Min Samples in Leaf",
                type="numeric",
                required=False,
                default=1,
                description="Minimum samples required in a leaf node",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """Validate model parameters."""
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        n_estimators = params.get("n_estimators", 200)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit Random Forest Regressor."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        target_col = cast(Hashable, train_df.columns[-1])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = SKRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5

        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "RandomForestRegressor",
            "task": "regression",
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_test.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "n_estimators": n_estimators,
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Not applicable for models."""
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for Random Forest Regressor."""
        n_estimators = params.get("n_estimators", 200)
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)

        code = "# Fit Random Forest Regressor\n"
        code += "from sklearn.ensemble import RandomForestRegressor\n"
        code += "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n\n"
        code += "model = RandomForestRegressor(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += "    random_state=42,\n"
        code += "    n_jobs=-1\n"
        code += ")\n\n"
        code += "model.fit(X_train, y_train)\n"
        code += "y_pred = model.predict(X_test)\n\n"
        code += "print(f'Test R2: {r2_score(y_test, y_pred):.4f}')\n"
        code += "print(f'Test MAE: {mean_absolute_error(y_test, y_pred):.4f}')\n"
        code += "print(f'Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}')"

        return code


class GradientBoostingClassifierAction(Action):
    """Fit a Gradient Boosting Classifier."""

    name = "GradientBoostingClassifier"
    description = "Gradient boosting trees for classification"
    mode = "modeling"
    category = "Classification"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_estimators",
                label="Number of Trees",
                type="numeric",
                required=False,
                default=200,
                description="Number of boosting stages",
            ),
            Parameter(
                name="learning_rate",
                label="Learning Rate",
                type="numeric",
                required=False,
                default=0.1,
                description="Learning rate shrinks contribution of each tree",
            ),
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=3,
                description="Max depth of individual trees",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        n_estimators = params.get("n_estimators", 200)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        learning_rate = params.get("learning_rate", 0.1)
        if learning_rate <= 0:
            return False, "learning_rate must be > 0"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        n_estimators = int(params.get("n_estimators", 200))
        learning_rate = float(params.get("learning_rate", 0.1))
        max_depth = int(params.get("max_depth", 3))

        target_col = cast(Hashable, train_df.columns[-1])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = SKGradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"

        test_precision = precision_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_recall = recall_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_f1 = f1_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )

        cm = confusion_matrix(y_test, y_pred_test)

        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "GradientBoostingClassifier",
            "task": "classification",
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "classes": list(model.classes_),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        n_estimators = params.get("n_estimators", 200)
        learning_rate = params.get("learning_rate", 0.1)
        max_depth = params.get("max_depth", 3)

        code = "# Fit Gradient Boosting Classifier\n"
        code += "from sklearn.ensemble import GradientBoostingClassifier\n"
        code += "model = GradientBoostingClassifier(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    learning_rate={learning_rate},\n"
        code += f"    max_depth={max_depth},\n"
        code += "    random_state=42\n"
        code += ")\n"
        code += "model.fit(X_train, y_train)\n"
        code += "print('Train accuracy:', model.score(X_train, y_train))\n"
        code += "print('Test accuracy:', model.score(X_test, y_test))"

        return code


class ExtraTreesClassifierAction(Action):
    """Fit an Extra Trees Classifier."""

    name = "ExtraTreesClassifier"
    description = "Extra Trees ensemble for classification"
    mode = "modeling"
    category = "Classification"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_estimators",
                label="Number of Trees",
                type="numeric",
                required=False,
                default=300,
                description="Number of trees in the ensemble",
            ),
            Parameter(
                name="max_depth",
                label="Maximum Depth",
                type="numeric",
                required=False,
                default=10,
                description="Maximum depth of each tree (None for unlimited)",
            ),
            Parameter(
                name="min_samples_split",
                label="Min Samples to Split",
                type="numeric",
                required=False,
                default=2,
                description="Minimum samples required to split an internal node",
            ),
            Parameter(
                name="min_samples_leaf",
                label="Min Samples in Leaf",
                type="numeric",
                required=False,
                default=1,
                description="Minimum samples required in a leaf node",
            ),
        ]

    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        if train_df is None or test_df is None:
            return False, "Must perform train/test split before fitting a model"

        n_estimators = params.get("n_estimators", 300)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        n_estimators = int(params.get("n_estimators", 300))
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))

        target_col = cast(Hashable, train_df.columns[-1])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = SKExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"

        test_precision = precision_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_recall = recall_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )
        test_f1 = f1_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division=cast(Any, 0),
        )

        cm = confusion_matrix(y_test, y_pred_test)

        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": "ExtraTreesClassifier",
            "task": "classification",
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
            "confusion_matrix": cm.tolist(),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
            "classes": list(model.classes_),
            "n_estimators": n_estimators,
        }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        n_estimators = params.get("n_estimators", 300)
        max_depth = params.get("max_depth", 10)
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)

        code = "# Fit Extra Trees Classifier\n"
        code += "from sklearn.ensemble import ExtraTreesClassifier\n"
        code += "model = ExtraTreesClassifier(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += "    random_state=42,\n"
        code += "    n_jobs=-1\n"
        code += ")\n"
        code += "model.fit(X_train, y_train)\n"
        code += "print('Train accuracy:', model.score(X_train, y_train))\n"
        code += "print('Test accuracy:', model.score(X_test, y_test))"

        return code
