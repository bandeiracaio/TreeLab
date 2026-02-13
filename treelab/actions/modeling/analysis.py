"""Analysis actions for modeling outputs."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast, Hashable
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from treelab.actions.base import Action, Parameter


class FeatureImportanceAction(Action):
    """Fit a model and compute feature importance for visualization."""

    name = "PlotFeatureImportance"
    description = "Fit a random forest and compute feature importance"
    mode = "modeling"

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
            Parameter(
                name="importance_method",
                label="Importance Method",
                type="select",
                required=False,
                default="permutation",
                options=["permutation", "gini"],
                description="Permutation importance (model-agnostic) or Gini importance",
            ),
            Parameter(
                name="n_repeats",
                label="Permutation Repeats",
                type="numeric",
                required=False,
                default=10,
                description="Number of permutation repeats (only for permutation method)",
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
            return (
                False,
                "Must perform train/test split before computing feature importance",
            )

        n_estimators = params.get("n_estimators", 200)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        n_repeats = params.get("n_repeats", 10)
        if params.get("importance_method") == "permutation" and n_repeats < 1:
            return False, "n_repeats must be >= 1 for permutation importance"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit model and compute feature importance."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        train_df = cast(pd.DataFrame, train_df)
        test_df = cast(pd.DataFrame, test_df)

        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        importance_method = params.get("importance_method", "permutation")
        n_repeats = int(params.get("n_repeats", 10))

        target_col = cast(Hashable, train_df.columns[-1])

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        model = SKRandomForestClassifier(
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
            zero_division="warn",
        )
        test_recall = recall_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division="warn",
        )
        test_f1 = f1_score(
            y_test,
            y_pred_test,
            average=average_param,
            zero_division="warn",
        )

        cm = confusion_matrix(y_test, y_pred_test)

        if importance_method == "permutation":
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1,
            )
            importances = cast(Any, perm).importances_mean
        else:
            importances = model.feature_importances_

        feature_importance = dict(zip(X_train.columns, importances))

        metadata = {
            "model_type": "FeatureImportanceRandomForest",
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
            "importance_method": importance_method,
            "n_repeats": n_repeats if importance_method == "permutation" else None,
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
        """Generate Python code for feature importance modeling."""
        n_estimators = params.get("n_estimators", 200)
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)
        importance_method = params.get("importance_method", "permutation")
        n_repeats = params.get("n_repeats", 10)

        code = "# Feature importance with Random Forest\n"
        code += "from sklearn.ensemble import RandomForestClassifier\n"
        code += "from sklearn.inspection import permutation_importance\n"
        code += "from sklearn.metrics import accuracy_score\n\n"
        code += "model = RandomForestClassifier(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += "    random_state=42,\n"
        code += "    n_jobs=-1\n"
        code += ")\n\n"
        code += "model.fit(X_train, y_train)\n"
        code += "y_pred = model.predict(X_test)\n"
        code += "print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')\n\n"
        code += f"importance_method = '{importance_method}'\n"
        code += "if importance_method == 'permutation':\n"
        code += f"    perm = permutation_importance(model, X_test, y_test, n_repeats={n_repeats}, random_state=42, n_jobs=-1)\n"
        code += "    importances = perm.importances_mean\n"
        code += "else:\n"
        code += "    importances = model.feature_importances_\n"
        code += "feature_importance = dict(zip(X_train.columns, importances))\n"
        code += "print('Top features:')\n"
        code += "for name, value in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:\n"
        code += "    print(f'{name}: {value:.4f}')"

        return code
