"""Hyperparameter tuning actions for TreeLab."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast, Hashable
from sklearn.model_selection import GridSearchCV
from sklearn.tree import (
    DecisionTreeClassifier as SKDecisionTreeClassifier,
    DecisionTreeRegressor as SKDecisionTreeRegressor,
)
from sklearn.ensemble import (
    RandomForestClassifier as SKRandomForestClassifier,
    RandomForestRegressor as SKRandomForestRegressor,
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


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _parse_optional_int_list(raw: str) -> List[Optional[int]]:
    values: List[Optional[int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(item))
    return values


class TuneHyperparametersAction(Action):
    """Tune hyperparameters with GridSearchCV."""

    name = "TuneHyperparameters"
    description = "Grid search hyperparameters for tree models"
    mode = "modeling"

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="task",
                label="Task Type",
                type="select",
                required=False,
                default="classification",
                options=["classification", "regression"],
                description="Choose classification or regression",
            ),
            Parameter(
                name="model_type",
                label="Model Type",
                type="select",
                required=False,
                default="random_forest",
                options=["decision_tree", "random_forest"],
                description="Decision tree or random forest",
            ),
            Parameter(
                name="max_depth_values",
                label="Max Depth Values",
                type="text",
                required=False,
                default="5,10,None",
                description="Comma-separated values for max_depth (use None for unlimited)",
            ),
            Parameter(
                name="min_samples_split_values",
                label="Min Samples Split Values",
                type="text",
                required=False,
                default="2,5,10",
                description="Comma-separated values for min_samples_split",
            ),
            Parameter(
                name="min_samples_leaf_values",
                label="Min Samples Leaf Values",
                type="text",
                required=False,
                default="1,2,4",
                description="Comma-separated values for min_samples_leaf",
            ),
            Parameter(
                name="n_estimators_values",
                label="N Estimators Values",
                type="text",
                required=False,
                default="100,200",
                description="Comma-separated values for n_estimators (random forest only)",
            ),
            Parameter(
                name="cv_folds",
                label="CV Folds",
                type="numeric",
                required=False,
                default=5,
                description="Number of cross-validation folds",
            ),
            Parameter(
                name="scoring",
                label="Scoring",
                type="text",
                required=False,
                default="",
                description="Optional scoring override (e.g., accuracy, f1, r2, neg_mean_absolute_error)",
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
            return False, "Must perform train/test split before tuning"

        cv_folds = params.get("cv_folds", 5)
        if cv_folds < 2:
            return False, "cv_folds must be >= 2"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before tuning")

        task = params.get("task", "classification")
        model_type = params.get("model_type", "random_forest")
        cv_folds = int(params.get("cv_folds", 5))

        max_depth_values = _parse_optional_int_list(
            params.get("max_depth_values", "5,10,None")
        )
        min_samples_split_values = _parse_int_list(
            params.get("min_samples_split_values", "2,5,10")
        )
        min_samples_leaf_values = _parse_int_list(
            params.get("min_samples_leaf_values", "1,2,4")
        )
        n_estimators_values = _parse_int_list(
            params.get("n_estimators_values", "100,200")
        )

        target_col = cast(Hashable, train_df.columns[-1])
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        if task == "classification":
            estimator = (
                SKRandomForestClassifier(random_state=42, n_jobs=-1)
                if model_type == "random_forest"
                else SKDecisionTreeClassifier(random_state=42)
            )
            default_scoring = "accuracy"
        else:
            estimator = (
                SKRandomForestRegressor(random_state=42, n_jobs=-1)
                if model_type == "random_forest"
                else SKDecisionTreeRegressor(random_state=42)
            )
            default_scoring = "r2"

        scoring_override = params.get("scoring", "").strip()
        scoring = scoring_override if scoring_override else default_scoring

        param_grid: Dict[str, Any] = {
            "max_depth": max_depth_values,
            "min_samples_split": min_samples_split_values,
            "min_samples_leaf": min_samples_leaf_values,
        }
        if model_type == "random_forest":
            param_grid["n_estimators"] = n_estimators_values

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
            refit=True,
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred_test = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)

        metadata: Dict[str, Any] = {
            "model_type": f"Tuned{best_model.__class__.__name__}",
            "task": task,
            "best_params": grid.best_params_,
            "best_score": float(grid.best_score_),
            "cv_folds": cv_folds,
            "scoring": scoring,
            "target_column": target_col,
        }

        if task == "classification":
            average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"
            metadata.update(
                {
                    "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
                    "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
                    "test_precision": float(
                        precision_score(
                            y_test,
                            y_pred_test,
                            average=average_param,
                            zero_division=cast(Any, 0),
                        )
                    ),
                    "test_recall": float(
                        recall_score(
                            y_test,
                            y_pred_test,
                            average=average_param,
                            zero_division=cast(Any, 0),
                        )
                    ),
                    "test_f1": float(
                        f1_score(
                            y_test,
                            y_pred_test,
                            average=average_param,
                            zero_division=cast(Any, 0),
                        )
                    ),
                    "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
                    "classes": list(best_model.classes_),
                }
            )
        else:
            metadata.update(
                {
                    "train_r2": float(r2_score(y_train, y_pred_train)),
                    "test_r2": float(r2_score(y_test, y_pred_test)),
                    "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
                    "test_rmse": float(mean_squared_error(y_test, y_pred_test) ** 0.5),
                    "y_test": y_test.tolist(),
                    "y_pred": y_pred_test.tolist(),
                }
            )

        if hasattr(best_model, "feature_importances_"):
            feature_importance = dict(
                zip(X_train.columns, best_model.feature_importances_)
            )
            metadata["feature_importance"] = {
                k: float(v) for k, v in feature_importance.items()
            }

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": best_model,
            "metadata": metadata,
        }

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """Not applicable for models."""
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        """Generate Python code for grid search tuning."""
        task = params.get("task", "classification")
        model_type = params.get("model_type", "random_forest")
        max_depth_values = params.get("max_depth_values", "5,10,None")
        min_samples_split_values = params.get("min_samples_split_values", "2,5,10")
        min_samples_leaf_values = params.get("min_samples_leaf_values", "1,2,4")
        n_estimators_values = params.get("n_estimators_values", "100,200")
        cv_folds = params.get("cv_folds", 5)
        scoring_override = params.get("scoring", "").strip()
        scoring_default = "accuracy" if task == "classification" else "r2"
        scoring = scoring_override if scoring_override else scoring_default

        code = "# Hyperparameter tuning with GridSearchCV\n"
        code += "from sklearn.model_selection import GridSearchCV\n"
        if task == "classification":
            if model_type == "random_forest":
                code += "from sklearn.ensemble import RandomForestClassifier\n"
                code += (
                    "estimator = RandomForestClassifier(random_state=42, n_jobs=-1)\n"
                )
            else:
                code += "from sklearn.tree import DecisionTreeClassifier\n"
                code += "estimator = DecisionTreeClassifier(random_state=42)\n"
        else:
            if model_type == "random_forest":
                code += "from sklearn.ensemble import RandomForestRegressor\n"
                code += (
                    "estimator = RandomForestRegressor(random_state=42, n_jobs=-1)\n"
                )
            else:
                code += "from sklearn.tree import DecisionTreeRegressor\n"
                code += "estimator = DecisionTreeRegressor(random_state=42)\n"

        code += "param_grid = {\n"
        code += f"    'max_depth': {max_depth_values.split(',')},\n"
        code += f"    'min_samples_split': {min_samples_split_values.split(',')},\n"
        code += f"    'min_samples_leaf': {min_samples_leaf_values.split(',')},\n"
        if model_type == "random_forest":
            code += f"    'n_estimators': {n_estimators_values.split(',')},\n"
        code += "}\n"
        code += f"grid = GridSearchCV(estimator, param_grid=param_grid, scoring='{scoring}', cv={cv_folds}, n_jobs=-1)\n"
        code += "grid.fit(X_train, y_train)\n"
        code += "print('Best params:', grid.best_params_)\n"
        code += "print('Best score:', grid.best_score_)"

        return code
