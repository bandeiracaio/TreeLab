"""SHAP-based model explanation actions."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, cast, Hashable
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
import shap
from treelab.actions.base import Action, Parameter


class SHAPSummaryAction(Action):
    """Fit a model and compute SHAP summary importance."""

    name = "SHAPSummary"
    description = "Compute SHAP feature importance using a random forest"
    mode = "modeling"
    category = "Model Explainability"

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
                name="max_samples",
                label="Max Samples",
                type="numeric",
                required=False,
                default=500,
                description="Max samples to use for SHAP computation",
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
            return False, "Must perform train/test split before running SHAP"

        n_estimators = params.get("n_estimators", 200)
        if n_estimators < 1:
            return False, "n_estimators must be >= 1"

        max_samples = params.get("max_samples", 500)
        if max_samples < 50:
            return False, "max_samples must be >= 50"

        return True, ""

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before SHAP")

        task = params.get("task", "classification")
        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = int(params.get("min_samples_split", 2))
        min_samples_leaf = int(params.get("min_samples_leaf", 1))
        max_samples = int(params.get("max_samples", 500))

        target_col = cast(Hashable, train_df.columns[-1])
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        if task == "classification":
            model = SKRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )
        else:
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

        sample_df = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)

        if isinstance(shap_values, list):
            stacked = np.stack([np.abs(values) for values in shap_values], axis=0)
            mean_abs = stacked.mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        if hasattr(mean_abs, "ndim") and mean_abs.ndim > 1:
            mean_abs = mean_abs.mean(axis=-1)

        shap_importance = dict(zip(sample_df.columns, mean_abs.tolist()))

        if isinstance(shap_values, list):
            shap_values_matrix = np.array(shap_values[0])
        else:
            shap_values_matrix = np.array(shap_values)

        metadata: Dict[str, Any] = {
            "model_type": "SHAPRandomForest",
            "task": task,
            "target_column": target_col,
            "shap_importance": {k: float(v) for k, v in shap_importance.items()},
            "shap_samples": int(sample_df.shape[0]),
            "shap_values": shap_values_matrix.tolist(),
            "shap_feature_values": sample_df.values.tolist(),
            "shap_feature_names": list(sample_df.columns),
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
                            zero_division="warn",
                        )
                    ),
                    "test_recall": float(
                        recall_score(
                            y_test,
                            y_pred_test,
                            average=average_param,
                            zero_division="warn",
                        )
                    ),
                    "test_f1": float(
                        f1_score(
                            y_test,
                            y_pred_test,
                            average=average_param,
                            zero_division="warn",
                        )
                    ),
                    "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
                    "classes": list(model.classes_),
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
        task = params.get("task", "classification")
        n_estimators = params.get("n_estimators", 200)
        max_depth = params.get("max_depth", 10)
        if max_depth == 0:
            max_depth = None
        min_samples_split = params.get("min_samples_split", 2)
        min_samples_leaf = params.get("min_samples_leaf", 1)
        max_samples = params.get("max_samples", 500)

        code = "# SHAP summary with Random Forest\n"
        if task == "classification":
            code += "from sklearn.ensemble import RandomForestClassifier\n"
            code += "model = RandomForestClassifier(\n"
        else:
            code += "from sklearn.ensemble import RandomForestRegressor\n"
            code += "model = RandomForestRegressor(\n"
        code += f"    n_estimators={n_estimators},\n"
        code += f"    max_depth={max_depth},\n"
        code += f"    min_samples_split={min_samples_split},\n"
        code += f"    min_samples_leaf={min_samples_leaf},\n"
        code += "    random_state=42,\n"
        code += "    n_jobs=-1\n"
        code += ")\n"
        code += "model.fit(X_train, y_train)\n"
        code += "import shap\n"
        code += f"sample_df = X_test.sample(n=min({max_samples}, len(X_test)), random_state=42)\n"
        code += "explainer = shap.TreeExplainer(model)\n"
        code += "shap_values = explainer.shap_values(sample_df)\n"
        code += "print('Computed SHAP values for', len(sample_df), 'rows')"

        return code
