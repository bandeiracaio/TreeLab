"""Base classes for model actions to reduce code duplication."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Type, cast, Hashable
from abc import ABC, abstractmethod

from treelab.actions.base import Action, Parameter


class BaseModelAction(Action, ABC):
    """Base class for model training actions."""

    model_family: str = ""
    task_type: str = ""

    @property
    @abstractmethod
    def sklearn_class(self) -> Type:
        """Return the sklearn class for this model."""
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters."""
        pass

    @property
    def metric_names(self) -> List[str]:
        """Return list of metric names to compute."""
        return []

    @property
    def higher_is_better(self) -> Dict[str, bool]:
        """Return which metrics are better when higher."""
        return {}

    _default_n_jobs: Optional[int] = None

    def get_parameters(self) -> List[Parameter]:
        return []

    def _get_target_column(
        self, train_df: pd.DataFrame, params: Dict[str, Any] = None
    ) -> Hashable:
        """Get target column from params (passed from state) or fall back to last column."""
        if params and "_target_column" in params:
            return params["_target_column"]
        return cast(Hashable, train_df.columns[-1])

    def _prepare_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: Hashable,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare X_train, y_train, X_test, y_test."""
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        return X_train, y_train, X_test, y_test

    def _compute_classification_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        average_param = "binary" if len(np.unique(y_train)) == 2 else "weighted"

        metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "test_precision": float(
                precision_score(
                    y_test, y_pred_test, average=average_param, zero_division=0
                )
            ),
            "test_recall": float(
                recall_score(
                    y_test, y_pred_test, average=average_param, zero_division=0
                )
            ),
            "test_f1": float(
                f1_score(y_test, y_pred_test, average=average_param, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }
        return metrics

    def _compute_regression_metrics(
        self,
        y_train: np.ndarray,
        y_pred_train: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute regression metrics."""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        return {
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "test_r2": float(r2_score(y_test, y_pred_test)),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "test_rmse": float(mean_squared_error(y_test, y_pred_test) ** 0.5),
            "y_test": y_test.tolist(),
            "y_pred": y_pred_test.tolist(),
        }

    def _create_base_metadata(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_col: Hashable,
        train_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create base metadata dictionary."""
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        metadata = {
            "model_type": self.name,
            "task": self.task_type,
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "target_column": target_col,
        }

        if hasattr(model, "classes_"):
            metadata["classes"] = list(model.classes_)

        if hasattr(model, "n_estimators"):
            metadata["n_estimators"] = model.n_estimators

        if hasattr(model, "get_depth"):
            metadata["tree_depth"] = model.get_depth()
            metadata["n_leaves"] = model.get_n_leaves()

        metadata.update(train_metrics)
        return metadata

    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute model training."""
        if train_df is None or test_df is None:
            raise ValueError("Train/test split must be completed before modeling")

        target_col = self._get_target_column(train_df, params)
        X_train, y_train, X_test, y_test = self._prepare_data(
            train_df, test_df, target_col
        )

        model_params = {k: params.get(k, v) for k, v in self.default_params.items()}
        if "random_state" not in model_params:
            model_params["random_state"] = 42
        if (
            "n_jobs" not in model_params
            and hasattr(self, "_default_n_jobs")
            and self._default_n_jobs is not None
        ):
            model_params["n_jobs"] = self._default_n_jobs

        model = self.sklearn_class(**model_params)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if self.task_type == "classification":
            train_metrics = self._compute_classification_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )
        else:
            train_metrics = self._compute_regression_metrics(
                y_train, y_pred_train, y_test, y_pred_test
            )

        metadata = self._create_base_metadata(
            model, X_train, y_train, X_test, y_test, target_col, train_metrics
        )

        return {
            "df": df,
            "train_df": train_df,
            "test_df": test_df,
            "model": model,
            "metadata": metadata,
        }


class ClassificationModelAction(BaseModelAction):
    """Base class for classification models."""

    task_type = "classification"

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        return f"# {self.name} - code generation not implemented"


class RegressionModelAction(BaseModelAction):
    """Base class for regression models."""

    task_type = "regression"

    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        return []

    def to_python_code(self, params: Dict[str, Any]) -> str:
        return f"# {self.name} - code generation not implemented"
