"""Base class for all TreeLab actions."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from dataclasses import dataclass


@dataclass
class Parameter:
    """Defines a parameter for an action."""

    name: str
    label: str
    type: str  # 'columns', 'column', 'numeric', 'text', 'select', 'boolean'
    required: bool = True
    default: Any = None
    options: Optional[List[Any]] = None  # For 'select' type
    description: str = ""


class Action(ABC):
    """Base class for all actions in TreeLab."""

    name: str = "BaseAction"
    description: str = "Base action class"
    mode: str = "transformation"  # 'transformation' or 'modeling'
    category: str = "General"  # Sub-category for organization

    @abstractmethod
    def get_parameters(self) -> List[Parameter]:
        """
        Return list of parameters this action needs.

        Returns:
            List of Parameter objects defining the action's inputs.
        """
        pass

    @abstractmethod
    def validate(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[bool, str]:
        """
        Validate parameters before execution.

        Args:
            df: Current DataFrame
            params: Dictionary of parameter values
            train_df: Training DataFrame (if split has been done)
            test_df: Test DataFrame (if split has been done)

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    def execute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute the action.

        Args:
            df: Current DataFrame
            params: Dictionary of parameter values
            train_df: Training DataFrame (if applicable)
            test_df: Test DataFrame (if applicable)

        Returns:
            Dictionary with keys:
                - 'df': Modified DataFrame (or None if not applicable)
                - 'train_df': Modified training DataFrame (or None)
                - 'test_df': Modified test DataFrame (or None)
                - 'model': Trained model object (for modeling actions)
                - 'metadata': Any additional metadata from the action
        """
        pass

    @abstractmethod
    def suggest_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Smart suggestion of which columns to use for this action.

        Args:
            df: Current DataFrame

        Returns:
            List of suggested column names
        """
        pass

    def get_visualization_data(
        self, df_before: pd.DataFrame, df_after: pd.DataFrame, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Return data for visualization of action effect.

        Args:
            df_before: DataFrame before action
            df_after: DataFrame after action
            params: Parameters used for action

        Returns:
            Dictionary with visualization data or None
        """
        return None

    @abstractmethod
    def to_python_code(self, params: Dict[str, Any]) -> str:
        """
        Generate Python code for this action.

        Args:
            params: Dictionary of parameter values

        Returns:
            String containing Python code
        """
        pass

    def to_bigquery_sql(
        self, params: Dict[str, Any], table_name: str = "input_table"
    ) -> str:
        """
        Generate BigQuery SQL for this action.

        Args:
            params: Dictionary of parameter values
            table_name: Name of the input table in SQL

        Returns:
            String containing BigQuery SQL
        """
        return f"-- Action {self.name} not supported in BigQuery SQL\nSELECT * FROM {table_name}"

    def __repr__(self):
        return f"{self.__class__.__name__}()"
