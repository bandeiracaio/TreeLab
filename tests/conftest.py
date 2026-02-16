"""Shared pytest fixtures for TreeLab tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "salary": [
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
                120000,
                130000,
                140000,
            ],
            "department": [
                "Sales",
                "Engineering",
                "Sales",
                "Engineering",
                "HR",
                "Sales",
                "HR",
                "Engineering",
                "Sales",
                "HR",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def titanic_dataframe():
    """Create a sample Titanic-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 2],
            "Name": ["Smith", "Johnson", "Williams", "Brown", "Jones"],
            "Sex": ["male", "female", "female", "male", "female"],
            "Age": [22, 38, 26, 35, 27],
            "SibSp": [1, 1, 0, 0, 1],
            "Parch": [0, 0, 0, 0, 0],
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05],
            "Embarked": ["S", "C", "S", "S", "C"],
        }
    )


@pytest.fixture
def dataframe_with_missing():
    """Create a DataFrame with missing values."""
    df = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": ["x", "y", None, "z", "w"],
            "d": [np.nan, np.nan, 3, 4, 5],
        }
    )
    return df


@pytest.fixture
def state_manager(sample_dataframe):
    """Create a StateManager with sample data."""
    from treelab.core.state_manager import StateManager

    return StateManager(sample_dataframe)


@pytest.fixture
def action_registry():
    """Create and populate an ActionRegistry."""
    from treelab.core.action_registry import ActionRegistry
    from treelab.actions.transformations import (
        DropColumnsAction,
        SimpleImputerAction,
        TrainTestSplitAction,
    )
    from treelab.actions.modeling import (
        DecisionTreeClassifierAction,
        RandomForestClassifierAction,
    )

    ActionRegistry.clear()
    ActionRegistry.register_transformation(DropColumnsAction)
    ActionRegistry.register_transformation(SimpleImputerAction)
    ActionRegistry.register_transformation(TrainTestSplitAction)
    ActionRegistry.register_modeling(DecisionTreeClassifierAction)
    ActionRegistry.register_modeling(RandomForestClassifierAction)

    yield ActionRegistry

    ActionRegistry.clear()


@pytest.fixture
def train_test_split_dataframe(sample_dataframe):
    """Create train/test split DataFrames."""
    df = sample_dataframe
    train_df = df.iloc[:7].copy()
    test_df = df.iloc[7:].copy()
    return train_df, test_df
