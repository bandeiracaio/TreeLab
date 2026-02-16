"""Tests for StateManager."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestStateManager:
    """Tests for StateManager class."""

    def test_initialization(self, sample_dataframe):
        """Test StateManager initializes correctly."""
        from treelab.core.state_manager import StateManager

        sm = StateManager(sample_dataframe)

        assert sm.df is not None
        assert sm.df.shape == sample_dataframe.shape
        assert len(sm.history) == 0
        assert len(sm.checkpoints) == 0

    def test_apply_action_updates_df(self, state_manager):
        """Test that apply_action updates the DataFrame."""
        result = {
            "df": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            "train_df": None,
            "test_df": None,
            "model": None,
            "metadata": {},
        }

        success = state_manager.apply_action("test_action", {"param": "value"}, result)

        assert success is True
        assert state_manager.df.shape == (3, 2)
        assert len(state_manager.history) == 1

    def test_apply_action_stores_history(self, state_manager):
        """Test that action history is recorded."""
        result = {
            "df": state_manager.df.copy(),
            "train_df": None,
            "test_df": None,
            "model": None,
            "metadata": {},
        }

        state_manager.apply_action("DropColumns", {"columns": ["age"]}, result)

        assert len(state_manager.history) == 1
        assert state_manager.history[0].action_name == "DropColumns"

    def test_create_checkpoint(self, state_manager):
        """Test checkpoint creation."""
        result = {
            "df": state_manager.df.copy(),
            "train_df": None,
            "test_df": None,
            "model": None,
            "metadata": {},
        }
        state_manager.apply_action("action1", {}, result)

        success = state_manager.create_checkpoint("test_checkpoint")

        assert success is True
        assert "test_checkpoint" in state_manager.checkpoints

    def test_create_duplicate_checkpoint_fails(self, state_manager):
        """Test that duplicate checkpoint names fail."""
        state_manager.create_checkpoint("dup_checkpoint")

        success = state_manager.create_checkpoint("dup_checkpoint")

        assert success is False

    def test_revert_to_checkpoint(self, state_manager):
        """Test checkpoint revert restores state."""
        df_before = state_manager.df.copy()

        result1 = {
            "df": pd.DataFrame({"a": [1, 2]}),
            "train_df": None,
            "test_df": None,
            "model": None,
            "metadata": {},
        }
        result2 = {
            "df": pd.DataFrame({"a": [1, 2, 3]}),
            "train_df": None,
            "test_df": None,
            "model": None,
            "metadata": {},
        }

        state_manager.apply_action("action1", {}, result1)
        state_manager.create_checkpoint("cp1")
        state_manager.apply_action("action2", {}, result2)

        assert len(state_manager.history) == 2

        success = state_manager.revert_to_checkpoint("cp1")

        assert success is True
        assert len(state_manager.history) == 1

    def test_switch_mode(self, state_manager):
        """Test mode switching."""
        assert state_manager.mode == "transformation"

        success = state_manager.switch_mode("modeling")

        assert success is True
        assert state_manager.mode == "modeling"

    def test_switch_invalid_mode_fails(self, state_manager):
        """Test that invalid mode fails."""
        success = state_manager.switch_mode("invalid_mode")

        assert success is False

    def test_is_split_done(self, state_manager):
        """Test train/test split detection."""
        assert state_manager.is_split_done() is False

        state_manager.train_df = pd.DataFrame({"a": [1, 2, 3]})
        state_manager.test_df = pd.DataFrame({"a": [4, 5]})

        assert state_manager.is_split_done() is True

    def test_reset(self, state_manager):
        """Test reset functionality."""
        state_manager.history.append("fake_record")
        state_manager.checkpoints = {"cp1": 0}
        state_manager.train_df = pd.DataFrame({"a": [1]})

        state_manager.reset()

        assert len(state_manager.history) == 0
        assert len(state_manager.checkpoints) == 0
        assert state_manager.train_df is None

    def test_get_state_info(self, state_manager):
        """Test state info retrieval."""
        info = state_manager.get_state_info()

        assert "mode" in info
        assert "df_shape" in info
        assert info["mode"] == "transformation"
