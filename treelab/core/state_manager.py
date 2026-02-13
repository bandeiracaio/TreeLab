"""State management for TreeLab - handles DataFrame states, history, and checkpoints."""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import copy


class ActionRecord:
    """Record of an executed action."""

    def __init__(self, action_name: str, params: Dict[str, Any], timestamp: datetime):
        self.action_name = action_name
        self.params = params
        self.timestamp = timestamp

    def __repr__(self):
        return (
            f"ActionRecord({self.action_name}, {self.timestamp.strftime('%H:%M:%S')})"
        )


class StateManager:
    """
    Manages the state of the data exploration session.

    Responsibilities:
    - Store current DataFrame state
    - Maintain history of all actions
    - Handle checkpoints (named save points)
    - Track train/test splits separately
    - Manage fitted models
    """

    def __init__(self, initial_df: pd.DataFrame):
        """
        Initialize StateManager with initial DataFrame.

        Args:
            initial_df: The starting DataFrame for the session
        """
        self.df = initial_df.copy()
        self._original_df = initial_df.copy()  # Keep original for reference

        # History tracking
        self.history: List[ActionRecord] = []
        self.checkpoints: Dict[str, int] = {}  # name -> history_index

        # Train/test split tracking
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.target_column: Optional[str] = None

        # Model tracking
        self.current_model: Optional[Any] = None
        self.model_metadata: Dict[str, Any] = {}

        # Mode tracking
        self.mode: str = "transformation"  # 'transformation' or 'modeling'

        # Session info
        self.session_start = datetime.now()

    def apply_action(
        self, action_name: str, params: Dict[str, Any], result: Dict[str, Any]
    ) -> bool:
        """
        Apply an action and update state.

        Args:
            action_name: Name of the action
            params: Parameters used for the action
            result: Result dictionary from action.execute()

        Returns:
            True if successful, False otherwise
        """
        try:
            # Update DataFrames
            if result.get("df") is not None:
                self.df = result["df"].copy()

            if result.get("train_df") is not None:
                self.train_df = result["train_df"].copy()

            if result.get("test_df") is not None:
                self.test_df = result["test_df"].copy()

            # Update model
            if result.get("model") is not None:
                self.current_model = result["model"]
                self.model_metadata = result.get("metadata", {})

            # Update target column if specified
            if "target_column" in result:
                self.target_column = result["target_column"]

            # Record action in history
            record = ActionRecord(action_name, params, datetime.now())
            self.history.append(record)

            return True

        except Exception as e:
            print(f"Error applying action: {e}")
            return False

    def create_checkpoint(self, name: str) -> bool:
        """
        Create a checkpoint at the current state.

        Args:
            name: Name for the checkpoint

        Returns:
            True if successful, False if name already exists
        """
        if name in self.checkpoints:
            return False

        self.checkpoints[name] = len(self.history)
        return True

    def get_checkpoints(self) -> Dict[str, int]:
        """Get all checkpoints."""
        return self.checkpoints.copy()

    def revert_to_checkpoint(self, name: str) -> bool:
        """
        Revert to a named checkpoint.

        This will reset the state to the checkpoint and remove all actions after it.

        Args:
            name: Name of the checkpoint

        Returns:
            True if successful, False if checkpoint doesn't exist
        """
        if name not in self.checkpoints:
            return False

        target_index = self.checkpoints[name]

        # Remove actions after checkpoint
        self.history = self.history[:target_index]

        # Remove checkpoints that are after this point
        checkpoints_to_remove = [
            cp for cp, idx in self.checkpoints.items() if idx > target_index
        ]
        for cp in checkpoints_to_remove:
            del self.checkpoints[cp]

        # TODO: In full implementation, we'd replay all actions from original_df
        # For MVP, we'll note this as a limitation
        print(f"Reverted to checkpoint: {name}")
        print(
            "Note: Full state replay not implemented in MVP. Please restart session for complex rollbacks."
        )

        return True

    def get_history(self) -> List[ActionRecord]:
        """Get the action history."""
        return self.history.copy()

    def get_history_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of the action history for display.

        Returns:
            List of dictionaries with action info
        """
        summary = []
        for idx, record in enumerate(self.history):
            checkpoint_names = [
                name for name, cp_idx in self.checkpoints.items() if cp_idx == idx
            ]

            summary.append(
                {
                    "index": idx + 1,
                    "action": record.action_name,
                    "timestamp": record.timestamp.strftime("%H:%M:%S"),
                    "checkpoints": checkpoint_names,
                }
            )

        return summary

    def switch_mode(self, new_mode: str) -> bool:
        """
        Switch between transformation and modeling modes.

        Args:
            new_mode: 'transformation' or 'modeling'

        Returns:
            True if successful
        """
        if new_mode not in ["transformation", "modeling"]:
            return False

        self.mode = new_mode
        return True

    def is_split_done(self) -> bool:
        """Check if train/test split has been performed."""
        return self.train_df is not None and self.test_df is not None

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current state.

        Returns:
            Dictionary with state information
        """
        info = {
            "mode": self.mode,
            "df_shape": self.df.shape if self.df is not None else None,
            "train_shape": self.train_df.shape if self.train_df is not None else None,
            "test_shape": self.test_df.shape if self.test_df is not None else None,
            "target_column": self.target_column,
            "has_model": self.current_model is not None,
            "num_actions": len(self.history),
            "num_checkpoints": len(self.checkpoints),
            "session_duration": (datetime.now() - self.session_start).seconds,
        }

        return info

    def reset(self):
        """Reset to original DataFrame state."""
        self.df = self._original_df.copy()
        self.history = []
        self.checkpoints = {}
        self.train_df = None
        self.test_df = None
        self.target_column = None
        self.current_model = None
        self.model_metadata = {}
        self.mode = "transformation"
