"""Logger for generating reproducible Python scripts from TreeLab sessions."""

from typing import List, Dict, Any
from datetime import datetime
from treelab.core.state_manager import StateManager, ActionRecord
from treelab.core.action_registry import ActionRegistry


class SessionLogger:
    """
    Generates Python scripts from action history.

    The generated script is a standalone Python file that reproduces
    all transformations and model fitting steps.
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize logger with state manager.

        Args:
            state_manager: The StateManager instance tracking the session
        """
        self.state_manager = state_manager

    def generate_script(self) -> str:
        """
        Generate a complete Python script from the session history.

        Returns:
            String containing the Python script
        """
        lines = []

        # Header
        lines.append(self._generate_header())
        lines.append("")

        # Imports
        lines.append(self._generate_imports())
        lines.append("")

        # Data loading placeholder
        lines.append(self._generate_data_loading())
        lines.append("")

        # Actions
        checkpoint_indices = {
            idx: names for names, idx in self.state_manager.checkpoints.items()
        }

        for idx, record in enumerate(self.state_manager.history):
            # Add checkpoint marker if exists
            if idx in checkpoint_indices:
                lines.append(f"# [*] Checkpoint: {checkpoint_indices[idx]}")
                lines.append(f"print('=== Checkpoint: {checkpoint_indices[idx]} ===')")
                lines.append(f"print(f'DataFrame shape: {{df.shape}}')")
                lines.append("")

            # Add action code
            action_code = self._generate_action_code(record, idx + 1)
            lines.append(action_code)
            lines.append("")

        # Footer
        lines.append(self._generate_footer())

        return "\n".join(lines)

    def _generate_header(self) -> str:
        """Generate script header with metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = (datetime.now() - self.state_manager.session_start).seconds

        header = f"""# TreeLab Session Script
# Generated: {timestamp}
# Session Duration: {duration} seconds
# Total Actions: {len(self.state_manager.history)}
# Mode: {self.state_manager.mode}
"""
        return header

    def _generate_imports(self) -> str:
        """Generate import statements."""
        imports = """# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
"""
        return imports

    def _generate_data_loading(self) -> str:
        """Generate data loading code."""
        loading = """# Load Data
# Replace the line below with your actual data loading code
# df = pd.read_csv('your_data.csv')
df = pd.read_csv('data/titanic.csv')  # Default Titanic dataset
print(f"Initial data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
"""
        return loading

    def _generate_action_code(self, record: ActionRecord, action_number: int) -> str:
        """
        Generate code for a specific action.

        Args:
            record: ActionRecord to generate code for
            action_number: Sequential number of the action

        Returns:
            Python code string
        """
        try:
            action_class = ActionRegistry.get_action_class(record.action_name)
            action_instance = action_class()

            code = f"# Action {action_number}: {record.action_name}\n"
            code += action_instance.to_python_code(record.params)

            return code

        except Exception as e:
            return f"# Action {action_number}: {record.action_name} (Error generating code: {e})"

    def _generate_footer(self) -> str:
        """Generate script footer."""
        footer = """# End of TreeLab Session Script
print("\\n=== Script Execution Complete ===")
"""
        return footer

    def save_script(self, filepath: str) -> bool:
        """
        Save the generated script to a file.

        Args:
            filepath: Path where to save the script

        Returns:
            True if successful
        """
        try:
            script = self.generate_script()
            with open(filepath, "w") as f:
                f.write(script)
            return True
        except Exception as e:
            print(f"Error saving script: {e}")
            return False
