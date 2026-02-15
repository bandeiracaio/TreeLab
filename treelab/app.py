"""Main TreeLab application."""

import pandas as pd
from typing import Optional
from pathlib import Path

from treelab.core.state_manager import StateManager
from treelab.core.action_registry import ActionRegistry
from treelab.core.logger import SessionLogger

# Import and register all actions
from treelab.actions.transformations import (
    DropColumnsAction,
    SimpleImputerAction,
    StandardScalerAction,
    MinMaxScalerAction,
    OneHotEncoderAction,
    LabelEncoderAction,
    TrainTestSplitAction,
    PCAAction,
    PolynomialFeaturesAction,
    RFEAction,
)

from treelab.actions.modeling import (
    DecisionTreeClassifierAction,
    RandomForestClassifierAction,
    DecisionTreeRegressorAction,
    RandomForestRegressorAction,
    FeatureImportanceAction,
    TuneHyperparametersAction,
    SHAPSummaryAction,
    BinningScorecardAction,
)


def register_all_actions():
    """Register all available actions."""
    # Transformation actions
    ActionRegistry.register_transformation(DropColumnsAction)
    ActionRegistry.register_transformation(SimpleImputerAction)
    ActionRegistry.register_transformation(StandardScalerAction)
    ActionRegistry.register_transformation(MinMaxScalerAction)
    ActionRegistry.register_transformation(OneHotEncoderAction)
    ActionRegistry.register_transformation(LabelEncoderAction)
    ActionRegistry.register_transformation(TrainTestSplitAction)
    ActionRegistry.register_transformation(PCAAction)
    ActionRegistry.register_transformation(PolynomialFeaturesAction)
    ActionRegistry.register_transformation(RFEAction)

    # Modeling actions
    ActionRegistry.register_modeling(DecisionTreeClassifierAction)
    ActionRegistry.register_modeling(RandomForestClassifierAction)
    ActionRegistry.register_modeling(DecisionTreeRegressorAction)
    ActionRegistry.register_modeling(RandomForestRegressorAction)
    ActionRegistry.register_modeling(FeatureImportanceAction)
    ActionRegistry.register_modeling(TuneHyperparametersAction)
    ActionRegistry.register_modeling(SHAPSummaryAction)
    ActionRegistry.register_modeling(BinningScorecardAction)


class TreeLab:
    """
    Main TreeLab application class.

    Usage:
        # With your own DataFrame
        app = TreeLab(df)
        app.run()

        # With default Titanic dataset
        app = TreeLab()
        app.run()
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        sample_frac: Optional[float] = None,
        sample_seed: Optional[int] = 42,
    ):
        """
        Initialize TreeLab application.

        Args:
            df: Optional DataFrame to use. If None, loads default Titanic dataset.
        """
        # Load data
        if df is None:
            df = self._load_default_dataset()

        if sample_frac is not None:
            if not (0 < sample_frac <= 1):
                raise ValueError("sample_frac must be between 0 and 1")
            df = df.sample(frac=sample_frac, random_state=sample_seed).reset_index(
                drop=True
            )

        # Initialize core components
        self.state_manager = StateManager(df)
        self.logger = SessionLogger(self.state_manager)

        # Register all actions
        register_all_actions()

        print(f"TreeLab initialized with {df.shape[0]} rows and {df.shape[1]} columns")

    def _load_default_dataset(self) -> pd.DataFrame:
        """Load the default Titanic dataset."""
        data_path = Path(__file__).parent.parent / "data" / "titanic.csv"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Default dataset not found at {data_path}. "
                "Please provide a DataFrame or ensure titanic.csv exists in data/ folder."
            )

        df = pd.read_csv(data_path)
        print(f"Loaded default Titanic dataset from {data_path}")
        return df

    def run(self, host="127.0.0.1", port=8050, debug=True):
        """
        Launch the Dash web application.

        Args:
            host: Host address to run on
            port: Port to run on
            debug: Enable debug mode
        """
        from treelab.ui.layout import create_app

        app = create_app(self.state_manager, self.logger)

        print(f"\n{'=' * 60}")
        print(f"TreeLab is starting...")
        print(f"{'=' * 60}")
        print(
            f"Dataset: {self.state_manager.df.shape[0]} rows x {self.state_manager.df.shape[1]} columns"
        )
        print(f"Open your browser to: http://{host}:{port}")
        print(f"{'=' * 60}\n")

        app.run(host=host, port=port, debug=debug)

    def get_state(self):
        """Get current state information."""
        return self.state_manager.get_state_info()

    def export_script(self, filepath: Optional[str] = None):
        """
        Export session as Python script.

        Args:
            filepath: Path to save script. If None, returns script as string.
        """
        if filepath is not None:
            self.logger.save_script(filepath)
            print(f"Script saved to {filepath}")
        else:
            return self.logger.generate_script()


# For backwards compatibility and quick testing
def launch(df: Optional[pd.DataFrame] = None, port=8050):
    """Quick launch function."""
    app = TreeLab(df)
    app.run(port=port)


if __name__ == "__main__":
    # Quick test
    app = TreeLab()
    app.run()
