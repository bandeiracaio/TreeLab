"""Tests for TreeLab application."""

import pytest
import pandas as pd
from treelab.app import TreeLab, _validate_dataframe


class TestTreeLab:
    """Tests for TreeLab application class."""

    def test_validate_dataframe_valid(self, sample_dataframe):
        """Test validation passes for valid DataFrame."""
        _validate_dataframe(sample_dataframe)

    def test_validate_dataframe_not_dataframe(self):
        """Test validation fails for non-DataFrame."""
        with pytest.raises(TypeError):
            _validate_dataframe("not a dataframe")

    def test_validate_dataframe_empty(self):
        """Test validation fails for empty DataFrame."""
        with pytest.raises(ValueError, match="empty"):
            _validate_dataframe(pd.DataFrame())

    def test_validate_dataframe_single_row(self):
        """Test validation fails for single row DataFrame."""
        with pytest.raises(ValueError, match="at least 2"):
            _validate_dataframe(pd.DataFrame({"a": [1]}))

    def test_treelab_initialization_with_dataframe(self, sample_dataframe):
        """Test TreeLab initializes with provided DataFrame."""
        app = TreeLab(df=sample_dataframe)

        assert app.state_manager is not None
        assert app.state_manager.df is not None

    def test_treelab_sample_frac_validation(self, sample_dataframe):
        """Test sample_frac validation."""
        with pytest.raises(ValueError):
            TreeLab(df=sample_dataframe, sample_frac=1.5)

        with pytest.raises(ValueError):
            TreeLab(df=sample_dataframe, sample_frac=0)

    def test_treelab_get_state(self, sample_dataframe):
        """Test get_state method."""
        app = TreeLab(df=sample_dataframe)

        state = app.get_state()

        assert "mode" in state
        assert "df_shape" in state
