"""Tests for ActionRegistry."""

import pytest


class TestActionRegistry:
    """Tests for ActionRegistry class."""

    def test_register_transformation(self, action_registry):
        """Test registering transformation actions."""
        from treelab.core.action_registry import ActionRegistry
        from treelab.actions.transformations import DropColumnsAction

        assert "DropColumnsAction" in str(ActionRegistry._transformation_actions)

    def test_register_modeling(self, action_registry):
        """Test registering modeling actions."""
        from treelab.core.action_registry import ActionRegistry

        actions = ActionRegistry.get_action_names("modeling")
        assert "DecisionTreeClassifier" in actions
        assert "RandomForestClassifier" in actions

    def test_get_actions_for_mode(self, action_registry):
        """Test getting actions by mode."""
        from treelab.core.action_registry import ActionRegistry

        trans_actions = ActionRegistry.get_actions_for_mode("transformation")
        assert isinstance(trans_actions, dict)
        assert len(trans_actions) > 0

    def test_get_action_class(self, action_registry):
        """Test getting action class by name."""
        from treelab.core.action_registry import ActionRegistry

        action_class = ActionRegistry.get_action_class("DropColumns")
        assert action_class is not None

    def test_get_action_class_not_found(self, action_registry):
        """Test that KeyError is raised for unknown actions."""
        from treelab.core.action_registry import ActionRegistry

        with pytest.raises(KeyError):
            ActionRegistry.get_action_class("NonExistentAction")

    def test_clear(self):
        """Test clearing the registry."""
        from treelab.core.action_registry import ActionRegistry

        ActionRegistry.clear()

        trans = ActionRegistry.get_actions_for_mode("transformation")
        modeling = ActionRegistry.get_actions_for_mode("modeling")

        assert len(trans) == 0
        assert len(modeling) == 0
