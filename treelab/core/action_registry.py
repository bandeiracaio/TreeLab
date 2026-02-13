"""Registry of all available actions in TreeLab."""

from typing import Dict, List, Type
from treelab.actions.base import Action


class ActionRegistry:
    """
    Central registry for all available actions.

    Actions are organized by mode (transformation or modeling).
    """

    _transformation_actions: Dict[str, Type[Action]] = {}
    _modeling_actions: Dict[str, Type[Action]] = {}

    @classmethod
    def register_transformation(cls, action_class: Type[Action]):
        """
        Register a transformation action.

        Args:
            action_class: The action class to register
        """
        cls._transformation_actions[action_class.name] = action_class

    @classmethod
    def register_modeling(cls, action_class: Type[Action]):
        """
        Register a modeling action.

        Args:
            action_class: The action class to register
        """
        cls._modeling_actions[action_class.name] = action_class

    @classmethod
    def get_actions_for_mode(cls, mode: str) -> Dict[str, Type[Action]]:
        """
        Get all actions available for a specific mode.

        Args:
            mode: 'transformation' or 'modeling'

        Returns:
            Dictionary mapping action names to action classes
        """
        if mode == "transformation":
            return cls._transformation_actions.copy()
        elif mode == "modeling":
            return cls._modeling_actions.copy()
        else:
            return {}

    @classmethod
    def get_action_names(cls, mode: str) -> List[str]:
        """
        Get list of action names for a mode.

        Args:
            mode: 'transformation' or 'modeling'

        Returns:
            List of action names
        """
        return list(cls.get_actions_for_mode(mode).keys())

    @classmethod
    def get_action_class(cls, name: str) -> Type[Action]:
        """
        Get an action class by name.

        Args:
            name: Name of the action

        Returns:
            The action class

        Raises:
            KeyError: If action not found
        """
        # Try transformation actions first
        if name in cls._transformation_actions:
            return cls._transformation_actions[name]

        # Try modeling actions
        if name in cls._modeling_actions:
            return cls._modeling_actions[name]

        raise KeyError(f"Action '{name}' not found in registry")

    @classmethod
    def get_all_actions(cls) -> Dict[str, Type[Action]]:
        """Get all registered actions."""
        all_actions = {}
        all_actions.update(cls._transformation_actions)
        all_actions.update(cls._modeling_actions)
        return all_actions

    @classmethod
    def clear(cls):
        """Clear all registered actions (useful for testing)."""
        cls._transformation_actions = {}
        cls._modeling_actions = {}
