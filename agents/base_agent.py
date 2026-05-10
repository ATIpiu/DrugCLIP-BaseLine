"""Base agent class — all sub-agents inherit this interface."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Standard interface for all agents.

    The `context` dict is the universal message bus shared across the main loop.
    Each agent reads what it needs and appends its decisions/results.
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(self, context: dict) -> dict:
        """Execute agent logic.

        Args:
            context: shared dict containing current iteration state, history, config, etc.
                     See MainAgent._init_context() for full schema.

        Returns:
            {
                "status": "ok" | "error",
                "data": {...},           # agent-specific structured output
                "summary": "one-line",
                "error": "message if status=error"
            }
        """
        pass
