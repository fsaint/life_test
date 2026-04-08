"""Abstract base class for all LLM + Memory adapters."""
from __future__ import annotations
from abc import ABC, abstractmethod
from runner.models import Event


class BaseAdapter(ABC):
    """
    Every LLM + Memory combination implements this interface.
    The runner only calls these four methods.
    """

    @property
    @abstractmethod
    def system_id(self) -> str:
        """Unique identifier matching systems.yaml."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all memory and conversation state for a new situation."""
        ...

    @abstractmethod
    def ingest_initial_state(self, content: str) -> dict:
        """
        Feed the plain-text initial_state.md into the system.
        Returns a log entry dict with at minimum: {tokens_used, latency_ms}.
        """
        ...

    @abstractmethod
    def ingest_event(self, event: Event) -> dict:
        """
        Feed one event into the system.
        Returns a log entry dict with at minimum:
          {event_id, event_type, tokens_used, latency_ms, status}
        """
        ...

    @abstractmethod
    def ask(self, question_text: str) -> str:
        """
        Query the system with a natural language question.
        Returns the raw string answer.
        """
        ...
