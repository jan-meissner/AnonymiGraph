from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractAnonymizer(ABC):
    @abstractmethod
    def anonymize(self, graph):
        """Anonymize and return the anonymized graph."""
