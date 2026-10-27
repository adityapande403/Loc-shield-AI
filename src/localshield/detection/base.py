"""Base detector interface and candidate construction utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set

from localshield.schemas import PIICandidate


class BaseDetector(ABC):
    """Abstract base class for all deterministic and model-based PII detectors."""

    def __init__(self, name: str, supported_categories: Set[str]):
        self.name = name
        self.supported_categories = supported_categories

    @abstractmethod
    def detect(self, text: str) -> List[PIICandidate]:
        """Scan input text and return a list of PIICandidate entities with character offsets."""
        pass
