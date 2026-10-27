"""Deterministic detection engine aggregation."""

from __future__ import annotations

from typing import List, Optional, Set

from localshield.detection.base import BaseDetector
from localshield.detection.indian_identifiers import IndianIdentifiersDetector
from localshield.detection.international import InternationalIdentifiersDetector
from localshield.detection.secrets import SecretsDetector
from localshield.normalization import normalize_with_offsets
from localshield.schemas import PIICandidate


class CompositeDeterministicDetector(BaseDetector):
    """Orchestrates all deterministic detectors (Indian, International, Secrets).

    Automatically handles Unicode normalization and maps character offsets back
    to the original text coordinates.
    """

    def __init__(self, entropy_threshold: float = 3.2):
        self.indian_detector = IndianIdentifiersDetector()
        self.intl_detector = InternationalIdentifiersDetector()
        self.secrets_detector = SecretsDetector(entropy_threshold=entropy_threshold)

        all_cats = (
            self.indian_detector.supported_categories
            | self.intl_detector.supported_categories
            | self.secrets_detector.supported_categories
        )
        super().__init__(name="composite_deterministic", supported_categories=all_cats)
        self.sub_detectors = [
            self.indian_detector,
            self.intl_detector,
            self.secrets_detector,
        ]

    def detect(self, text: str, categories: Optional[Set[str]] = None) -> List[PIICandidate]:
        """Execute all detectors on original text and normalized text with offset mapping."""
        if not text:
            return []

        all_candidates: List[PIICandidate] = []

        # 1. Direct detection on original text
        for detector in self.sub_detectors:
            try:
                candidates = detector.detect(text)
                for cand in candidates:
                    if categories is None or cand.entity_type in categories:
                        all_candidates.append(cand)
            except Exception:
                # Deterministic detectors should never crash pipeline
                continue

        # 2. Also run on normalized text (for zero-width stripping, devanagari digits, etc.)
        norm_text, offset_tracker = normalize_with_offsets(text)
        if norm_text != text:
            for detector in self.sub_detectors:
                try:
                    norm_candidates = detector.detect(norm_text)
                    for cand in norm_candidates:
                        if categories is not None and cand.entity_type not in categories:
                            continue
                        # Map character span back to original text!
                        orig_start, orig_end = offset_tracker.map_span(cand.start, cand.end)
                        orig_slice = text[orig_start:orig_end]

                        all_candidates.append(
                            PIICandidate(
                                entity_type=cand.entity_type,
                                start=orig_start,
                                end=orig_end,
                                text=orig_slice,
                                confidence=cand.confidence,
                                source=f"{cand.source}:normalized",
                                detector_name=cand.detector_name,
                                is_validated=cand.is_validated,
                                metadata={**cand.metadata, "normalized_match": True},
                            )
                        )
                except Exception:
                    continue

        return all_candidates


__all__ = [
    "BaseDetector",
    "CompositeDeterministicDetector",
    "IndianIdentifiersDetector",
    "InternationalIdentifiersDetector",
    "SecretsDetector",
]
