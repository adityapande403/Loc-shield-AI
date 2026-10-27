"""Span resolution engine for conflict resolution, nesting, and deterministic tie-breaking."""

from __future__ import annotations

from typing import List, Optional

from localshield.schemas import PIICandidate
from localshield.taxonomy import get_category_priority


class SpanResolver:
    """Resolves overlapping, nested, and conflicting PII candidate spans."""

    def __init__(self, prefer_deterministic: bool = True):
        self.prefer_deterministic = prefer_deterministic

    def resolve(self, candidates: List[PIICandidate], text_length: Optional[int] = None) -> List[PIICandidate]:
        """Resolve conflicts and return a non-overlapping, sorted list of candidates.

        Resolution Policy:
        1. Boundary check: Drop spans out of bounds (0 <= start <= end <= text_length).
        2. Exact duplicate deduplication: Keep highest confidence & deterministic source.
        3. Priority ranking: Use taxonomy priority rank (Critical structured > Contextual).
        4. Confidence & length tie-breaking: Deterministic ordering for exact reproducibility.
        """
        if not candidates:
            return []

        # 1. Filter invalid bounds
        valid_candidates: List[PIICandidate] = []
        for c in candidates:
            if c.start < 0 or c.end < c.start:
                continue
            if text_length is not None and c.end > text_length:
                continue
            if c.start == c.end:
                # Discard zero-length spans
                continue
            valid_candidates.append(c)

        if not valid_candidates:
            return []

        # 2. Sort candidates by starting offset, then by priority rank descending, then by length descending
        def sort_key(c: PIICandidate):
            priority = get_category_priority(c.entity_type)
            is_determ = 1 if "deterministic" in c.source else 0
            span_len = c.end - c.start
            # We want to process entities primarily by priority & confidence
            return (
                -priority,
                -is_determ,
                -round(c.confidence, 4),
                -span_len,
                c.start,
            )

        # 3. Greedy non-overlapping interval selection based on priority
        # Sort all candidates by preference
        ranked_candidates = sorted(valid_candidates, key=sort_key)
        accepted: List[PIICandidate] = []

        for cand in ranked_candidates:
            # Check overlap with already accepted higher-priority candidates
            overlap = False
            for existing in accepted:
                # Overlap condition: max(cand.start, existing.start) < min(cand.end, existing.end)
                if max(cand.start, existing.start) < min(cand.end, existing.end):
                    overlap = True
                    break

            if not overlap:
                accepted.append(cand)

        # 4. Final output sorted strictly by start offset ascending
        accepted.sort(key=lambda c: (c.start, c.end))
        return accepted
