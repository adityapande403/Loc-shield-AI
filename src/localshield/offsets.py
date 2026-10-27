"""Character offset tracking and bi-directional span mapping between original and normalized text."""

from __future__ import annotations

from typing import List, Tuple


class OffsetTracker:
    """Tracks mapping from normalized text back to original text character offsets.

    Enables safe normalization (e.g. zero-width removal, digit conversion, NFKC)
    without corrupting the original character coordinates required for redaction.
    """

    def __init__(self, original_text: str, normalized_text: str, norm_to_orig_map: List[int]):
        self.original_text = original_text
        self.normalized_text = normalized_text
        self._norm_to_orig = norm_to_orig_map

    @classmethod
    def identity(cls, text: str) -> OffsetTracker:
        """Create an identity tracker where normalized text == original text."""
        return cls(original_text=text, normalized_text=text, norm_to_orig_map=list(range(len(text))))

    def map_span(self, norm_start: int, norm_end: int) -> Tuple[int, int]:
        """Map a character span [norm_start, norm_end) in normalized text to [orig_start, orig_end) in original text."""
        if not self._norm_to_orig:
            return 0, 0

        # Clamp boundaries to valid normalized indices
        n_len = len(self._norm_to_orig)
        if norm_start >= n_len:
            return len(self.original_text), len(self.original_text)

        clamped_start = max(0, min(norm_start, n_len - 1))
        orig_start = self._norm_to_orig[clamped_start]

        if norm_end <= norm_start:
            return orig_start, orig_start

        clamped_end_idx = max(0, min(norm_end - 1, n_len - 1))
        orig_end = self._norm_to_orig[clamped_end_idx] + 1

        # Ensure valid non-negative range within original text
        orig_start = max(0, min(orig_start, len(self.original_text)))
        orig_end = max(orig_start, min(orig_end, len(self.original_text)))

        return orig_start, orig_end

    def get_original_slice(self, norm_start: int, norm_end: int) -> str:
        """Extract the exact substring from original text corresponding to normalized span."""
        orig_start, orig_end = self.map_span(norm_start, norm_end)
        return self.original_text[orig_start:orig_end]
