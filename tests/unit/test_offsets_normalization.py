"""Unit tests for Unicode normalization, zero-width evasion detection, and offset tracking."""

import pytest
from localshield.detection import CompositeDeterministicDetector
from localshield.normalization import normalize_with_offsets
from localshield.offsets import OffsetTracker


class TestOffsetTracker:
    def test_identity_mapping(self):
        text = "Hello World"
        tracker = OffsetTracker.identity(text)
        s, e = tracker.map_span(0, 5)
        assert (s, e) == (0, 5)
        assert tracker.get_original_slice(0, 5) == "Hello"

    def test_zero_width_space_offset_preservation(self):
        # Original text has zero-width space '\u200b' at index 2
        orig_text = "98\u200b76"
        norm_text, tracker = normalize_with_offsets(orig_text)

        assert norm_text == "9876"
        # In normalized text, span 0..4 covers all 4 digits
        orig_s, orig_e = tracker.map_span(0, 4)
        assert (orig_s, orig_e) == (0, 5)
        assert orig_text[orig_s:orig_e] == "98\u200b76"

    def test_zero_width_evasion_in_composite_detector(self):
        detector = CompositeDeterministicDetector()
        # Phone number with zero-width characters inside
        text = "Call 98\u200b7654\u200c3210 immediately."
        cands = detector.detect(text)
        phone_cands = [c for c in cands if c.entity_type == "PHONE_NUMBER"]

        assert len(phone_cands) >= 1
        cand = phone_cands[0]
        # Must span across the zero-width characters in original text
        assert cand.text == "98\u200b7654\u200c3210"
        assert text[cand.start:cand.end] == cand.text
