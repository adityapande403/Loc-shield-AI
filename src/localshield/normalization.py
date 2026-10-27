"""Unicode normalization, character mapping, and script detection with offset preservation."""

from __future__ import annotations

import unicodedata
import regex as re
from typing import Dict, List, Set, Tuple

from localshield.offsets import OffsetTracker

# Zero-width & invisible characters frequently used to evade detection
ZERO_WIDTH_CHARS = frozenset({
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner (ZWNJ)
    "\u200D",  # Zero-width joiner (ZWJ)
    "\uFEFF",  # Byte order mark (BOM) / zero-width no-break space
    "\u2060",  # Word joiner
    "\u180E",  # Mongolian vowel separator
    "\u00AD",  # Soft hyphen
})

# Devanagari digits to ASCII (०-९ -> 0-9)
DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}

# Fullwidth ASCII conversion (e.g. ０-９, ａ-ｚ, Ａ-Ｚ)
FULLWIDTH_OFFSET = 0xFEE0


def detect_scripts(text: str) -> Set[str]:
    """Detect writing scripts present in text (e.g. 'latin', 'devanagari', 'mixed')."""
    scripts: Set[str] = set()
    for ch in text:
        name = unicodedata.name(ch, "")
        if "DEVANAGARI" in name:
            scripts.add("devanagari")
        elif "LATIN" in name:
            scripts.add("latin")
        elif "ARABIC" in name:
            scripts.add("arabic")
        elif "EMOJI" in name or 0x1F300 <= ord(ch) <= 0x1F9FF:
            scripts.add("emoji")
    return scripts


def normalize_with_offsets(text: str) -> Tuple[str, OffsetTracker]:
    """Normalize text (strip zero-width characters, normalize unicode digits & fullwidth)

    Returns:
        Tuple of (normalized_text, offset_tracker) where offset_tracker can map
        any character range in normalized_text back to exact original character indices.
    """
    normalized_chars: List[str] = []
    norm_to_orig_map: List[int] = []

    for orig_idx, ch in enumerate(text):
        # 1. Skip zero-width characters (they won't appear in normalized text)
        if ch in ZERO_WIDTH_CHARS:
            continue

        # 2. Convert Devanagari digits to ASCII
        if ch in DEVANAGARI_DIGITS:
            normalized_chars.append(DEVANAGARI_DIGITS[ch])
            norm_to_orig_map.append(orig_idx)
            continue

        # 3. Convert Fullwidth characters to ASCII equivalents
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            ascii_ch = chr(code - FULLWIDTH_OFFSET)
            normalized_chars.append(ascii_ch)
            norm_to_orig_map.append(orig_idx)
            continue

        # Default: keep character as-is
        normalized_chars.append(ch)
        norm_to_orig_map.append(orig_idx)

    normalized_text = "".join(normalized_chars)
    tracker = OffsetTracker(
        original_text=text,
        normalized_text=normalized_text,
        norm_to_orig_map=norm_to_orig_map,
    )
    return normalized_text, tracker
