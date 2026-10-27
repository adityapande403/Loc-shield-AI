"""Redaction engine supporting Masking, Consistent Pseudonymization, Type-Preserving, and Selective Redaction."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from localshield.pseudonymization import EncryptedMappingStore
from localshield.schemas import (
    PIICandidate,
    RedactionMode,
    RedactionResult,
    RedactionSpan,
)
from localshield.taxonomy import DEFAULT_REDACT_CATEGORIES, get_canonical_placeholder


class RedactionEngine:
    """Core redaction engine implementing offset-safe replacements and pseudonym tracking."""

    def __init__(self, mapping_store: Optional[EncryptedMappingStore] = None):
        self.mapping_store = mapping_store or EncryptedMappingStore()

    def redact(
        self,
        text: str,
        candidates: List[PIICandidate],
        categories: Optional[Set[str]] = None,
        mode: RedactionMode = RedactionMode.MASK,
        enable_reversible: bool = False,
        session_id: Optional[str] = None,
        partial_mask_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> RedactionResult:
        """Apply redaction to text using resolved candidates."""
        start_time = time.perf_counter()

        if not text or not candidates:
            return RedactionResult(
                redacted_text=text,
                spans=[],
                detected_count=len(candidates),
                redacted_count=0,
                processing_time_ms=(time.perf_counter() - start_time) * 1000.0,
                mode=mode,
            )

        active_categories = categories if categories is not None else DEFAULT_REDACT_CATEGORIES

        # 1. Filter candidates to only those eligible for redaction
        target_candidates: List[PIICandidate] = [
            c for c in candidates if c.entity_type.upper() in active_categories
        ]

        # 2. Build placeholders for each candidate
        # Tracking for consistent pseudonymization (Mode B & C)
        entity_seen_counts: Dict[str, int] = {}
        entity_value_to_placeholder: Dict[str, str] = {}
        reversible_mapping: Dict[str, str] = {}

        # First pass (left to right) to assign consistent placeholders
        candidate_placeholders: List[str] = []
        for cand in target_candidates:
            raw_val = cand.text
            cat = cand.entity_type.upper()
            canonical = get_canonical_placeholder(cat)

            if mode == RedactionMode.MASK:
                placeholder = f"[{canonical}]"

            elif mode in (RedactionMode.PSEUDONYMIZE, RedactionMode.TYPE_PRESERVING):
                # Normalize key for consistency (e.g. case insensitive for emails/names)
                lookup_key = f"{cat}:{raw_val.strip().lower()}"
                if lookup_key in entity_value_to_placeholder:
                    placeholder = entity_value_to_placeholder[lookup_key]
                else:
                    curr_idx = entity_seen_counts.get(cat, 0) + 1
                    entity_seen_counts[cat] = curr_idx
                    placeholder = f"[{canonical}_{curr_idx}]"
                    entity_value_to_placeholder[lookup_key] = placeholder

            elif mode == RedactionMode.SELECTIVE:
                # Check for partial masking rule
                if partial_mask_rules and cat in partial_mask_rules:
                    rule = partial_mask_rules[cat]
                    keep_last = rule.get("keep_last", 0)
                    mask_char = rule.get("mask_char", "X")
                    if keep_last > 0 and len(raw_val) > keep_last:
                        placeholder = (mask_char * (len(raw_val) - keep_last)) + raw_val[-keep_last:]
                    else:
                        placeholder = f"[{canonical}]"
                else:
                    placeholder = f"[{canonical}]"
            else:
                placeholder = f"[{canonical}]"

            candidate_placeholders.append(placeholder)
            if enable_reversible:
                reversible_mapping[placeholder] = raw_val

        # 3. Apply replacements right-to-left (descending start offset) for offset safety
        # Pair candidates with placeholders and sort descending by start
        paired = list(zip(target_candidates, candidate_placeholders))
        paired.sort(key=lambda item: item[0].start, reverse=True)

        working_text = text
        generated_spans: List[RedactionSpan] = []

        for cand, placeholder in paired:
            # Validate bounds
            if cand.start < 0 or cand.end > len(working_text) or cand.start > cand.end:
                continue

            # Replace slice
            before = working_text[:cand.start]
            after = working_text[cand.end:]
            working_text = before + placeholder + after

        # 4. Now compute exact forward offsets in the final redacted text
        # Process left to right on original target_candidates
        paired_ltr = list(zip(target_candidates, candidate_placeholders))
        paired_ltr.sort(key=lambda item: item[0].start)

        offset_shift = 0
        for cand, placeholder in paired_ltr:
            orig_len = cand.end - cand.start
            repl_len = len(placeholder)

            redacted_start = cand.start + offset_shift
            redacted_end = redacted_start + repl_len
            offset_shift += (repl_len - orig_len)

            generated_spans.append(
                RedactionSpan(
                    entity_type=cand.entity_type,
                    original_start=cand.start,
                    original_end=cand.end,
                    redacted_start=redacted_start,
                    redacted_end=redacted_end,
                    placeholder=placeholder,
                    confidence=cand.confidence,
                    source=cand.source,
                    is_validated=cand.is_validated,
                )
            )

        # 5. Handle reversible session storage if enabled
        persisted_session_id: Optional[str] = None
        if enable_reversible and reversible_mapping:
            persisted_session_id = self.mapping_store.save_session(
                mapping=reversible_mapping, session_id=session_id
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RedactionResult(
            redacted_text=working_text,
            spans=generated_spans,
            detected_count=len(candidates),
            redacted_count=len(generated_spans),
            processing_time_ms=elapsed_ms,
            mode=mode,
            session_id=persisted_session_id,
        )
