"""Unit tests for the 4 Redaction Modes and Reversible Encrypted Storage."""

import pytest
from localshield.pseudonymization import EncryptedMappingStore
from localshield.redaction import RedactionEngine
from localshield.schemas import PIICandidate, RedactionMode


class TestRedactionModes:
    def setup_method(self):
        self.engine = RedactionEngine()

    def test_mode_a_masking(self):
        text = "Rahul Sharma called 9876543210."
        cands = [
            PIICandidate(entity_type="PERSON", start=0, end=12, text="Rahul Sharma", source="test"),
            PIICandidate(entity_type="PHONE_NUMBER", start=20, end=30, text="9876543210", source="test"),
        ]
        res = self.engine.redact(text, cands, mode=RedactionMode.MASK)
        assert res.redacted_text == "[PERSON] called [PHONE_NUMBER]."
        assert res.redacted_count == 2
        # Check span offsets in redacted text
        assert res.spans[0].placeholder == "[PERSON]"
        assert res.spans[0].redacted_start == 0
        assert res.spans[0].redacted_end == len("[PERSON]")

    def test_mode_b_consistent_pseudonymization(self):
        text = "Rahul Sharma met Amit, then Rahul Sharma left."
        cands = [
            PIICandidate(entity_type="PERSON", start=0, end=12, text="Rahul Sharma", source="test"),
            PIICandidate(entity_type="PERSON", start=17, end=21, text="Amit", source="test"),
            PIICandidate(entity_type="PERSON", start=28, end=40, text="Rahul Sharma", source="test"),
        ]
        res = self.engine.redact(text, cands, mode=RedactionMode.PSEUDONYMIZE)
        assert res.redacted_text == "[PERSON_1] met [PERSON_2], then [PERSON_1] left."

    def test_mode_c_type_preserving(self):
        text = "Contact: 9876543210"
        cands = [
            PIICandidate(entity_type="PHONE_NUMBER", start=9, end=19, text="9876543210", source="test"),
        ]
        res = self.engine.redact(text, cands, mode=RedactionMode.TYPE_PRESERVING)
        assert res.redacted_text == "Contact: [PHONE_NUMBER_1]"

    def test_mode_d_selective_and_partial_masking(self):
        text = "Card 4532015012345678 and Phone 9876543210"
        cands = [
            PIICandidate(entity_type="CREDIT_CARD", start=5, end=21, text="4532015012345678", source="test"),
            PIICandidate(entity_type="PHONE_NUMBER", start=32, end=42, text="9876543210", source="test"),
        ]
        # Only redact CREDIT_CARD with partial mask (keep last 4 digits)
        partial_rules = {"CREDIT_CARD": {"keep_last": 4, "mask_char": "X"}}
        res = self.engine.redact(
            text,
            cands,
            categories={"CREDIT_CARD"},
            mode=RedactionMode.SELECTIVE,
            partial_mask_rules=partial_rules,
        )
        assert res.redacted_text == "Card XXXXXXXXXXXX5678 and Phone 9876543210"


class TestReversiblePseudonymization:
    def test_reversible_encryption_and_authorization(self, tmp_path):
        store = EncryptedMappingStore(storage_dir=str(tmp_path))
        engine = RedactionEngine(mapping_store=store)

        original_text = "Transfer money to Vikram Malhotra."
        cands = [
            PIICandidate(entity_type="PERSON", start=18, end=33, text="Vikram Malhotra", source="test"),
        ]

        # 1. Redact with reversible enabled
        res = engine.redact(
            original_text,
            cands,
            mode=RedactionMode.PSEUDONYMIZE,
            enable_reversible=True,
        )
        assert res.session_id is not None
        assert "[PERSON_1]" in res.redacted_text

        # 2. Unauthorized attempt must raise PermissionError
        with pytest.raises(PermissionError):
            store.reidentify(res.redacted_text, res.session_id, authorized=False)

        # 3. Authorized re-identification restores exact text
        restored = store.reidentify(res.redacted_text, res.session_id, authorized=True)
        assert restored == original_text
