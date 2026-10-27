"""Regression test suite on fixed corpus covering all supported categories and languages."""

import pytest
from localshield import LocalShield, RedactionMode
from localshield.detection.verhoeff import generate_verhoeff_check_digit


class TestRegressionCorpus:
    def setup_method(self):
        self.shield = LocalShield()

    @pytest.mark.parametrize(
        "input_text,expected_category,expected_placeholder",
        [
            ("Call me at 9876543210 please", "PHONE_NUMBER", "[PHONE_NUMBER]"),
            ("Email admin@localshield.internal", "EMAIL", "[EMAIL]"),
            ("Host IP is 192.168.1.100", "IP_ADDRESS", "[IP_ADDRESS]"),
            ("Permanent Account Number: ABCPE1234F", "PAN", "[PAN]"),
            ("IFSC Code: SBIN0001234", "IFSC", "[IFSC]"),
            ("Car registration DL 01 AB 1234", "VEHICLE_REGISTRATION", "[VEHICLE_REGISTRATION]"),
            ("Secret token: AKIAIOSFODNN7EXAMPLE", "API_KEY", "[API_KEY]"),
            ("User @aditya_kumar logged in", "USERNAME", "[USERNAME]"),
            ("Born on date of birth: 15/08/1990", "DATE_OF_BIRTH", "[DATE_OF_BIRTH]"),
        ],
    )
    def test_fixed_regression_categories(self, input_text, expected_category, expected_placeholder):
        res = self.shield.redact(input_text, mode=RedactionMode.MASK)
        assert expected_placeholder in res.redacted_text
        assert any(s.entity_type == expected_category for s in res.spans)

    def test_aadhaar_regression(self):
        cd = generate_verhoeff_check_digit("23456789012")
        aadh_num = f"2345 6789 012{cd}"
        text = f"My Aadhaar number is {aadh_num}"
        res = self.shield.redact(text, mode=RedactionMode.MASK)
        assert "[AADHAAR]" in res.redacted_text
        assert aadh_num not in res.redacted_text

    def test_negative_regression_no_redaction(self):
        text = "The quarterly technical report was approved by the committee without modification."
        res = self.shield.redact(text)
        assert res.redacted_text == text
        assert res.redacted_count == 0

    def test_overlapping_resolution_regression(self):
        # A phone number inside address context must resolve deterministically
        text = "Visit Flat 402, Shanti Heights, Sector 62, Noida or call 9876543210"
        cands = self.shield.detect(text)
        types = [c.entity_type for c in cands]
        assert "PHONE_NUMBER" in types
