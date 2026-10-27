"""Unit tests for deterministic identifiers (Aadhaar, PAN, Card, Phone, IFSC, Secrets, etc.)."""

import pytest
from localshield.detection import CompositeDeterministicDetector
from localshield.detection.indian_identifiers import IndianIdentifiersDetector
from localshield.detection.international import InternationalIdentifiersDetector
from localshield.detection.luhn import generate_luhn_check_digit, validate_luhn
from localshield.detection.secrets import SecretsDetector, calculate_shannon_entropy
from localshield.detection.verhoeff import generate_verhoeff_check_digit, validate_verhoeff


class TestVerhoeffAndAadhaar:
    def test_verhoeff_generation_and_validation(self):
        prefix = "23456789012"
        cd = generate_verhoeff_check_digit(prefix)
        full_aadhaar = prefix + str(cd)
        assert validate_verhoeff(full_aadhaar) is True
        # Tampered check digit must fail
        tampered = prefix + str((cd + 1) % 10)
        assert validate_verhoeff(tampered) is False

    def test_aadhaar_detector_positive_and_negative(self):
        detector = IndianIdentifiersDetector()
        valid_cd = generate_verhoeff_check_digit("23456789012")
        valid_num = f"2345 6789 012{valid_cd}"

        text = f"My Aadhaar is {valid_num} and fake is 0123 4567 8901 and tampered is 2345 6789 0129."
        cands = detector.detect(text)
        aadhaar_cands = [c for c in cands if c.entity_type == "AADHAAR"]

        assert len(aadhaar_cands) >= 1
        assert any(c.text == valid_num for c in aadhaar_cands)
        # Numbers starting with 0 or with invalid checksum must NOT be detected as valid Aadhaar
        assert not any("0123 4567 8901" in c.text for c in aadhaar_cands)


class TestLuhnAndCards:
    def test_luhn_generation_and_validation(self):
        prefix = "45320150123456"
        cd = generate_luhn_check_digit(prefix)
        card_num = prefix + str(cd)
        assert validate_luhn(card_num) is True
        # Tampered number
        assert validate_luhn(prefix + str((cd + 1) % 10)) is False

    def test_credit_card_detector(self):
        detector = InternationalIdentifiersDetector()
        valid_card = "453201501234567"  # Valid Luhn Visa test prefix
        invalid_card = "453201501234568"

        text = f"Card: {valid_card}, Invalid: {invalid_card}"
        cands = detector.detect(text)
        card_cands = [c for c in cands if c.entity_type == "CREDIT_CARD"]

        assert len(card_cands) == 1
        assert card_cands[0].text == valid_card
        assert card_cands[0].metadata.get("brand") == "VISA"


class TestPANAndIFSC:
    def test_pan_statutory_validation(self):
        detector = IndianIdentifiersDetector()
        # 4th char: P (person), C (company) -> valid
        valid_pan_person = "ABCPE1234F"
        valid_pan_company = "ABCCE5678G"
        # 4th char: X (not an entity type) -> invalid
        invalid_pan = "ABCXE1234F"

        text = f"PANs: {valid_pan_person}, {valid_pan_company}, {invalid_pan}"
        cands = detector.detect(text)
        pan_cands = [c for c in cands if c.entity_type == "PAN" and c.is_validated]

        texts = [c.text for c in pan_cands]
        assert valid_pan_person in texts
        assert valid_pan_company in texts
        assert invalid_pan not in texts

    def test_ifsc_code_detection(self):
        detector = IndianIdentifiersDetector()
        text = "Transfer to SBIN0001234 or HDFC0000456"
        cands = detector.detect(text)
        ifsc_cands = [c for c in cands if c.entity_type == "IFSC"]
        assert len(ifsc_cands) == 2
        assert ifsc_cands[0].text == "SBIN0001234"
        assert ifsc_cands[1].text == "HDFC0000456"


class TestPhoneAndEmailAndIP:
    def test_phone_numbers(self):
        detector = IndianIdentifiersDetector()
        text = "Call +91 9876543210 or 8888899999 or 09123456789. Ignore 12345."
        cands = detector.detect(text)
        phone_cands = [c for c in cands if c.entity_type == "PHONE_NUMBER"]
        assert len(phone_cands) >= 2

    def test_email_and_ip(self):
        detector = InternationalIdentifiersDetector()
        text = "Contact user@example.com from 192.168.1.1 or 2001:0db8:85a3:0000:0000:8a2e:0370:7334. Ignore 999.999.999.999."
        cands = detector.detect(text)
        emails = [c for c in cands if c.entity_type == "EMAIL"]
        ips = [c for c in cands if c.entity_type == "IP_ADDRESS"]

        assert len(emails) == 1
        assert emails[0].text == "user@example.com"
        assert len(ips) >= 1
        assert any(c.text == "192.168.1.1" for c in ips)
        # Invalid IP 999.999.999.999 must NOT be detected
        assert not any(c.text == "999.999.999.999" for c in ips)


class TestSecretsAndEntropy:
    def test_shannon_entropy(self):
        low_entropy = "aaaaaaaaaaaa"
        high_entropy = "k9#mP$2vL@8qZ!1x"
        assert calculate_shannon_entropy(low_entropy) < 1.0
        assert calculate_shannon_entropy(high_entropy) > 3.5

    def test_api_key_patterns(self):
        detector = SecretsDetector()
        text = "AWS: AKIAIOSFODNN7EXAMPLE, GitHub: ghp_1234567890abcdefghijklmnopqrstuvwxyz, Key: sk-proj-1234567890abcdef12345678"
        cands = detector.detect(text)
        api_keys = [c for c in cands if c.entity_type == "API_KEY"]
        assert len(api_keys) >= 3
