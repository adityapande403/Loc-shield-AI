"""Security tests: strict offline operation, no PII leakage in logs, input bounds, and safe error handling."""

import socket
import pytest
from localshield import LocalShield, LocalShieldConfig


class TestSecurityAndOfflineOperation:
    def test_strict_offline_mode_no_network_calls(self, monkeypatch):
        """Verify that redaction pipeline operates strictly offline without opening network sockets."""
        def mock_connect(*args, **kwargs):
            raise ConnectionRefusedError("Network access blocked in offline mode!")

        # Monkeypatch socket connect to ensure no network calls occur during inference
        monkeypatch.setattr(socket.socket, "connect", mock_connect)

        shield = LocalShield()
        text = "Rahul Sharma called 9876543210 from Mumbai regarding account ABCPE1234F."
        res = shield.redact(text)

        assert "[PHONE_NUMBER]" in res.redacted_text
        assert "[PAN]" in res.redacted_text
        assert "9876543210" not in res.redacted_text

    def test_max_input_length_enforcement(self):
        """Verify bounded input sizes prevent memory exhaustion attacks."""
        cfg = LocalShieldConfig(max_input_chars=1000)
        shield = LocalShield(config=cfg)

        huge_text = "A" * 2000
        with pytest.raises(ValueError, match="exceeds maximum configured limit"):
            shield.detect(huge_text)

    def test_malformed_unicode_and_adversarial_separators(self):
        """Verify malformed Unicode, combining characters, and surrogate pairs fail safely."""
        shield = LocalShield()
        malformed_inputs = [
            "Test\x00\x01\x02NullBytes",
            "\u200b\u200c\u200d\ufeff\u2060" * 50,
            "Adversarial RTL: \u202eRahul Sharma\u202c called 9876543210",
            "Mixed: राहुल\u0300\u0301\u0302 Sharma 9876543210",
        ]
        for item in malformed_inputs:
            # Must process safely without crashing
            res = shield.redact(item)
            assert isinstance(res.redacted_text, str)

    def test_no_sensitive_values_in_safe_representation(self):
        """Verify candidate representation excludes raw sensitive values for logging."""
        shield = LocalShield()
        cands = shield.detect("Aadhaar: 234567890124, Secret: AKIAIOSFODNN7EXAMPLE")
        for c in cands:
            safe_dict = c.to_safe_dict()
            assert "text" not in safe_dict
            assert "AKIAIOSFODNN7EXAMPLE" not in str(safe_dict)
            assert "234567890124" not in str(safe_dict)
