"""Integration tests for end-to-end LocalShield engine and FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from localshield import LocalShield, RedactionMode
from localshield.api.app import create_app


class TestEndToEndPipeline:
    def setup_method(self):
        self.shield = LocalShield()

    def test_end_to_end_multilingual_detection_and_redaction(self):
        text = "Rahul Sharma from Mumbai called regarding HDFC Bank account. Contact: 9876543210, PAN: ABCPE1234F."
        res = self.shield.redact(text, mode=RedactionMode.MASK)

        assert "[PHONE_NUMBER]" in res.redacted_text
        assert "[PAN]" in res.redacted_text
        assert "[PERSON]" in res.redacted_text
        assert "9876543210" not in res.redacted_text
        assert "ABCPE1234F" not in res.redacted_text

    def test_batch_redaction(self):
        texts = [
            "Contact Amit Kumar at 9999988888.",
            "Meeting at Google New Delhi office.",
            "System logs show normal CPU load at 22%.",
        ]
        results = self.shield.redact_batch(texts, mode=RedactionMode.MASK)
        assert len(results) == 3
        assert "[PHONE_NUMBER]" in results[0].redacted_text
        assert "[ORGANIZATION]" in results[1].redacted_text
        assert results[2].redacted_text == texts[2]


class TestFastAPIService:
    def test_api_endpoints(self):
        app = create_app()
        client = TestClient(app)

        # Health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] in ("healthy", "degraded")

        # Ready
        r_ready = client.get("/ready")
        assert r_ready.status_code == 200
        assert r_ready.json()["ready"] is True

        # Detect
        r_det = client.post("/detect", json={"text": "Email test@example.com from 10.0.0.1"})
        assert r_det.status_code == 200
        assert r_det.json()["total_detected"] >= 2

        # Redact
        r_red = client.post("/redact", json={"text": "Call Rahul at 9876543210", "mode": "mask"})
        assert r_red.status_code == 200
        assert "[PHONE_NUMBER]" in r_red.json()["redacted_text"]

        # Analyze
        r_ana = client.post("/analyze", json={"text": "Call 9876543210 with key AKIAIOSFODNN7EXAMPLE"})
        assert r_ana.status_code == 200
        assert r_ana.json()["risk_score"] > 0.8
