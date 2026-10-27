"""Deterministic and heuristic detection for API keys, access tokens, auth tokens, and secrets."""

from __future__ import annotations

import collections
import math
import regex as re
from typing import List, Set

from localshield.detection.base import BaseDetector
from localshield.schemas import PIICandidate


def calculate_shannon_entropy(s: str) -> float:
    """Calculate the Shannon entropy of a string (in bits per symbol)."""
    if not s:
        return 0.0
    freq = collections.Counter(s)
    total = len(s)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


class SecretsDetector(BaseDetector):
    """Detector for API keys, bearer/access tokens, JWTs, and high-entropy secrets."""

    def __init__(self, entropy_threshold: float = 3.2):
        super().__init__(
            name="secrets_detector",
            supported_categories={"API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "OTHER_SENSITIVE_ID"}
        )
        self.entropy_threshold = entropy_threshold

        # High-confidence prefix patterns
        self.known_patterns = [
            (
                "AWS_ACCESS_KEY",
                "API_KEY",
                re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
                1.0,
            ),
            (
                "GITHUB_TOKEN",
                "API_KEY",
                re.compile(r"\b(gh[pousr]_[A-Za-z0-9_]{36,255})\b"),
                1.0,
            ),
            (
                "SLACK_TOKEN",
                "API_KEY",
                re.compile(r"\b(xox[baprs]-[0-9a-zA-Z-]{10,72})\b"),
                1.0,
            ),
            (
                "GOOGLE_API_KEY",
                "API_KEY",
                re.compile(r"\b(AIza[0-9A-Za-z_-]{35})\b"),
                1.0,
            ),
            (
                "OPENAI_API_KEY",
                "API_KEY",
                re.compile(r"\b(sk-(?:proj-|live-)?[a-zA-Z0-9_-]{20,})\b"),
                0.95,
            ),
            (
                "ANTHROPIC_KEY",
                "API_KEY",
                re.compile(r"\b(sk-ant-[a-zA-Z0-9_-]{20,})\b"),
                0.95,
            ),
            (
                "JWT_TOKEN",
                "ACCESS_TOKEN",
                re.compile(r"\b(eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})\b"),
                0.98,
            ),
            (
                "PRIVATE_KEY_BLOCK",
                "AUTH_TOKEN",
                re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
                1.0,
            ),
        ]

        # Context-dependent patterns: keyword = "secret_string"
        self.contextual_secret_regex = re.compile(
            r"""(?i)\b(api[_-]?key|secret|token|auth[_-]?token|bearer|access[_-]?token|client[_-]?secret|password|passwd|api[_-]?secret)\s*[:=]\s*["']?([A-Za-z0-9_\-\.]{12,128})["']?""",
            re.MULTILINE
        )

    def detect(self, text: str) -> List[PIICandidate]:
        candidates: List[PIICandidate] = []

        # 1. Match known vendor formats
        for pattern_name, category, regex_pat, base_confidence in self.known_patterns:
            for match in regex_pat.finditer(text):
                # If capture group exists, use it
                span_text = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                start, end = match.span(1) if match.lastindex and match.lastindex >= 1 else match.span(0)

                candidates.append(
                    PIICandidate(
                        entity_type=category,
                        start=start,
                        end=end,
                        text=span_text,
                        confidence=base_confidence,
                        source=f"secrets:{pattern_name.lower()}",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"pattern": pattern_name, "length": len(span_text)},
                    )
                )

        # 2. Contextual secrets with entropy check
        for match in self.contextual_secret_regex.finditer(text):
            kw = match.group(1).lower()
            secret_val = match.group(2)
            start, end = match.span(2)

            entropy = calculate_shannon_entropy(secret_val)
            # Only classify if length >= 12 and entropy >= threshold or hexadecimal / base64
            if len(secret_val) >= 12 and entropy >= self.entropy_threshold:
                cat = "API_KEY" if "key" in kw or "secret" in kw else "AUTH_TOKEN"
                candidates.append(
                    PIICandidate(
                        entity_type=cat,
                        start=start,
                        end=end,
                        text=secret_val,
                        confidence=min(1.0, 0.70 + (entropy / 10.0)),
                        source="secrets:contextual_entropy",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={
                            "keyword": kw,
                            "entropy": round(entropy, 2),
                            "length": len(secret_val)
                        },
                    )
                )

        return candidates
