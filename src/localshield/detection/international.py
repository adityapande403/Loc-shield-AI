"""Deterministic detection and algorithmic validation for international identifiers.

Includes credit/debit cards (Luhn validation), RFC 5322 emails, international phone numbers,
IPv4 and IPv6 addresses, URLs containing embedded credentials, dates of birth, and usernames.
"""

from __future__ import annotations

import ipaddress
import regex as re
from typing import List, Set

from localshield.detection.base import BaseDetector
from localshield.detection.luhn import validate_luhn
from localshield.schemas import PIICandidate


class InternationalIdentifiersDetector(BaseDetector):
    """Detector for international structured identifiers and network credentials."""

    def __init__(self):
        super().__init__(
            name="international_identifiers_detector",
            supported_categories={
                "CREDIT_CARD",
                "EMAIL",
                "PHONE_NUMBER",
                "IP_ADDRESS",
                "AUTH_TOKEN",
                "DATE_OF_BIRTH",
                "USERNAME",
            }
        )

        # Credit / Debit Cards: 13-19 digits, optional spaces or dashes
        self.card_regex = re.compile(
            r"(?<!\d)(?:[3-6]\d{3}[\s\-]?(?:\d{4}[\s\-]?){2,3}\d{1,4}|[3-6]\d{12,18})(?!\d)"
        )

        # Standard Email RFC 5322 pattern
        self.email_regex = re.compile(
            r"(?i)\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63})\b"
        )

        # International Phone: E.164 with international country code
        self.intl_phone_regex = re.compile(
            r"""(?<!\w)\+(\d{1,3})[-.\s]?\(?(\d{1,4})\)?[-.\s]?(\d{1,4})[-.\s]?(\d{1,9})\b"""
        )

        # IPv4 pattern (with explicit 0-255 octet bounds)
        self.ipv4_regex = re.compile(
            r"(?<![0-9.])(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?![0-9.])"
        )

        # IPv6 pattern
        self.ipv6_regex = re.compile(
            r"(?<![0-9a-fA-F:])(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}(?![0-9a-fA-F:])|"
            r"(?<![0-9a-fA-F:])((?:[0-9a-fA-F]{1,4}:){1,6}|:):((?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}|:)(?![0-9a-fA-F:])"
        )

        # URL with embedded credentials: http(s)://user:password@host
        self.url_credential_regex = re.compile(
            r"(?i)https?://([A-Za-z0-9_.~-]+):([^@\s/:]+)@([A-Za-z0-9_.~-]+)"
        )

        # Date of Birth with context
        self.dob_regex = re.compile(
            r"""(?i)(?:(?:d\.?o\.?b\.?|date\s*of\s*birth|born\s*(?:on)?|जन्म\s*तिथि|janm\s*tithi)\s*[:=\-]?\s*)"""
            r"""(\b(?:[0-3]?[0-9][\/\-.][0-1]?[0-9][\/\-.](?:19|20)\d{2}|(?:19|20)\d{2}[\/\-.][0-1]?[0-9][\/\-.][0-3]?[0-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[0-3]?[0-9],?\s+(?:19|20)\d{2})\b)"""
        )

        # Username handles (@handle or username: ...)
        self.username_handle_regex = re.compile(
            r"(?<![\w@])@([a-zA-Z0-9_]{3,30})\b"
        )
        self.username_context_regex = re.compile(
            r"""(?i)(?:(?:user(?:name)?|login|handle)\s*[:=\-]\s*)([a-zA-Z0-9_.-]{3,32})\b"""
        )

    def detect(self, text: str) -> List[PIICandidate]:
        candidates: List[PIICandidate] = []

        # 1. Credit Cards with Luhn Check
        for match in self.card_regex.finditer(text):
            raw_val = match.group(0)
            digits = re.sub(r"[\s\-]", "", raw_val)
            if 13 <= len(digits) <= 19 and validate_luhn(digits):
                # Identify card network brand
                brand = "UNKNOWN"
                if digits.startswith("4"):
                    brand = "VISA"
                elif digits.startswith(("51", "52", "53", "54", "55", "22", "23", "24", "25", "26", "27")):
                    brand = "MASTERCARD"
                elif digits.startswith(("34", "37")):
                    brand = "AMEX"
                elif digits.startswith(("60", "65", "81", "82", "35")):
                    brand = "RUPAY"
                elif digits.startswith("6011") or digits.startswith("65"):
                    brand = "DISCOVER"

                candidates.append(
                    PIICandidate(
                        entity_type="CREDIT_CARD",
                        start=match.start(),
                        end=match.end(),
                        text=raw_val,
                        confidence=0.99,
                        source="deterministic:luhn_card",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"brand": brand, "digits_count": len(digits)},
                    )
                )

        # 2. Email Addresses
        for match in self.email_regex.finditer(text):
            email_val = match.group(1)
            # Remove any trailing period or comma from punctuation
            clean_email = email_val.rstrip(".,;")
            end_pos = match.start(1) + len(clean_email)
            candidates.append(
                PIICandidate(
                    entity_type="EMAIL",
                    start=match.start(1),
                    end=end_pos,
                    text=clean_email,
                    confidence=0.98,
                    source="deterministic:rfc5322_email",
                    detector_name=self.name,
                    is_validated=True,
                    metadata={"domain": clean_email.split("@")[-1]},
                )
            )

        # 3. International Phones
        for match in self.intl_phone_regex.finditer(text):
            phone_val = match.group(0)
            digits = re.sub(r"\D", "", phone_val)
            if 7 <= len(digits) <= 15:
                candidates.append(
                    PIICandidate(
                        entity_type="PHONE_NUMBER",
                        start=match.start(),
                        end=match.end(),
                        text=phone_val,
                        confidence=0.90,
                        source="deterministic:international_phone",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"type": "e164_intl"},
                    )
                )

        # 4. IP Addresses (IPv4 and IPv6)
        for match in self.ipv4_regex.finditer(text):
            ip_str = match.group(0)
            # Strict validation using standard library ipaddress
            try:
                ipaddress.IPv4Address(ip_str)
                candidates.append(
                    PIICandidate(
                        entity_type="IP_ADDRESS",
                        start=match.start(),
                        end=match.end(),
                        text=ip_str,
                        confidence=0.95,
                        source="deterministic:ipv4",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"version": 4},
                    )
                )
            except ValueError:
                pass

        for match in self.ipv6_regex.finditer(text):
            ip_str = match.group(0)
            try:
                ipaddress.IPv6Address(ip_str)
                candidates.append(
                    PIICandidate(
                        entity_type="IP_ADDRESS",
                        start=match.start(),
                        end=match.end(),
                        text=ip_str,
                        confidence=0.95,
                        source="deterministic:ipv6",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"version": 6},
                    )
                )
            except ValueError:
                pass

        # 5. URL Credentials
        for match in self.url_credential_regex.finditer(text):
            cred_str = f"{match.group(1)}:{match.group(2)}"
            start = match.start(1)
            end = match.end(2)
            candidates.append(
                PIICandidate(
                    entity_type="AUTH_TOKEN",
                    start=start,
                    end=end,
                    text=cred_str,
                    confidence=0.98,
                    source="deterministic:url_credential",
                    detector_name=self.name,
                    is_validated=True,
                    metadata={"has_username": True, "has_password": True},
                )
            )

        # 6. Date of Birth
        for match in self.dob_regex.finditer(text):
            dob_val = match.group(1)
            start, end = match.span(1)
            candidates.append(
                PIICandidate(
                    entity_type="DATE_OF_BIRTH",
                    start=start,
                    end=end,
                    text=dob_val,
                    confidence=0.92,
                    source="deterministic:dob_context",
                    detector_name=self.name,
                    is_validated=True,
                )
            )

        # 7. Usernames
        for match in self.username_handle_regex.finditer(text):
            handle = match.group(0)
            # Avoid email trailing parts or domains
            prev_char = text[match.start() - 1] if match.start() > 0 else ""
            if prev_char not in (" ", "\t", "\n", "(", "[", "{", ":", ","):
                continue
            candidates.append(
                PIICandidate(
                    entity_type="USERNAME",
                    start=match.start(),
                    end=match.end(),
                    text=handle,
                    confidence=0.85,
                    source="deterministic:handle",
                    detector_name=self.name,
                    is_validated=True,
                )
            )

        for match in self.username_context_regex.finditer(text):
            uname = match.group(1)
            start, end = match.span(1)
            candidates.append(
                PIICandidate(
                    entity_type="USERNAME",
                    start=start,
                    end=end,
                    text=uname,
                    confidence=0.88,
                    source="deterministic:username_context",
                    detector_name=self.name,
                    is_validated=True,
                )
            )

        return candidates
