"""Deterministic detection and algorithmic validation for Indian identifiers.

Covers Aadhaar (Verhoeff checksum), PAN (statutory 10-char format rules), Indian phone numbers,
IFSC codes, vehicle registration plates, bank account numbers, passports, and driving licenses.
"""

from __future__ import annotations

import regex as re
from typing import List, Set

from localshield.detection.base import BaseDetector
from localshield.detection.verhoeff import validate_verhoeff
from localshield.schemas import PIICandidate

# Indian States and Union Territories alpha codes for Vehicle and DL validation
INDIAN_STATE_CODES = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "CH", "DN", "DD", "DL", "GA",
    "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN",
    "ML", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UP",
    "UK", "UA", "WB", "BH"  # BH = Bharat Series
})

# Prominent Indian Bank 4-letter IFSC prefixes
KNOWN_BANK_PREFIXES = frozenset({
    "SBIN", "HDFC", "ICIC", "UTIB", "PUNB", "BARB", "KKBK", "YESB", "IDIB",
    "CNRB", "UBIN", "IOBA", "BKID", "MAHB", "PSIB", "BDBL", "IDFB", "CSBK",
    "KVBL", "FDRL", "SIBL", "RATN", "INDB", "AIRP", "PYTM", "IPOS", "ESFB"
})


class IndianIdentifiersDetector(BaseDetector):
    """Detector for structured Indian national, financial, and vehicular identifiers."""

    def __init__(self):
        super().__init__(
            name="indian_identifiers_detector",
            supported_categories={
                "AADHAAR",
                "PAN",
                "PHONE_NUMBER",
                "IFSC",
                "BANK_ACCOUNT",
                "VEHICLE_REGISTRATION",
                "PASSPORT",
                "DRIVER_LICENSE",
            }
        )

        # Aadhaar: 12 digits, must not start with 0 or 1, optional spaces or dashes
        self.aadhaar_regex = re.compile(
            r"(?<!\d)(?:(?:[2-9]\d{3}\s\d{4}\s\d{4})|(?:[2-9]\d{3}-\d{4}-\d{4})|(?:[2-9]\d{11}))(?!\d)"
        )

        # PAN: 5 letters (4th is entity type), 4 digits, 1 letter
        self.pan_regex = re.compile(
            r"(?<![A-Za-z0-9])([A-Z]{3}[ABCFGHLJPTF][A-Z][0-9]{4}[A-Z])(?![A-Za-z0-9])"
        )

        # Indian Phone: (+91 or 0 or direct 10 digits starting with 6, 7, 8, 9)
        self.phone_regex = re.compile(
            r"(?<!\d)(?:(?:\+91[\s\-]?)|\b0)?([6-9]\d{4}[\s\-]?\d{5}|[6-9]\d{9})(?!\d)"
        )

        # IFSC: 4 alpha bank code, '0', 6 alphanumeric branch code
        self.ifsc_regex = re.compile(
            r"(?<![A-Za-z0-9])([A-Z]{4}0[A-Z0-9]{6})(?![A-Za-z0-9])"
        )

        # Vehicle Registration: e.g. DL-01-AB-1234, MH12DE1432, 22 BH 1234 AA
        self.vehicle_regex = re.compile(
            r"(?<![A-Za-z0-9])([A-Z]{2}[-\s]?[0-9]{1,2}[-\s]?[A-Z]{1,3}[-\s]?[0-9]{4}|[0-9]{2}[-\s]?BH[-\s]?[0-9]{4}[-\s]?[A-Z]{1,2})(?![A-Za-z0-9])"
        )

        # Bank Account: 9-18 digits preceded by context keywords
        self.bank_account_regex = re.compile(
            r"""(?i)(?:(?:account|acct|a/c|acc|khata|खाता|खाता संख्या)\s*(?:no\.?|num(?:ber)?|#)?\s*[:=\-]?\s*)([0-9]{9,18})\b"""
        )

        # Passport: 1 letter (A-PR-WY) + 7 digits with context or boundary
        self.passport_context_regex = re.compile(
            r"""(?i)(?:passport(?:\s*(?:no\.?|num(?:ber)?|#))?\s*[:=\-]?\s*)([A-PR-WY][0-9]{7})\b"""
        )
        self.passport_boundary_regex = re.compile(
            r"(?<![A-Za-z0-9])([A-PR-WY][0-9]{7})(?![A-Za-z0-9])"
        )

        # Driving License: State code + RTO + digits (e.g. DL-1420110012345 or DL14 20110012345)
        self.dl_regex = re.compile(
            r"(?<![A-Za-z0-9])([A-Z]{2}[-\s]?[0-9]{2}[-\s]?(?:19|20)[0-9]{2}[-\s]?[0-9]{7})(?![A-Za-z0-9])"
        )
        self.dl_context_regex = re.compile(
            r"""(?i)(?:driving\s*licen[sc]e|dl\s*no\.?|dl\s*#)\s*[:=\-]?\s*([A-Z]{2}[-\s]?[0-9]{2,15})\b"""
        )

    def detect(self, text: str) -> List[PIICandidate]:
        candidates: List[PIICandidate] = []

        # 1. Aadhaar detection with Verhoeff validation
        for match in self.aadhaar_regex.finditer(text):
            raw_match = match.group(0)
            digits_only = re.sub(r"[\s\-]", "", raw_match)
            if len(digits_only) == 12:
                # Verhoeff mathematical validation
                is_valid = validate_verhoeff(digits_only)
                # We report candidate if it passes Verhoeff checksum
                if is_valid:
                    candidates.append(
                        PIICandidate(
                            entity_type="AADHAAR",
                            start=match.start(),
                            end=match.end(),
                            text=raw_match,
                            confidence=0.99,
                            source="deterministic:aadhaar_verhoeff",
                            detector_name=self.name,
                            is_validated=True,
                            metadata={"format": "12_digit_verhoeff"},
                        )
                    )

        # 2. PAN detection with format validation
        for match in self.pan_regex.finditer(text):
            pan_val = match.group(1)
            # Verify 4th character is valid entity type
            entity_type_char = pan_val[3]
            is_valid = entity_type_char in "ABCFGHLJPTF"
            candidates.append(
                PIICandidate(
                    entity_type="PAN",
                    start=match.start(1),
                    end=match.end(1),
                    text=pan_val,
                    confidence=0.98 if is_valid else 0.80,
                    source="deterministic:pan_format",
                    detector_name=self.name,
                    is_validated=is_valid,
                    metadata={"entity_char": entity_type_char},
                )
            )

        # 3. IFSC detection
        for match in self.ifsc_regex.finditer(text):
            ifsc_val = match.group(1)
            bank_prefix = ifsc_val[:4]
            is_known = bank_prefix in KNOWN_BANK_PREFIXES
            candidates.append(
                PIICandidate(
                    entity_type="IFSC",
                    start=match.start(1),
                    end=match.end(1),
                    text=ifsc_val,
                    confidence=0.98 if is_known else 0.90,
                    source="deterministic:ifsc_format",
                    detector_name=self.name,
                    is_validated=True,
                    metadata={"bank_prefix": bank_prefix, "known_bank": is_known},
                )
            )

        # 4. Indian Phone Numbers
        for match in self.phone_regex.finditer(text):
            full_str = match.group(0)
            # Normalize digits
            digits = re.sub(r"\D", "", full_str)
            if digits.startswith("91") and len(digits) == 12:
                digits = digits[2:]
            elif digits.startswith("0") and len(digits) == 11:
                digits = digits[1:]

            if len(digits) == 10 and digits[0] in "6789":
                candidates.append(
                    PIICandidate(
                        entity_type="PHONE_NUMBER",
                        start=match.start(),
                        end=match.end(),
                        text=full_str,
                        confidence=0.95,
                        source="deterministic:indian_phone",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"country_code": "IN", "normalized_digits": digits},
                    )
                )

        # 5. Vehicle Registration
        for match in self.vehicle_regex.finditer(text):
            plate_val = match.group(1)
            clean_plate = re.sub(r"[\s\-]", "", plate_val).upper()
            state_prefix = clean_plate[:2]
            # Check if starts with a valid Indian state code or year+BH
            is_valid_state = state_prefix in INDIAN_STATE_CODES or "BH" in clean_plate[:4]
            if is_valid_state and len(clean_plate) >= 8:
                candidates.append(
                    PIICandidate(
                        entity_type="VEHICLE_REGISTRATION",
                        start=match.start(1),
                        end=match.end(1),
                        text=plate_val,
                        confidence=0.92,
                        source="deterministic:vehicle_registration",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"state_code": state_prefix},
                    )
                )

        # 6. Bank Account Numbers (contextual)
        for match in self.bank_account_regex.finditer(text):
            acc_num = match.group(1)
            start, end = match.span(1)
            # Verify length is typical (9 to 18 digits)
            if 9 <= len(acc_num) <= 18:
                candidates.append(
                    PIICandidate(
                        entity_type="BANK_ACCOUNT",
                        start=start,
                        end=end,
                        text=acc_num,
                        confidence=0.92,
                        source="deterministic:bank_account_context",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"length": len(acc_num)},
                    )
                )

        # 7. Passport
        for match in self.passport_context_regex.finditer(text):
            pass_num = match.group(1)
            candidates.append(
                PIICandidate(
                    entity_type="PASSPORT",
                    start=match.start(1),
                    end=match.end(1),
                    text=pass_num,
                    confidence=0.95,
                    source="deterministic:passport_context",
                    detector_name=self.name,
                    is_validated=True,
                    metadata={"country": "IN"},
                )
            )

        # 8. Driving License
        for match in self.dl_regex.finditer(text):
            dl_num = match.group(1)
            clean_dl = re.sub(r"[\s\-]", "", dl_num).upper()
            if clean_dl[:2] in INDIAN_STATE_CODES:
                candidates.append(
                    PIICandidate(
                        entity_type="DRIVER_LICENSE",
                        start=match.start(1),
                        end=match.end(1),
                        text=dl_num,
                        confidence=0.92,
                        source="deterministic:dl_format",
                        detector_name=self.name,
                        is_validated=True,
                        metadata={"state": clean_dl[:2]},
                    )
                )
        for match in self.dl_context_regex.finditer(text):
            dl_val = match.group(1)
            candidates.append(
                PIICandidate(
                    entity_type="DRIVER_LICENSE",
                    start=match.start(1),
                    end=match.end(1),
                    text=dl_val,
                    confidence=0.90,
                    source="deterministic:dl_context",
                    detector_name=self.name,
                    is_validated=True,
                )
            )

        return candidates
