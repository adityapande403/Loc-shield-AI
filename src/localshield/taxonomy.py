"""Centralized, versioned PII Taxonomy for LocalShield AI.

Defines all supported PII categories, sensitivity classifications, default redaction
policies, priority weights for conflict resolution, and validation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Set


class SensitivityLevel(str, Enum):
    """Sensitivity classification for data protection compliance."""
    CRITICAL = "CRITICAL"    # High financial/auth damage (Card, Aadhaar, API Keys, Tokens)
    HIGH = "HIGH"            # Direct PII (Passport, PAN, Bank Account, Phone, Email)
    MEDIUM = "MEDIUM"        # Quasi-identifiers (Person name, Address, Vehicle Registration)
    LOW = "LOW"              # Contextual (Organization, Location, DOB, Username)


class DetectionSourceType(str, Enum):
    """Origin mechanism for PII candidates."""
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CategoryDefinition:
    """Metadata describing a single PII category."""
    name: str
    description: str
    sensitivity: SensitivityLevel
    default_detection: bool
    default_redaction: bool
    priority_rank: int       # Higher rank wins in span conflict resolution
    canonical_placeholder: str
    source_type: DetectionSourceType
    validation_type: str     # e.g., 'verhoeff', 'luhn', 'checksum', 'entropy', 'context'


TAXONOMY_VERSION = "1.0.0"

# Centralized definitions for all required categories
PII_CATEGORIES: Dict[str, CategoryDefinition] = {
    "AADHAAR": CategoryDefinition(
        name="AADHAAR",
        description="Indian 12-digit unique identity number validated via Verhoeff checksum",
        sensitivity=SensitivityLevel.CRITICAL,
        default_detection=True,
        default_redaction=True,
        priority_rank=100,
        canonical_placeholder="AADHAAR",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="verhoeff",
    ),
    "CREDIT_CARD": CategoryDefinition(
        name="CREDIT_CARD",
        description="Payment card numbers (Visa, Mastercard, RuPay, Amex, etc.) validated with Luhn algorithm",
        sensitivity=SensitivityLevel.CRITICAL,
        default_detection=True,
        default_redaction=True,
        priority_rank=95,
        canonical_placeholder="CREDIT_CARD",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="luhn",
    ),
    "API_KEY": CategoryDefinition(
        name="API_KEY",
        description="Cloud provider keys, GitHub/Slack/OpenAI API keys, and high-entropy secrets",
        sensitivity=SensitivityLevel.CRITICAL,
        default_detection=True,
        default_redaction=True,
        priority_rank=90,
        canonical_placeholder="API_KEY",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="entropy_and_prefix",
    ),
    "ACCESS_TOKEN": CategoryDefinition(
        name="ACCESS_TOKEN",
        description="OAuth tokens, JWTs (JSON Web Tokens), bearer tokens",
        sensitivity=SensitivityLevel.CRITICAL,
        default_detection=True,
        default_redaction=True,
        priority_rank=88,
        canonical_placeholder="ACCESS_TOKEN",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="jwt_or_prefix",
    ),
    "AUTH_TOKEN": CategoryDefinition(
        name="AUTH_TOKEN",
        description="Session cookies, authentication secrets, private keys",
        sensitivity=SensitivityLevel.CRITICAL,
        default_detection=True,
        default_redaction=True,
        priority_rank=87,
        canonical_placeholder="AUTH_TOKEN",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="entropy_and_prefix",
    ),
    "PAN": CategoryDefinition(
        name="PAN",
        description="Indian Permanent Account Number (10 alphanumeric: 5 alpha, 4 numeric, 1 alpha)",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=85,
        canonical_placeholder="PAN",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="pan_format_rules",
    ),
    "PASSPORT": CategoryDefinition(
        name="PASSPORT",
        description="Indian & international passport numbers",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=80,
        canonical_placeholder="PASSPORT",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="passport_format",
    ),
    "DRIVER_LICENSE": CategoryDefinition(
        name="DRIVER_LICENSE",
        description="Driving license identifiers (Indian state-code format & international)",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=78,
        canonical_placeholder="DRIVER_LICENSE",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="dl_format",
    ),
    "BANK_ACCOUNT": CategoryDefinition(
        name="BANK_ACCOUNT",
        description="Bank account numbers detected with surrounding contextual account keywords",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=75,
        canonical_placeholder="BANK_ACCOUNT",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="contextual_numeric",
    ),
    "IFSC": CategoryDefinition(
        name="IFSC",
        description="Indian Financial System Code (11 characters: 4 bank code + '0' + 6 branch alphanumeric)",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=72,
        canonical_placeholder="IFSC",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="ifsc_format_and_bank_code",
    ),
    "EMAIL": CategoryDefinition(
        name="EMAIL",
        description="Email addresses adhering to standard RFC specifications",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=70,
        canonical_placeholder="EMAIL",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="rfc5322_regex",
    ),
    "PHONE_NUMBER": CategoryDefinition(
        name="PHONE_NUMBER",
        description="Indian (+91, 10 digits starting with 6-9) and international E.164 phone numbers",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=68,
        canonical_placeholder="PHONE_NUMBER",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="phone_format_rules",
    ),
    "IP_ADDRESS": CategoryDefinition(
        name="IP_ADDRESS",
        description="IPv4 and IPv6 network addresses with octet boundary validation",
        sensitivity=SensitivityLevel.MEDIUM,
        default_detection=True,
        default_redaction=True,
        priority_rank=65,
        canonical_placeholder="IP_ADDRESS",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="ip_range_check",
    ),
    "VEHICLE_REGISTRATION": CategoryDefinition(
        name="VEHICLE_REGISTRATION",
        description="Indian vehicle license plate numbers (e.g. DL-01-AB-1234, MH12DE1432, BH series)",
        sensitivity=SensitivityLevel.MEDIUM,
        default_detection=True,
        default_redaction=True,
        priority_rank=60,
        canonical_placeholder="VEHICLE_REGISTRATION",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="rto_state_code_format",
    ),
    "DATE_OF_BIRTH": CategoryDefinition(
        name="DATE_OF_BIRTH",
        description="Dates of birth identified by birth-specific context and valid date format",
        sensitivity=SensitivityLevel.MEDIUM,
        default_detection=True,
        default_redaction=True,
        priority_rank=58,
        canonical_placeholder="DATE_OF_BIRTH",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="dob_context_and_calendar",
    ),
    "PERSON": CategoryDefinition(
        name="PERSON",
        description="Names of individuals in English, Hindi (Devanagari), and Romanized/Code-mixed Hindi",
        sensitivity=SensitivityLevel.MEDIUM,
        default_detection=True,
        default_redaction=True,
        priority_rank=50,
        canonical_placeholder="PERSON",
        source_type=DetectionSourceType.MODEL,
        validation_type="ml_token_classifier",
    ),
    "ADDRESS": CategoryDefinition(
        name="ADDRESS",
        description="Physical and postal street addresses, landmarks, PIN code locations",
        sensitivity=SensitivityLevel.MEDIUM,
        default_detection=True,
        default_redaction=True,
        priority_rank=45,
        canonical_placeholder="ADDRESS",
        source_type=DetectionSourceType.HYBRID,
        validation_type="hybrid_context_postal",
    ),
    "ORGANIZATION": CategoryDefinition(
        name="ORGANIZATION",
        description="Company, corporate, agency, educational, or governmental body names",
        sensitivity=SensitivityLevel.LOW,
        default_detection=True,
        default_redaction=True,
        priority_rank=48,
        canonical_placeholder="ORGANIZATION",
        source_type=DetectionSourceType.MODEL,
        validation_type="ml_token_classifier",
    ),
    "LOCATION": CategoryDefinition(
        name="LOCATION",
        description="Geographical locations, cities, states, countries",
        sensitivity=SensitivityLevel.LOW,
        default_detection=True,
        default_redaction=True,
        priority_rank=40,
        canonical_placeholder="LOCATION",
        source_type=DetectionSourceType.MODEL,
        validation_type="ml_token_classifier",
    ),
    "USERNAME": CategoryDefinition(
        name="USERNAME",
        description="User handles (@handle), system logins, account usernames",
        sensitivity=SensitivityLevel.LOW,
        default_detection=True,
        default_redaction=True,
        priority_rank=55,
        canonical_placeholder="USERNAME",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="handle_or_context",
    ),
    "OTHER_SENSITIVE_ID": CategoryDefinition(
        name="OTHER_SENSITIVE_ID",
        description="Other high-entropy credentials or custom identifiers",
        sensitivity=SensitivityLevel.HIGH,
        default_detection=True,
        default_redaction=True,
        priority_rank=20,
        canonical_placeholder="OTHER_SENSITIVE_ID",
        source_type=DetectionSourceType.DETERMINISTIC,
        validation_type="generic_heuristic",
    ),
}

ALL_CATEGORIES: FrozenSet[str] = frozenset(PII_CATEGORIES.keys())
DEFAULT_DETECT_CATEGORIES: FrozenSet[str] = frozenset(
    k for k, v in PII_CATEGORIES.items() if v.default_detection
)
DEFAULT_REDACT_CATEGORIES: FrozenSet[str] = frozenset(
    k for k, v in PII_CATEGORIES.items() if v.default_redaction
)
CRITICAL_CATEGORIES: FrozenSet[str] = frozenset(
    k for k, v in PII_CATEGORIES.items() if v.sensitivity in (SensitivityLevel.CRITICAL, SensitivityLevel.HIGH)
)


def get_category_priority(category: str) -> int:
    """Return resolution priority rank for a given category (default 10)."""
    defn = PII_CATEGORIES.get(category.upper())
    return defn.priority_rank if defn else 10


def get_canonical_placeholder(category: str) -> str:
    """Return standard placeholder tag for category."""
    defn = PII_CATEGORIES.get(category.upper())
    return defn.canonical_placeholder if defn else category.upper()
