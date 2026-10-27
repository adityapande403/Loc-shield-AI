"""LocalShield AI: Privacy-First Local PII Detection & Redaction Engine."""

from localshield.config import LocalShieldConfig
from localshield.schemas import (
    PIICandidate,
    RedactionMode,
    RedactionResult,
    RedactionSpan,
)
from localshield.shield import LocalShield
from localshield.taxonomy import ALL_CATEGORIES, TAXONOMY_VERSION

__version__ = "0.1.0"

__all__ = [
    "LocalShield",
    "LocalShieldConfig",
    "RedactionMode",
    "PIICandidate",
    "RedactionResult",
    "RedactionSpan",
    "ALL_CATEGORIES",
    "TAXONOMY_VERSION",
]
