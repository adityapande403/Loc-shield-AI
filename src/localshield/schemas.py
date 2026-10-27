"""Type definitions, request/response schemas, and candidate models for LocalShield AI."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator


class RedactionMode(str, Enum):
    """Supported redaction strategies."""
    MASK = "mask"                             # Mode A: [PERSON], [PHONE_NUMBER]
    PSEUDONYMIZE = "pseudonymize"             # Mode B: [PERSON_1], [PERSON_1] (consistent scope)
    TYPE_PRESERVING = "type_preserving"       # Mode C: [PHONE_NUMBER_1]
    SELECTIVE = "selective"                   # Mode D: Specific category selection & partial masking


class PIICandidate(BaseModel):
    """Represents a detected PII entity candidate prior to redaction."""
    entity_type: str = Field(..., description="Taxonomy category name (e.g. AADHAAR, PERSON)")
    start: int = Field(..., ge=0, description="Start character index in original raw text")
    end: int = Field(..., ge=0, description="End character index in original raw text")
    text: str = Field(..., description="Matched text span")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(..., description="Detection source identifier")
    detector_name: str = Field(default="deterministic", description="Module/rule name")
    is_validated: bool = Field(default=True, description="Whether mathematical or format validation passed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic/validation metadata")

    @field_validator("end")
    @classmethod
    def validate_span_bounds(cls, v: int, info) -> int:
        start = info.data.get("start")
        if start is not None and v < start:
            raise ValueError(f"Span end ({v}) cannot be less than start ({start})")
        return v

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return representation excluding raw sensitive text for privacy-preserving logs."""
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": self.source,
            "detector_name": self.detector_name,
            "is_validated": self.is_validated,
        }


class RedactionSpan(BaseModel):
    """Metadata describing a redacted character span."""
    entity_type: str
    original_start: int
    original_end: int
    redacted_start: int
    redacted_end: int
    placeholder: str
    confidence: float
    source: str
    is_validated: bool = True


class RedactionResult(BaseModel):
    """Complete output of a redaction operation. Note: raw input text is not logged."""
    redacted_text: str
    spans: List[RedactionSpan] = Field(default_factory=list)
    detected_count: int = 0
    redacted_count: int = 0
    processing_time_ms: float = 0.0
    mode: RedactionMode = RedactionMode.MASK
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# API Schemas
class DetectRequest(BaseModel):
    text: str = Field(..., max_length=1_000_000, description="Text to scan for PII")
    categories: Optional[List[str]] = Field(default=None, description="Optional category filter")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    entities: List[PIICandidate]
    total_detected: int
    processing_time_ms: float
    degraded_mode: bool = False


class RedactRequest(BaseModel):
    text: str = Field(..., max_length=1_000_000, description="Text to redact")
    categories: Optional[List[str]] = Field(default=None, description="Categories to redact")
    mode: RedactionMode = Field(default=RedactionMode.MASK, description="Redaction mode")
    enable_reversible: bool = Field(default=False, description="Enable reversible pseudonymization")
    session_id: Optional[str] = Field(default=None, description="Identifier for reversible session")
    partial_mask_rules: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Optional rules for partial masking (e.g. keep last 4 digits of card)"
    )


class RedactResponse(BaseModel):
    redacted_text: str
    spans: List[RedactionSpan]
    detected_count: int
    redacted_count: int
    processing_time_ms: float
    mode: str
    session_id: Optional[str] = None
    degraded_mode: bool = False


class BatchRedactRequest(BaseModel):
    texts: List[str] = Field(..., max_length=500, description="Batch of texts (max 500 items)")
    categories: Optional[List[str]] = None
    mode: RedactionMode = RedactionMode.MASK


class BatchRedactResponse(BaseModel):
    results: List[RedactResponse]
    total_processed: int
    total_processing_time_ms: float


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=1_000_000)


class CategoryStat(BaseModel):
    count: int
    confidence_avg: float
    validated_count: int


class AnalyzeResponse(BaseModel):
    total_candidates: int
    risk_score: float  # 0.0 (safe) to 1.0 (severe leak)
    category_breakdown: Dict[str, CategoryStat]
    candidates: List[PIICandidate]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    inference_backend: str
    version: str
    offline_mode: bool


class TaxonomyResponse(BaseModel):
    version: str
    total_categories: int
    categories: Dict[str, Dict[str, Any]]
