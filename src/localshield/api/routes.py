"""FastAPI route definitions for detection, redaction, batch, analyze, and diagnostics."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from localshield.api.security import verify_api_key
from localshield.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRedactRequest,
    BatchRedactResponse,
    DetectRequest,
    DetectResponse,
    HealthResponse,
    RedactRequest,
    RedactResponse,
    TaxonomyResponse,
)
from localshield.shield import LocalShield
from localshield.taxonomy import PII_CATEGORIES, TAXONOMY_VERSION

router = APIRouter(dependencies=[Depends(verify_api_key)])


def get_shield() -> LocalShield:
    """Dependency provider for LocalShield singleton."""
    global _GLOBAL_SHIELD
    if _GLOBAL_SHIELD is None:
        _GLOBAL_SHIELD = LocalShield()
    return _GLOBAL_SHIELD


_GLOBAL_SHIELD: LocalShield = None


def set_shield_instance(shield: LocalShield):
    global _GLOBAL_SHIELD
    _GLOBAL_SHIELD = shield


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe: verifies service process is running."""
    shield = get_shield()
    return shield.health()


@router.get("/ready")
async def readiness_check():
    """Readiness probe: verifies model and deterministic engines are operational."""
    shield = get_shield()
    if not shield.model_loaded and not shield.degraded_mode:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LocalShield engine is initializing or not ready.",
        )
    return {
        "ready": True,
        "model_loaded": shield.model_loaded,
        "degraded_mode": shield.degraded_mode,
        "backend": shield.config.inference_backend,
    }


@router.get("/version")
async def get_version():
    return {
        "service": "LocalShield AI",
        "version": "0.1.0",
        "taxonomy_version": TAXONOMY_VERSION,
    }


@router.get("/taxonomy", response_model=TaxonomyResponse)
async def get_taxonomy():
    """Return versioned PII taxonomy and category definitions."""
    cats_dict = {
        name: {
            "description": defn.description,
            "sensitivity": defn.sensitivity.value,
            "default_detection": defn.default_detection,
            "default_redaction": defn.default_redaction,
            "priority_rank": defn.priority_rank,
            "canonical_placeholder": defn.canonical_placeholder,
            "validation_type": defn.validation_type,
        }
        for name, defn in PII_CATEGORIES.items()
    }
    return TaxonomyResponse(
        version=TAXONOMY_VERSION,
        total_categories=len(cats_dict),
        categories=cats_dict,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_pii(payload: DetectRequest):
    """Detect PII entities without modifying input text."""
    start_time = time.perf_counter()
    shield = get_shield()

    cat_set = set(payload.categories) if payload.categories else None
    entities = shield.detect(
        text=payload.text,
        categories=cat_set,
        min_confidence=payload.min_confidence,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return DetectResponse(
        entities=entities,
        total_detected=len(entities),
        processing_time_ms=round(elapsed_ms, 2),
        degraded_mode=shield.degraded_mode,
    )


@router.post("/redact", response_model=RedactResponse)
async def redact_pii(payload: RedactRequest):
    """Detect and redact sensitive text using the requested mode."""
    shield = get_shield()
    cat_set = set(payload.categories) if payload.categories else None

    result = shield.redact(
        text=payload.text,
        categories=cat_set,
        mode=payload.mode,
        enable_reversible=payload.enable_reversible,
        session_id=payload.session_id,
        partial_mask_rules=payload.partial_mask_rules,
    )

    return RedactResponse(
        redacted_text=result.redacted_text,
        spans=result.spans,
        detected_count=result.detected_count,
        redacted_count=result.redacted_count,
        processing_time_ms=round(result.processing_time_ms, 2),
        mode=result.mode.value,
        session_id=result.session_id,
        degraded_mode=shield.degraded_mode,
    )


@router.post("/redact/batch", response_model=BatchRedactResponse)
async def redact_pii_batch(payload: BatchRedactRequest):
    """Redact multiple text items in a single request."""
    start_time = time.perf_counter()
    shield = get_shield()
    cat_set = set(payload.categories) if payload.categories else None

    results = shield.redact_batch(
        texts=payload.texts,
        categories=cat_set,
        mode=payload.mode,
    )

    formatted_responses = [
        RedactResponse(
            redacted_text=r.redacted_text,
            spans=r.spans,
            detected_count=r.detected_count,
            redacted_count=r.redacted_count,
            processing_time_ms=round(r.processing_time_ms, 2),
            mode=r.mode.value,
            session_id=r.session_id,
            degraded_mode=shield.degraded_mode,
        )
        for r in results
    ]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return BatchRedactResponse(
        results=formatted_responses,
        total_processed=len(results),
        total_processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    """Analyze text for sensitive data leakage risk score and category counts."""
    shield = get_shield()
    return shield.analyze(payload.text)
