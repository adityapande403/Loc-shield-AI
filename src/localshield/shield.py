"""High-level LocalShield orchestrator, combining deterministic and ML contextual detection."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from localshield.config import LocalShieldConfig
from localshield.detection import CompositeDeterministicDetector
from localshield.detection.contextual_model import ContextualModelDetector
from localshield.pseudonymization import EncryptedMappingStore
from localshield.redaction import RedactionEngine
from localshield.resolution import SpanResolver
from localshield.schemas import (
    AnalyzeResponse,
    CategoryStat,
    HealthResponse,
    PIICandidate,
    RedactionMode,
    RedactionResult,
)
from localshield.taxonomy import (
    ALL_CATEGORIES,
    CRITICAL_CATEGORIES,
    DEFAULT_DETECT_CATEGORIES,
    DEFAULT_REDACT_CATEGORIES,
    TAXONOMY_VERSION,
)

logger = logging.getLogger("localshield")


class LocalShield:
    """Production-grade local PII detection and redaction engine.

    Integrates deterministic format validators (Verhoeff Aadhaar, Luhn Credit Card,
    Statutory PAN, IFSC, etc.) with a compact quantized multilingual token-classifier.
    """

    def __init__(
        self,
        config: Optional[LocalShieldConfig] = None,
        deterministic_detector: Optional[CompositeDeterministicDetector] = None,
        contextual_detector: Optional[ContextualModelDetector] = None,
        resolver: Optional[SpanResolver] = None,
        redaction_engine: Optional[RedactionEngine] = None,
    ):
        self.config = config or LocalShieldConfig.from_env()
        self.deterministic_detector = (
            deterministic_detector
            or CompositeDeterministicDetector(entropy_threshold=self.config.api_key_entropy_threshold)
        )
        self.resolver = resolver or SpanResolver()

        # Initialize encrypted storage for reversible pseudonymization
        mapping_store = EncryptedMappingStore(
            storage_dir=self.config.storage_dir,
            encryption_key=self.config.reversible_key.encode("utf-8") if self.config.reversible_key else None,
            ttl_seconds=self.config.reversible_ttl_seconds,
        )
        self.redaction_engine = redaction_engine or RedactionEngine(mapping_store=mapping_store)

        # Contextual model initialization with degraded mode fallback
        self.contextual_detector = contextual_detector
        self.model_loaded = False
        self.degraded_mode = False

        if self.contextual_detector is None and self.config.inference_backend != "none":
            model_path = Path(self.config.model_path)
            if model_path.exists():
                try:
                    self.contextual_detector = ContextualModelDetector(
                        model_dir=str(model_path),
                        backend=self.config.inference_backend,
                        prefer_quantized=self.config.prefer_quantized,
                        num_threads=self.config.onnx_num_threads,
                        min_confidence=self.config.min_confidence,
                    )
                    self.model_loaded = True
                except Exception as e:
                    logger.warning(
                        f"Contextual model initialization failed: {e}. Falling back to deterministic degraded mode."
                    )
                    self.degraded_mode = True
            else:
                self.degraded_mode = True

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./models/localshield_multilingual",
        backend: str = "onnx",
        prefer_quantized: bool = True,
        offline: bool = True,
    ) -> LocalShield:
        """Create a LocalShield instance with default production settings."""
        cfg = LocalShieldConfig(
            model_path=model_path,
            inference_backend=backend,
            prefer_quantized=prefer_quantized,
            offline_mode=offline,
        )
        return cls(config=cfg)

    def detect(
        self,
        text: str,
        categories: Optional[Set[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> List[PIICandidate]:
        """Scan text and return resolved non-overlapping PII candidates with character offsets."""
        if not text:
            return []

        if len(text) > self.config.max_input_chars:
            raise ValueError(
                f"Input text length ({len(text)}) exceeds maximum configured limit ({self.config.max_input_chars})"
            )

        min_conf = min_confidence if min_confidence is not None else self.config.min_confidence
        raw_candidates: List[PIICandidate] = []

        # 1. Run deterministic detectors (Indian, International, Secrets)
        raw_candidates.extend(self.deterministic_detector.detect(text, categories=categories))

        # 2. Run contextual ML model if available
        if self.model_loaded and self.contextual_detector:
            try:
                ml_candidates = self.contextual_detector.detect(text)
                for c in ml_candidates:
                    if categories is None or c.entity_type in categories:
                        if c.confidence >= min_conf:
                            raw_candidates.append(c)
            except Exception as e:
                logger.error(f"Error during contextual model prediction: {e}")

        # 3. Resolve conflicts, overlaps, and priority nesting
        resolved = self.resolver.resolve(raw_candidates, text_length=len(text))
        return resolved

    def redact(
        self,
        text: str,
        categories: Optional[Set[str]] = None,
        mode: RedactionMode = RedactionMode.MASK,
        enable_reversible: bool = False,
        session_id: Optional[str] = None,
        partial_mask_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> RedactionResult:
        """Detect and redact sensitive spans in text using the configured mode."""
        candidates = self.detect(text, categories=categories)
        target_cats = categories if categories is not None else self.config.redact_categories

        result = self.redaction_engine.redact(
            text=text,
            candidates=candidates,
            categories=target_cats,
            mode=mode,
            enable_reversible=enable_reversible,
            session_id=session_id,
            partial_mask_rules=partial_mask_rules,
        )
        if self.degraded_mode:
            result.metadata["degraded_mode"] = True
            result.metadata["warning"] = "Operating in degraded mode without contextual ML model"

        return result

    def redact_batch(
        self,
        texts: List[str],
        categories: Optional[Set[str]] = None,
        mode: RedactionMode = RedactionMode.MASK,
    ) -> List[RedactionResult]:
        """Process a batch of texts for PII redaction."""
        if len(texts) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum allowed limit {self.config.max_batch_size}"
            )
        return [self.redact(t, categories=categories, mode=mode) for t in texts]

    def analyze(self, text: str) -> AnalyzeResponse:
        """Analyze text and provide risk score and breakdown by PII category."""
        start_time = time.perf_counter()
        candidates = self.detect(text)

        category_stats: Dict[str, Dict[str, Any]] = {}
        has_critical = False

        for c in candidates:
            cat = c.entity_type
            if cat in CRITICAL_CATEGORIES:
                has_critical = True

            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "confidences": [], "validated": 0}

            category_stats[cat]["count"] += 1
            category_stats[cat]["confidences"].append(c.confidence)
            if c.is_validated:
                category_stats[cat]["validated"] += 1

        formatted_stats = {
            k: CategoryStat(
                count=v["count"],
                confidence_avg=round(float(sum(v["confidences"]) / len(v["confidences"])), 3),
                validated_count=v["validated"],
            )
            for k, v in category_stats.items()
        }

        # Calculate composite risk score
        risk_score = 0.0
        if candidates:
            base_risk = min(1.0, len(candidates) * 0.15)
            if has_critical:
                base_risk = max(base_risk, 0.85)
            risk_score = round(base_risk, 2)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return AnalyzeResponse(
            total_candidates=len(candidates),
            risk_score=risk_score,
            category_breakdown=formatted_stats,
            candidates=candidates,
            processing_time_ms=elapsed_ms,
        )

    def health(self) -> HealthResponse:
        """Return engine operational status, backend details, and degradation flags."""
        return HealthResponse(
            status="healthy" if not self.degraded_mode else "degraded",
            model_loaded=self.model_loaded,
            model_name=str(self.config.model_path) if self.model_loaded else None,
            inference_backend=self.config.inference_backend,
            version="0.1.0",
            offline_mode=self.config.offline_mode,
        )
