"""Configuration management for LocalShield AI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field

from localshield.taxonomy import ALL_CATEGORIES, DEFAULT_REDACT_CATEGORIES


class LocalShieldConfig(BaseModel):
    """Runtime configuration for LocalShield engine and service."""

    # Backend & Model configuration
    model_path: Optional[str] = Field(
        default="./models/localshield_multilingual",
        description="Path to pretrained/finetuned contextual model directory"
    )
    inference_backend: str = Field(
        default="onnx",
        description="Inference backend: 'onnx', 'pytorch', or 'none' (deterministic only)"
    )
    prefer_quantized: bool = Field(
        default=True,
        description="Prefer INT8 dynamic quantized model if available"
    )
    onnx_num_threads: int = Field(
        default=4,
        description="Number of CPU threads for ONNX Runtime session"
    )

    # Detection & Redaction defaults
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled_categories: Set[str] = Field(default_factory=lambda: set(ALL_CATEGORIES))
    redact_categories: Set[str] = Field(default_factory=lambda: set(DEFAULT_REDACT_CATEGORIES))
    max_input_chars: int = Field(default=1_000_000, description="Max allowed characters per text")
    max_batch_size: int = Field(default=500, description="Max texts in a batch request")

    # Entropy & Heuristics
    api_key_entropy_threshold: float = Field(
        default=3.5,
        description="Shannon entropy threshold for candidate secret detection"
    )

    # Privacy & Security
    offline_mode: bool = Field(
        default=True,
        description="Strict offline enforcement; prevents any runtime network calls"
    )
    log_raw_pii: bool = Field(
        default=False,
        description="Never log raw PII under any circumstance"
    )
    allow_telemetry: bool = Field(
        default=False,
        description="Telemetry disabled by default"
    )

    # Reversible storage configuration
    storage_dir: str = Field(
        default="./localshield_storage",
        description="Local directory for encrypted mapping storage"
    )
    reversible_key: Optional[str] = Field(
        default=None,
        description="Base64 32-byte key for AES-GCM/Fernet encryption; can be loaded via env"
    )
    reversible_ttl_seconds: int = Field(
        default=86400,
        description="Time-to-live for stored reversible mappings"
    )

    # Service / API config
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    api_auth_token: Optional[str] = Field(default=None)

    @classmethod
    def from_env(cls) -> LocalShieldConfig:
        """Instantiate configuration from environment variables."""
        return cls(
            model_path=os.getenv("LOCALSHIELD_MODEL_PATH", "./models/localshield_multilingual"),
            inference_backend=os.getenv("LOCALSHIELD_BACKEND", "onnx"),
            prefer_quantized=os.getenv("LOCALSHIELD_PREFER_QUANTIZED", "1").lower() in ("1", "true", "yes"),
            onnx_num_threads=int(os.getenv("LOCALSHIELD_ONNX_THREADS", "4")),
            min_confidence=float(os.getenv("LOCALSHIELD_MIN_CONFIDENCE", "0.5")),
            max_input_chars=int(os.getenv("LOCALSHIELD_MAX_CHARS", "1000000")),
            offline_mode=os.getenv("LOCALSHIELD_OFFLINE", "1").lower() in ("1", "true", "yes"),
            api_host=os.getenv("LOCALSHIELD_HOST", "127.0.0.1"),
            api_port=int(os.getenv("LOCALSHIELD_PORT", "8000")),
            api_auth_token=os.getenv("LOCALSHIELD_AUTH_TOKEN"),
            reversible_key=os.getenv("LOCALSHIELD_REVERSIBLE_KEY"),
        )
