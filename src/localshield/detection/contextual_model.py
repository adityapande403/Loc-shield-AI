"""Contextual ML Token Classification Detector wrapper."""

from __future__ import annotations

from typing import List, Optional, Set

from localshield.detection.base import BaseDetector
from localshield.inference.onnx_backend import ONNXTokenClassificationBackend
from localshield.inference.pytorch_backend import PyTorchTokenClassificationBackend
from localshield.schemas import PIICandidate


class ContextualModelDetector(BaseDetector):
    """Detector wrapping token-classification model for PERSON, ADDRESS, ORGANIZATION, and LOCATION."""

    def __init__(
        self,
        model_dir: str = "./models/localshield_multilingual",
        backend: str = "onnx",
        prefer_quantized: bool = True,
        num_threads: int = 4,
        min_confidence: float = 0.5,
    ):
        super().__init__(
            name="contextual_model_detector",
            supported_categories={"PERSON", "ADDRESS", "ORGANIZATION", "LOCATION"},
        )
        self.min_confidence = min_confidence
        self.backend_name = backend

        if backend == "onnx":
            self.backend = ONNXTokenClassificationBackend(
                model_dir=model_dir,
                prefer_quantized=prefer_quantized,
                num_threads=num_threads,
            )
        elif backend == "pytorch":
            self.backend = PyTorchTokenClassificationBackend(model_dir=model_dir)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Must be 'onnx' or 'pytorch'.")

    def detect(self, text: str) -> List[PIICandidate]:
        return self.backend.predict(text, min_confidence=self.min_confidence)
