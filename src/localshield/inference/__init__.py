"""Inference backends for LocalShield AI."""

from localshield.inference.onnx_backend import ONNXTokenClassificationBackend
from localshield.inference.pytorch_backend import PyTorchTokenClassificationBackend

__all__ = ["ONNXTokenClassificationBackend", "PyTorchTokenClassificationBackend"]
