"""ONNX Runtime CPU inference backend with BIO span extraction and offset mapping."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from localshield.schemas import PIICandidate

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ADDR", "I-ADDR",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}
TAG_TO_CATEGORY = {
    "PER": "PERSON",
    "ADDR": "ADDRESS",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
}


class ONNXTokenClassificationBackend:
    """High-performance portable CPU inference session using ONNX Runtime."""

    def __init__(
        self,
        model_dir: str = "./models/localshield_multilingual",
        prefer_quantized: bool = True,
        num_threads: int = 4,
    ):
        self.model_dir = Path(model_dir)
        self.prefer_quantized = prefer_quantized
        self.num_threads = num_threads

        # Locate model file
        int8_file = self.model_dir / "model_quantized.onnx"
        fp32_file = self.model_dir / "model.onnx"

        if prefer_quantized and int8_file.exists():
            self.onnx_path = int8_file
            self.is_quantized = True
        elif fp32_file.exists():
            self.onnx_path = fp32_file
            self.is_quantized = False
        else:
            raise FileNotFoundError(
                f"No ONNX model found in {model_dir}. Run training/export_onnx.py first."
            )

        # Initialize session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

    def predict(self, text: str, min_confidence: float = 0.5) -> List[PIICandidate]:
        """Run token classification on text and return BIO decoded PIICandidate entities."""
        if not text or not text.strip():
            return []

        # Tokenize with offsets
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"][0]

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Run ONNX inference
        logits = self.session.run(None, ort_inputs)[0]  # shape: (1, seq_len, num_labels)
        probs = self._softmax(logits[0], axis=-1)
        preds = np.argmax(probs, axis=-1)

        # BIO Span Decoding
        candidates: List[PIICandidate] = []
        current_cat: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_probs: List[float] = []

        for idx, (pred_id, (start_char, end_char)) in enumerate(zip(preds, offset_mapping)):
            # Skip special tokens
            if start_char == end_char == 0:
                continue

            tag_label = ID2LABEL.get(int(pred_id), "O")
            token_conf = float(probs[idx, pred_id])

            if tag_label == "O":
                if current_cat is not None:
                    # Flush span
                    self._flush_span(
                        text, current_cat, current_start, current_end, current_probs, min_confidence, candidates
                    )
                    current_cat = None
                continue

            prefix, tag_code = tag_label.split("-", 1)
            cat_name = TAG_TO_CATEGORY.get(tag_code, tag_code)

            if prefix == "B":
                if current_cat is not None:
                    self._flush_span(
                        text, current_cat, current_start, current_end, current_probs, min_confidence, candidates
                    )
                current_cat = cat_name
                current_start = int(start_char)
                current_end = int(end_char)
                current_probs = [token_conf]

            elif prefix == "I":
                if current_cat == cat_name:
                    # Extend span
                    current_end = int(end_char)
                    current_probs.append(token_conf)
                else:
                    # Mismatched I-tag, treat as new span start
                    if current_cat is not None:
                        self._flush_span(
                            text, current_cat, current_start, current_end, current_probs, min_confidence, candidates
                        )
                    current_cat = cat_name
                    current_start = int(start_char)
                    current_end = int(end_char)
                    current_probs = [token_conf]

        if current_cat is not None:
            self._flush_span(
                text, current_cat, current_start, current_end, current_probs, min_confidence, candidates
            )

        return candidates

    def _flush_span(
        self,
        text: str,
        category: str,
        start: Optional[int],
        end: Optional[int],
        probs: List[float],
        min_conf: float,
        out_list: List[PIICandidate],
    ):
        if start is None or end is None or start >= end:
            return

        avg_conf = float(np.mean(probs)) if probs else 0.5
        if avg_conf < min_conf:
            return

        span_text = text[start:end]
        # Clean trailing punctuation from entity span
        stripped_text = span_text.rstrip(".,;:!? ")
        end_adjusted = start + len(stripped_text)
        if start >= end_adjusted:
            return

        out_list.append(
            PIICandidate(
                entity_type=category,
                start=start,
                end=end_adjusted,
                text=text[start:end_adjusted],
                confidence=round(avg_conf, 4),
                source="ml:onnx_quantized" if self.is_quantized else "ml:onnx_fp32",
                detector_name="onnx_token_classifier",
                is_validated=True,
                metadata={"backend": "onnx", "quantized": self.is_quantized},
            )
        )

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
