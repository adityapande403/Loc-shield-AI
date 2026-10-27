"""PyTorch inference backend for reference and fallback contextual detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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


class PyTorchTokenClassificationBackend:
    """PyTorch reference backend for token classification."""

    def __init__(self, model_dir: str = "./models/localshield_multilingual"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, min_confidence: float = 0.5) -> List[PIICandidate]:
        if not text or not text.strip():
            return []

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].cpu().numpy()

        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)

        candidates: List[PIICandidate] = []
        current_cat: Optional[str] = None
        current_start: Optional[int] = None
        current_end: Optional[int] = None
        current_probs: List[float] = []

        for idx, (pred_id, (start_char, end_char)) in enumerate(zip(preds, offset_mapping)):
            if start_char == end_char == 0:
                continue

            tag_label = ID2LABEL.get(int(pred_id), "O")
            token_conf = float(probs[idx, pred_id])

            if tag_label == "O":
                if current_cat is not None:
                    self._flush(text, current_cat, current_start, current_end, current_probs, min_confidence, candidates)
                    current_cat = None
                continue

            prefix, tag_code = tag_label.split("-", 1)
            cat_name = TAG_TO_CATEGORY.get(tag_code, tag_code)

            if prefix == "B":
                if current_cat is not None:
                    self._flush(text, current_cat, current_start, current_end, current_probs, min_confidence, candidates)
                current_cat = cat_name
                current_start = int(start_char)
                current_end = int(end_char)
                current_probs = [token_conf]
            elif prefix == "I":
                if current_cat == cat_name:
                    current_end = int(end_char)
                    current_probs.append(token_conf)
                else:
                    if current_cat is not None:
                        self._flush(text, current_cat, current_start, current_end, current_probs, min_confidence, candidates)
                    current_cat = cat_name
                    current_start = int(start_char)
                    current_end = int(end_char)
                    current_probs = [token_conf]

        if current_cat is not None:
            self._flush(text, current_cat, current_start, current_end, current_probs, min_confidence, candidates)

        return candidates

    def _flush(self, text, category, start, end, probs, min_conf, out):
        if start is None or end is None or start >= end:
            return
        avg_conf = float(np.mean(probs)) if probs else 0.5
        if avg_conf < min_conf:
            return
        span_text = text[start:end].rstrip(".,;:!? ")
        end_adj = start + len(span_text)
        if start >= end_adj:
            return
        out.append(
            PIICandidate(
                entity_type=category,
                start=start,
                end=end_adj,
                text=text[start:end_adj],
                confidence=round(avg_conf, 4),
                source="ml:pytorch_cpu",
                detector_name="pytorch_token_classifier",
                is_validated=True,
                metadata={"backend": "pytorch"},
            )
        )
