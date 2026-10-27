"""Fine-tune multilingual token-classification encoder (DistilBERT) for contextual PII detection.

Aligns token offset mappings with BIO entity labels, handles subword masking, trains with
AdamW and linear learning rate schedule, evaluates per-category F1, and saves model checkpoints.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ADDR", "I-ADDR",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}

ENTITY_TO_TAG = {
    "PERSON": "PER",
    "ADDRESS": "ADDR",
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
}


def get_optimal_device() -> torch.device:
    """Detect available accelerator and verify hardware kernel support."""
    if torch.cuda.is_available():
        try:
            # Verify if kernel image is available for architecture (e.g. Blackwell sm_120)
            t = torch.zeros((1,), device="cuda")
            _ = t + 1
            print(f"CUDA accelerator verified: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            print(f"Notice: GPU detected ({torch.cuda.get_device_name(0)}), but PyTorch lacks sm_120 kernels.")
            print("Using high-performance multi-threaded CPU engine.")
    
    threads = min(16, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    print(f"Using CPU engine with {threads} parallel worker threads.")
    return torch.device("cpu")


class TokenClassificationDataset(Dataset):
    """Dataset with exact character-to-subword BIO label alignment."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 64):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        entities = sample.get("entities", [])

        # Tokenize with offsets
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Build BIO labels
        labels = torch.full((self.max_length,), -100, dtype=torch.long)

        # Precompute character-to-entity map
        char_tags = [None] * len(text)
        for ent in entities:
            tag = ENTITY_TO_TAG.get(ent["label"])
            if not tag:
                continue
            s, e = ent["start"], ent["end"]
            for c_idx in range(s, min(e, len(text))):
                char_tags[c_idx] = (tag, c_idx == s)

        # Align subwords
        prev_tag = None
        for i in range(self.max_length):
            if attention_mask[i] == 0:
                continue

            start_char, end_char = offset_mapping[i].tolist()
            if start_char == end_char == 0:
                continue

            matched_tag = None
            is_first = False
            for c in range(start_char, end_char):
                if c < len(char_tags) and char_tags[c] is not None:
                    matched_tag, is_first = char_tags[c]
                    break

            if matched_tag:
                if is_first or prev_tag != matched_tag:
                    labels[i] = LABEL2ID[f"B-{matched_tag}"]
                else:
                    labels[i] = LABEL2ID[f"I-{matched_tag}"]
                prev_tag = matched_tag
            else:
                labels[i] = LABEL2ID["O"]
                prev_tag = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate(model, dataloader, device) -> Dict[str, Any]:
    """Calculate token-level and category-level precision, recall, and F1."""
    model.eval()
    total_loss = 0.0

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Collect active labels (excluding -100)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    lbl_id = labels[i, j].item()
                    if lbl_id != -100:
                        true_labels.append(lbl_id)
                        pred_labels.append(preds[i, j].item())

    avg_loss = total_loss / max(1, len(dataloader))

    # Calculate metrics
    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)

    # Per-category metrics
    category_metrics = {}
    for tag_name, tag_code in [("PERSON", "PER"), ("ADDRESS", "ADDR"), ("ORGANIZATION", "ORG"), ("LOCATION", "LOC")]:
        b_id = LABEL2ID[f"B-{tag_code}"]
        i_id = LABEL2ID[f"I-{tag_code}"]
        
        relevant_true = (true_arr == b_id) | (true_arr == i_id)
        relevant_pred = (pred_arr == b_id) | (pred_arr == i_id)

        tp = np.sum(relevant_true & relevant_pred)
        fp = np.sum((~relevant_true) & relevant_pred)
        fn = np.sum(relevant_true & (~relevant_pred))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        category_metrics[tag_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(relevant_true)),
        }

    macro_f1 = np.mean([m["f1"] for m in category_metrics.values()])

    return {
        "val_loss": round(avg_loss, 4),
        "macro_f1": round(float(macro_f1), 4),
        "categories": category_metrics,
    }


def main():
    model_name = "distilbert-base-multilingual-cased"
    output_dir = Path("models/localshield_multilingual")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Building datasets (max_length=64)...")
    train_dataset = TokenClassificationDataset("data/processed/train.jsonl", tokenizer, max_length=64)
    val_dataset = TokenClassificationDataset("data/processed/val.jsonl", tokenizer, max_length=64)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = get_optimal_device()

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Starting fine-tuning: {epochs} epochs, {total_steps} total steps (batch_size=32)...")
    best_val_f1 = -1.0
    training_history = []

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            train_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"Evaluating Epoch {epoch} on validation set...")
        val_metrics = evaluate(model, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_metrics["val_loss"],
            "macro_f1": val_metrics["macro_f1"],
            "categories": val_metrics["categories"],
        }
        training_history.append(epoch_record)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f}")
        for cat, m in val_metrics["categories"].items():
            print(f"   - {cat:<14} Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | F1: {m['f1']:.4f}")

        # Save best checkpoint
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            print(f"--> New best model (Macro F1 {best_val_f1:.4f}). Saving checkpoint to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    total_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_time:.1f} seconds. Best Val Macro F1: {best_val_f1:.4f}")

    # Save training report
    report = {
        "model_architecture": "DistilBertForTokenClassification",
        "base_model": model_name,
        "epochs": epochs,
        "best_val_macro_f1": best_val_f1,
        "total_training_time_seconds": round(total_time, 2),
        "history": training_history,
        "labels": LABEL_LIST,
    }
    report_path = output_dir / "training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Training metrics saved -> {report_path}")


if __name__ == "__main__":
    main()
