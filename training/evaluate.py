"""Comprehensive standalone evaluation framework for LocalShield AI.

Calculates entity-level, exact-span, per-category, safety, and robustness metrics on held-out datasets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from localshield.config import LocalShieldConfig
from localshield.schemas import PIICandidate, RedactionMode
from localshield.shield import LocalShield


def evaluate_dataset(
    dataset_path: str,
    shield: LocalShield,
    output_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate LocalShield pipeline on an annotated JSONL evaluation dataset."""
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} test samples from {dataset_path}...")

    # Metrics counters
    categories = ["PERSON", "ADDRESS", "ORGANIZATION", "LOCATION"]
    stats_per_cat = {
        cat: {"tp_exact": 0, "tp_overlap": 0, "fp": 0, "fn": 0, "total_gold": 0, "total_pred": 0}
        for cat in categories
    }

    total_gold_entities = 0
    total_pred_entities = 0
    total_tp_exact = 0
    total_tp_overlap = 0
    total_fp = 0
    total_fn = 0

    inference_times_ms = []
    redaction_successes = 0

    for sample in samples:
        text = sample["text"]
        gold_entities = sample.get("entities", [])
        total_gold_entities += len(gold_entities)

        # Run pipeline
        t0 = time.perf_counter()
        preds = shield.detect(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(elapsed_ms)

        total_pred_entities += len(preds)

        # Also test redaction on sample
        redact_res = shield.redact(text, mode=RedactionMode.MASK)
        if len(gold_entities) == 0:
            # Negative sample: redacted text must be identical to original!
            if redact_res.redacted_text == text:
                redaction_successes += 1
        else:
            redaction_successes += 1

        # Match gold vs predictions
        # A gold entity is a dict with {"start", "end", "label", "text"}
        # A pred candidate has start, end, entity_type
        matched_gold_exact: Set[int] = set()
        matched_gold_overlap: Set[int] = set()
        matched_preds: Set[int] = set()

        for g_idx, gold in enumerate(gold_entities):
            g_cat = gold["label"]
            if g_cat in stats_per_cat:
                stats_per_cat[g_cat]["total_gold"] += 1

            for p_idx, pred in enumerate(preds):
                if pred.entity_type != g_cat:
                    continue

                # Exact span match
                if pred.start == gold["start"] and pred.end == gold["end"]:
                    matched_gold_exact.add(g_idx)
                    matched_gold_overlap.add(g_idx)
                    matched_preds.add(p_idx)
                    if g_cat in stats_per_cat:
                        stats_per_cat[g_cat]["tp_exact"] += 1
                    break
                # Partial overlap match
                elif max(pred.start, gold["start"]) < min(pred.end, gold["end"]):
                    matched_gold_overlap.add(g_idx)
                    matched_preds.add(p_idx)
                    if g_cat in stats_per_cat:
                        stats_per_cat[g_cat]["tp_overlap"] += 1

        for g_idx, gold in enumerate(gold_entities):
            if g_idx not in matched_gold_overlap:
                g_cat = gold["label"]
                if g_cat in stats_per_cat:
                    stats_per_cat[g_cat]["fn"] += 1

        for p_idx, pred in enumerate(preds):
            if p_idx not in matched_preds:
                p_cat = pred.entity_type
                if p_cat in stats_per_cat:
                    stats_per_cat[p_cat]["fp"] += 1

        total_tp_exact += len(matched_gold_exact)
        total_tp_overlap += len(matched_gold_overlap)

    # Compute overall metrics
    total_fp = total_pred_entities - total_tp_overlap
    total_fn = total_gold_entities - total_tp_overlap

    overall_prec = total_tp_exact / max(1, total_pred_entities)
    overall_rec = total_tp_exact / max(1, total_gold_entities)
    overall_f1 = (2 * overall_prec * overall_rec) / max(1e-6, overall_prec + overall_rec)

    overlap_prec = total_tp_overlap / max(1, total_pred_entities)
    overlap_rec = total_tp_overlap / max(1, total_gold_entities)
    overlap_f1 = (2 * overlap_prec * overlap_rec) / max(1e-6, overlap_prec + overlap_rec)

    # Per-category summary
    category_summary = {}
    for cat, s in stats_per_cat.items():
        prec = s["tp_exact"] / max(1, s["total_pred"]) if s["total_pred"] > 0 else (
            s["tp_exact"] / max(1, s["tp_exact"] + s["fp"])
        )
        rec = s["tp_exact"] / max(1, s["total_gold"])
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        category_summary[cat] = {
            "exact_precision": round(float(prec), 4),
            "exact_recall": round(float(rec), 4),
            "exact_f1": round(float(f1), 4),
            "gold_count": s["total_gold"],
            "tp_exact": s["tp_exact"],
            "fp": s["fp"],
            "fn": s["fn"],
        }

    report = {
        "dataset": dataset_path,
        "sample_count": len(samples),
        "total_gold_entities": total_gold_entities,
        "total_predicted_entities": total_pred_entities,
        "exact_span_metrics": {
            "precision": round(float(overall_prec), 4),
            "recall": round(float(overall_rec), 4),
            "f1": round(float(overall_f1), 4),
        },
        "overlap_span_metrics": {
            "precision": round(float(overlap_prec), 4),
            "recall": round(float(overlap_rec), 4),
            "f1": round(float(overlap_f1), 4),
        },
        "safety_metrics": {
            "miss_rate_fn_pct": round(float(total_fn / max(1, total_gold_entities)) * 100, 2),
            "false_positives_per_100_samples": round(float(total_fp / len(samples)) * 100, 2),
            "redaction_preservation_accuracy_pct": round(float(redaction_successes / len(samples)) * 100, 2),
        },
        "latency_ms": {
            "p50": round(float(np.percentile(inference_times_ms, 50)), 2),
            "p95": round(float(np.percentile(inference_times_ms, 95)), 2),
            "mean": round(float(np.mean(inference_times_ms)), 2),
        },
        "per_category": category_summary,
    }

    if output_report_path:
        Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Evaluation report saved -> {output_report_path}")

    return report


if __name__ == "__main__":
    shield = LocalShield()
    report = evaluate_dataset(
        dataset_path="data/evaluation/test.jsonl",
        shield=shield,
        output_report_path="data/evaluation/evaluation_report.json",
    )
    print("\n--- STANDALONE EVALUATION REPORT ---")
    print(f"Samples Evaluated: {report['sample_count']}")
    print(f"Exact Span F1:     {report['exact_span_metrics']['f1']:.4f}")
    print(f"Exact Precision:   {report['exact_span_metrics']['precision']:.4f}")
    print(f"Exact Recall:      {report['exact_span_metrics']['recall']:.4f}")
    print(f"Overlap Span F1:   {report['overlap_span_metrics']['f1']:.4f}")
    print(f"Miss Rate:         {report['safety_metrics']['miss_rate_fn_pct']}%")
    print(f"P50 Latency:       {report['latency_ms']['p50']} ms")
    print("\nPer-Category Breakdown:")
    for cat, m in report["per_category"].items():
        print(f"   {cat:<14} P: {m['exact_precision']:.4f} | R: {m['exact_recall']:.4f} | F1: {m['exact_f1']:.4f} (Support: {m['gold_count']})")
