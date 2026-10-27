"""Export trained PyTorch model to ONNX FP32 and INT8 Dynamic Quantization.

Includes numerical parity verification, prediction agreement benchmarks, and CPU latency measurements.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer

LABEL_LIST = [
    "O",
    "B-PER", "I-PER",
    "B-ADDR", "I-ADDR",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


def export_and_quantize():
    model_dir = Path("models/localshield_multilingual")
    onnx_fp32_path = model_dir / "model.onnx"
    onnx_int8_path = model_dir / "model_quantized.onnx"

    print("Loading PyTorch model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    # 1. Export to ONNX FP32
    print(f"Exporting to ONNX FP32 -> {onnx_fp32_path}...")
    dummy_text = "Rahul Sharma called HDFC Bank in Mumbai."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    dummy_input_ids = inputs["input_ids"]
    dummy_attention_mask = inputs["attention_mask"]

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(onnx_fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Validate ONNX graph
    onnx_model = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model checker: PASSED!")

    # 2. Dynamic INT8 Quantization
    print(f"Quantizing to INT8 -> {onnx_int8_path}...")
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    print("Quantization: COMPLETE!")

    # 3. Model Sizes
    pt_size_mb = (model_dir / "model.safetensors").stat().st_size / (1024 * 1024)
    onnx_fp32_mb = onnx_fp32_path.stat().st_size / (1024 * 1024)
    onnx_int8_mb = onnx_int8_path.stat().st_size / (1024 * 1024)

    print(f"Model Sizes:")
    print(f"   PyTorch SafeTensors: {pt_size_mb:.2f} MB")
    print(f"   ONNX FP32:           {onnx_fp32_mb:.2f} MB")
    print(f"   ONNX INT8 Quantized: {onnx_int8_mb:.2f} MB ({100 * (1 - onnx_int8_mb / pt_size_mb):.1f}% reduction)")

    # 4. Parity Testing
    print("Running numerical and prediction parity verification...")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(str(onnx_fp32_path), sess_options, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(onnx_int8_path), sess_options, providers=["CPUExecutionProvider"])

    test_samples = [
        "Customer Rahul Sharma reported an issue from Bengaluru regarding account at Infosys.",
        "Mera naam Amit Kumar hai aur main Mumbai me rehta hu.",
        "कृपया Rohan Joshi ko HDFC Bank New Delhi bhejo.",
        "Flat 402, Shanti Heights, Sector 62, Noida is the registered address.",
        "All systems are functioning normally with no PII present.",
    ]

    total_tokens = 0
    fp32_disagreements = 0
    int8_disagreements = 0
    max_fp32_abs_diff = 0.0

    for sample in test_samples:
        enc = tokenizer(sample, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            pt_out = model(input_ids=input_ids, attention_mask=attention_mask)
            pt_logits = pt_out.logits.numpy()
            pt_preds = np.argmax(pt_logits, axis=-1)[0]

        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        }
        fp32_logits = sess_fp32.run(None, ort_inputs)[0]
        fp32_preds = np.argmax(fp32_logits, axis=-1)[0]

        int8_logits = sess_int8.run(None, ort_inputs)[0]
        int8_preds = np.argmax(int8_logits, axis=-1)[0]

        # Differences
        abs_diff = np.max(np.abs(pt_logits - fp32_logits))
        max_fp32_abs_diff = max(max_fp32_abs_diff, float(abs_diff))

        seq_len = len(pt_preds)
        total_tokens += seq_len
        fp32_disagreements += int(np.sum(pt_preds != fp32_preds))
        int8_disagreements += int(np.sum(pt_preds != int8_preds))

    fp32_disagree_rate = fp32_disagreements / max(1, total_tokens)
    int8_disagree_rate = int8_disagreements / max(1, total_tokens)

    print(f"Parity Results across test suite ({total_tokens} tokens):")
    print(f"   Max FP32 Logits Abs Diff: {max_fp32_abs_diff:.6f}")
    print(f"   ONNX FP32 Disagreement Rate: {fp32_disagree_rate * 100:.2f}%")
    print(f"   ONNX INT8 Disagreement Rate: {int8_disagree_rate * 100:.2f}%")

    # 5. Latency Benchmarks (CPU)
    print("Benchmarking CPU inference latency...")
    benchmark_text = "Customer Rahul Sharma reported from Bengaluru regarding account at Infosys."
    enc = tokenizer(benchmark_text, return_tensors="np")
    ort_inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    # Warmup
    for _ in range(10):
        sess_fp32.run(None, ort_inputs)
        sess_int8.run(None, ort_inputs)

    def measure_latency(sess, inputs, iters=50) -> Dict[str, float]:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run(None, inputs)
            times.append((time.perf_counter() - t0) * 1000.0)
        return {
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p95_ms": round(float(np.percentile(times, 95)), 2),
            "p99_ms": round(float(np.percentile(times, 99)), 2),
            "mean_ms": round(float(np.mean(times)), 2),
            "throughput_req_per_sec": round(1000.0 / float(np.mean(times)), 1),
        }

    fp32_lat = measure_latency(sess_fp32, ort_inputs)
    int8_lat = measure_latency(sess_int8, ort_inputs)

    print(f"CPU Latency (Single Request):")
    print(f"   ONNX FP32: P50={fp32_lat['p50_ms']}ms | P95={fp32_lat['p95_ms']}ms | Throughput={fp32_lat['throughput_req_per_sec']} req/s")
    print(f"   ONNX INT8: P50={int8_lat['p50_ms']}ms | P95={int8_lat['p95_ms']}ms | Throughput={int8_lat['throughput_req_per_sec']} req/s")

    report = {
        "model_sizes_mb": {
            "pytorch_safetensors": round(pt_size_mb, 2),
            "onnx_fp32": round(onnx_fp32_mb, 2),
            "onnx_int8": round(onnx_int8_mb, 2),
            "compression_ratio": round(pt_size_mb / onnx_int8_mb, 2),
        },
        "parity": {
            "max_fp32_abs_diff": max_fp32_abs_diff,
            "fp32_token_disagreement_rate": fp32_disagree_rate,
            "int8_token_disagreement_rate": int8_disagree_rate,
            "parity_verified": fp32_disagree_rate == 0.0,
        },
        "latency_cpu_ms": {
            "fp32": fp32_lat,
            "int8": int8_lat,
        },
    }

    report_path = model_dir / "onnx_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    export_and_quantize()
