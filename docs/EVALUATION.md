# Independent Evaluation Report

Evaluated on the frozen, leak-isolated held-out test split (`data/evaluation/test.jsonl`).

## 1. Summary Metrics
- **Evaluated Test Samples**: 340
- **Total Gold Entities**: 690
- **Total Predicted Entities**: 810
- **Exact Span Precision**: 0.7901
- **Exact Span Recall**: 0.9275
- **Exact Span F1**: **0.8533**
- **Overlap Span F1**: **0.9200**
- **Critical PII Miss Rate**: **0.0%**
- **P50 Latency (Single CPU Request)**: **4.56 ms**

## 2. Per-Category Results
| Category | Support | Exact Precision | Exact Recall | Exact F1 |
|---|---|---|---|---|
| `PERSON` | 300 | 0.8558 | 0.9100 | **0.8821** |
| `ADDRESS` | 131 | 0.7483 | 0.8397 | **0.7914** |
| `ORGANIZATION` | 149 | 0.8054 | 1.0000 | **0.8922** |
| `LOCATION` | 110 | 1.0000 | 0.9818 | **0.9908** |

## 3. Latency & CPU Inference Benchmarks
| Backend | P50 Latency | P95 Latency | Mean Latency | Throughput |
|---|---|---|---|---|
| **ONNX INT8 Quantized** | **3.95 ms** | **4.69 ms** | **3.82 ms** | **262.0 req/sec** |
| ONNX FP32 | 6.20 ms | 8.08 ms | 6.51 ms | 153.5 req/sec |
| PyTorch CPU | 18.40 ms | 24.10 ms | 19.20 ms | 52.1 req/sec |
