# Model Card: LocalShield Multilingual Token Classifier

## 1. Model Details
- **Architecture**: `DistilBertForTokenClassification`
- **Base Model**: `distilbert-base-multilingual-cased`
- **Parameters**: ~134 Million
- **Vocabulary**: 119,547 tokens covering Latin, Devanagari, and 100+ languages
- **Labels**: BIO tagging schema (`O`, `B-PER`, `I-PER`, `B-ADDR`, `I-ADDR`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`)

## 2. Checkpoints & Formats
| Format | File Path | File Size | Memory Footprint |
|---|---|---|---|
| PyTorch SafeTensors | `models/localshield_multilingual/model.safetensors` | 514.01 MB | ~550 MB |
| ONNX FP32 | `models/localshield_multilingual/model.onnx` | 514.13 MB | ~520 MB |
| ONNX INT8 Quantized | `models/localshield_multilingual/model_quantized.onnx` | 128.91 MB | ~140 MB |

## 3. Training & Validation Performance
- **Validation Macro F1**: 0.9931
- **Per-Category Validation**:
  - `PERSON`: Precision 0.9738 | Recall 1.0000 | F1 0.9867
  - `ADDRESS`: Precision 0.9976 | Recall 1.0000 | F1 0.9988
  - `ORGANIZATION`: Precision 1.0000 | Recall 0.9936 | F1 0.9968
  - `LOCATION`: Precision 0.9763 | Recall 1.0000 | F1 0.9880

## 4. ONNX Parity & Quantization
- **Maximum Logits Absolute Difference (FP32 vs PyTorch)**: `0.000007` (7e-6)
- **Token Prediction Disagreement Rate (FP32)**: `0.00%`
- **Token Prediction Disagreement Rate (INT8)**: `1.12%`
- **Compression**: 74.9% size reduction with INT8 dynamic quantization
