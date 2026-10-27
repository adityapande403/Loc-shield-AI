"""Upload trained model weights, ONNX graphs, tokenizer, and model card to Hugging Face Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Adityapande403/LocShield-AI"
MODEL_DIR = Path("models/localshield_multilingual")

MODEL_CARD_CONTENT = """---
language:
- en
- hi
license: apache-2.0
tags:
- token-classification
- pii
- privacy
- redaction
- onnx
- quantized
- multilingual
- distilbert
datasets:
- synthetic
metrics:
- f1
- precision
- recall
pipeline_tag: token-classification
---

# LocShield-AI: Multilingual PII Detection & Token Classification Model

LocShield-AI is a privacy-first, offline-capable token classification model based on `distilbert-base-multilingual-cased`, optimized for contextual PII detection across English, Hindi (Devanagari script), and Romanized Hindi (Hinglish).

Part of the **LocalShield AI** ecosystem: [https://github.com/adityapande403/Loc-shield-AI](https://github.com/adityapande403/Loc-shield-AI)

## Model Details
- **Architecture**: `DistilBertForTokenClassification`
- **Base Encoder**: `distilbert-base-multilingual-cased` (~134M parameters)
- **Entities Covered**:
  - `PERSON` (Individual names in English, Hindi, and Hinglish)
  - `ADDRESS` (Street and postal addresses, landmarks, PIN locations)
  - `ORGANIZATION` (Companies, banks, institutions)
  - `LOCATION` (Cities, states, countries)
- **Labels**: BIO tagging schema (`O`, `B-PER`, `I-PER`, `B-ADDR`, `I-ADDR`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`)
- **Formats Available**:
  - `model.safetensors` - PyTorch SafeTensors (514 MB)
  - `model.onnx` - ONNX FP32 Graph (514 MB)
  - `model_quantized.onnx` - Dynamic INT8 Quantized ONNX (128.9 MB, 74.9% compression)

## Performance & Quality
- **Validation Macro F1**: **0.9931**
  - `PERSON`: F1 0.9867 (Precision: 0.9738, Recall: 1.0000)
  - `ADDRESS`: F1 0.9988 (Precision: 0.9976, Recall: 1.0000)
  - `ORGANIZATION`: F1 0.9968 (Precision: 1.0000, Recall: 0.9936)
  - `LOCATION`: F1 0.9880 (Precision: 0.9763, Recall: 1.0000)
- **CPU Inference Latency (ONNX INT8)**:
  - **P50 Latency**: **2.27 ms**
  - **Throughput**: **428.3 requests/second** on CPU
  - **Memory Footprint**: ~140 MB RSS

## Quickstart: Python (Transformers)

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("Adityapande403/LocShield-AI")
model = AutoModelForTokenClassification.from_pretrained("Adityapande403/LocShield-AI")

classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = classifier("Customer Rahul Sharma reported from Bengaluru regarding account at HDFC Bank.")
print(results)
```

## Quickstart: LocalShield Engine (Recommended)

To run with deterministic validators (Aadhaar Verhoeff, Luhn cards, PAN, IFSC, etc.) and complete redaction:

```bash
git clone https://github.com/adityapande403/Loc-shield-AI.git
cd Loc-shield-AI
pip install -e .
```

```python
from localshield import LocalShield

shield = LocalShield.from_pretrained("Adityapande403/LocShield-AI")
result = shield.redact("Call Rahul Sharma at 9876543210 regarding PAN ABCPE1234F.")
print(result.redacted_text)
# Output: Call [PERSON] at [PHONE_NUMBER] regarding PAN [PAN].
```

## License
Apache-2.0
"""


def main():
    api = HfApi()
    print(f"Uploading model artifacts to https://huggingface.co/{REPO_ID}...")

    # Write Hugging Face README.md temporarily in model directory
    readme_path = MODEL_DIR / "README.md"
    readme_path.write_text(MODEL_CARD_CONTENT, encoding="utf-8")

    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="feat: upload trained SafeTensors, ONNX FP32, INT8 quantized weights and model card",
    )

    print(f"Upload complete! Model is live at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
