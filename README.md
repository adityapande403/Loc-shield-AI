# LocalShield AI
> **Privacy-First Local PII Detection & Redaction Engine**

LocalShield AI is a local, offline-capable PII detection and redaction engine designed to sanitize sensitive data before it reaches external LLMs, analytics services, logging platforms, databases, or third-party APIs.

---

## Key Features

- **Strict Offline Execution**: Operates locally with zero external API calls or runtime downloads.
- **Hybrid Detection**:
  - **Deterministic Validators**: Algorithmic mathematical validation for Aadhaar (Verhoeff checksum), Credit Cards (Luhn algorithm), statutory PAN rules, IFSC codes, vehicle plates, emails, IPs, and high-entropy API keys.
  - **Contextual ML Model**: Fine-tuned multilingual DistilBERT model (~134M params) covering English, Hindi (Devanagari), and Romanized Hindi (Hinglish) for contextual entities (`PERSON`, `ADDRESS`, `ORGANIZATION`, `LOCATION`).
- **ONNX CPU Optimization**: Quantized to dynamic INT8 (128.9 MB, 74.9% compression) delivering **3.95 ms P50 latency** and **260+ requests/sec** on CPU.
- **Evasion-Resistant Normalization**: `OffsetTracker` identifies hidden zero-width spaces (`\u200b`, `\u200c`) and normalizes Devanagari numerals without modifying raw string offsets.
- **4 Redaction Modes**:
  1. **Mode A (Masking)**: `[PERSON] called [PHONE_NUMBER].`
  2. **Mode B (Consistent Pseudonymization)**: `[PERSON_1] called [PERSON_1].`
  3. **Mode C (Type-Preserving)**: `Contact: [PHONE_NUMBER_1]`
  4. **Mode D (Selective & Partial)**: Configurable category selection and partial masking (`XXXX-XXXX-1234`).
- **Optional Reversible Pseudonymization**: Authenticated symmetric AES/Fernet encrypted local mapping with TTL expiration and authorization controls.
- **Multi-Platform Interfaces**: Python SDK, RESTful FastAPI service, Node.js SDK, and CLI.

---

## Supported PII Categories

| Category | Description | Primary Engine |
|---|---|---|
| `AADHAAR` | Indian 12-digit UIDAI number (Verhoeff checksum) | Deterministic (Verhoeff) |
| `PAN` | Indian Permanent Account Number (10 alphanumeric statutory) | Deterministic (Format Rules) |
| `CREDIT_CARD` | Visa, Mastercard, Amex, RuPay, Discover (Luhn check) | Deterministic (Luhn) |
| `IFSC` | Indian Financial System Code (11 alphanumeric, bank prefix) | Deterministic |
| `BANK_ACCOUNT` | 9-18 digit account numbers with context keywords | Deterministic + Context |
| `PHONE_NUMBER` | Indian (+91, 10-digit) and E.164 international numbers | Deterministic |
| `EMAIL` | RFC 5322 compliant email addresses | Deterministic |
| `IP_ADDRESS` | IPv4 (0-255 octet bounds) and IPv6 addresses | Deterministic |
| `API_KEY` | AWS, GitHub, Slack, OpenAI, Google keys & high entropy | Deterministic + Entropy |
| `ACCESS_TOKEN` | OAuth bearer tokens and JWTs | Deterministic |
| `AUTH_TOKEN` | Embedded credentials and private keys | Deterministic |
| `VEHICLE_REGISTRATION` | Indian state RTO license plates (DL, MH, BH series) | Deterministic |
| `DATE_OF_BIRTH` | Birth dates with calendar and context validation | Deterministic + Context |
| `USERNAME` | User handles (@handle) and logins | Deterministic |
| `PERSON` | Individual names in English, Hindi, and Hinglish | ML Token Classifier |
| `ADDRESS` | Physical, postal street addresses and landmarks | ML Token Classifier |
| `ORGANIZATION` | Corporate, banking, educational, or government bodies | ML Token Classifier |
| `LOCATION` | Cities, states, countries, and geographical regions | ML Token Classifier |
| `PASSPORT` | Indian and international passport numbers | Deterministic + Context |
| `DRIVER_LICENSE` | State-code driving license formats | Deterministic + Context |

---

## Installation

### Prerequisites
- Python >= 3.10
- Node.js >= 18 (for Node.js SDK)

```bash
git clone https://github.com/LocalShield/localshield-ai.git
cd localshield-ai
pip install -e .
```

---

## Quickstart: Python SDK

```python
from localshield import LocalShield, RedactionMode

# Initialize engine (loads INT8 quantized ONNX model)
shield = LocalShield.from_pretrained("./models/localshield_multilingual")

# 1. Basic Redaction (Masking)
result = shield.redact("Contact Rahul Sharma at 9876543210 regarding PAN ABCPE1234F.")
print(result.redacted_text)
# Output: Contact [PERSON] at [PHONE_NUMBER] regarding PAN [PAN].

# 2. Consistent Pseudonymization
res_pseudo = shield.redact(
    "Rahul Sharma called Priya. Later Rahul Sharma emailed Priya.",
    mode=RedactionMode.PSEUDONYMIZE
)
print(res_pseudo.redacted_text)
# Output: [PERSON_1] called [PERSON_2]. Later [PERSON_1] emailed [PERSON_2].

# 3. Detection Candidates
candidates = shield.detect("Transfer to SBIN0001234")
for c in candidates:
    print(c.entity_type, c.text, c.confidence)
```

---

## CLI Usage

```bash
# Detect PII in text file
localshield detect --input sample.txt

# Redact PII to output file
localshield redact --input sample.txt --output redacted.txt --mode mask

# Benchmark CPU inference performance
localshield benchmark --backend onnx-int8 --iterations 50

# Validate local model artifacts
localshield validate-model --model ./models/localshield_multilingual

# Run local FastAPI service
localshield serve --host 127.0.0.1 --port 8000
```

---

## Node.js SDK

```javascript
const { LocalShieldClient } = require("./sdk_node");

const client = new LocalShieldClient({ baseUrl: "http://127.0.0.1:8000" });

async function sanitizePrompt(userPrompt) {
  const result = await client.redact(userPrompt, { mode: "mask" });
  return result.redacted_text;
}

sanitizePrompt("Customer Rahul Sharma called 9876543210").then(console.log);
// Output: Customer [PERSON] called [PHONE_NUMBER]
```

---

## Running the Automated Test Suite

```bash
python -m pytest -v tests/
```
All 38 test cases cover:
- Deterministic validators (Verhoeff Aadhaar, Luhn Credit Card, PAN, IFSC, Secrets)
- Zero-width evasion offset preservation
- Redaction modes & encrypted reversible storage
- End-to-end pipeline and FastAPI integration
- Strict offline network isolation
- Node.js live integration

---

## Documentation

- [Architecture & Pipeline](docs/ARCHITECTURE.md)
- [Security & Offline Policy](docs/SECURITY.md)
- [Threat Model](docs/THREAT_MODEL.md)
- [Dataset Card](docs/DATASET_CARD.md)
- [Model Card](docs/MODEL_CARD.md)
- [Evaluation Report](docs/EVALUATION.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

---

## License
Apache 2.0. See [LICENSE](LICENSE) for details.
