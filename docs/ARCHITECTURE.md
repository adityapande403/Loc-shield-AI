# LocalShield AI Architecture

LocalShield AI is a privacy-first, offline-capable PII detection and redaction system designed to prevent sensitive identifiers from leaving local perimeter boundaries.

```
                      +-----------------------------------+
                      |            Input Text             |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |         Input Validation          |
                      |  (Length Bounds, Unicode Guard)   |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Unicode & Script Normalization  |
                      |   (OffsetTracker Bi-directional)  |
                      +-----------------------------------+
                                        |
               +------------------------+------------------------+
               |                                                 |
               v                                                 v
+-------------------------------+             +----------------------------------+
|    Deterministic Detectors    |             |       Contextual ML Model        |
| - Verhoeff Aadhaar Checksum   |             | - DistilBERT Multilingual        |
| - Statutory PAN Formats       |             | - INT8 Quantized ONNX Runtime    |
| - Luhn Credit/Debit Cards     |             | - BIO Span Offset Decoder        |
| - IFSC & Bank Accounts        |             | - PERSON, ADDRESS, ORG, LOC      |
| - Phone, Email, IP, Vehicles  |             +----------------------------------+
| - High-Entropy API Secrets    |                                |
+-------------------------------+                                |
               |                                                 |
               +------------------------+------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |       Span Resolution Engine      |
                      |  (Deduplication, Conflict Rules,  |
                      |   Priority & Nesting Resolution)  |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |      Policy & Validation Gate     |
                      |  (Category Filtering, Minimum     |
                      |   Confidence Thresholds)          |
                      +-----------------------------------+
                                        |
                                        v
                      +-----------------------------------+
                      |          Redaction Engine         |
                      | - Mode A: Masking                 |
                      | - Mode B: Consistent Pseudonyms   |
                      | - Mode C: Type-Preserving         |
                      | - Mode D: Selective & Partial     |
                      +-----------------------------------+
                                        |
               +------------------------+------------------------+
               |                                                 |
               v                                                 v
+-------------------------------+             +----------------------------------+
|        Redacted Text          |             |   Optional Encrypted Storage     |
|   (Safe for External LLMs)    |             | - Authenticated Fernet (AES-128) |
+-------------------------------+             | - TTL Expiration & Authorized    |
                                              |   Re-identification Access       |
                                              +----------------------------------+
```

---

## 1. Normalization & Offset Tracking
To prevent evasion through zero-width spaces (`\u200b`, `\u200c`), fullwidth characters, or Devanagari numerals (`०-९`), LocalShield normalizes text while maintaining an exact character coordinate index map (`OffsetTracker`). When a detector finds an entity in normalized text, its coordinates map back to the original text.

## 2. Dual Detection Engines
- **Deterministic Engine**: Implements mathematical validation including the Verhoeff algorithm for 12-digit Aadhaar numbers, the Luhn mod-10 algorithm for credit card numbers, statutory 4th-character entity rules for PAN, IFSC bank prefix checks, and Shannon entropy analysis for cryptographic secrets.
- **Contextual Token Classifier**: A fine-tuned multilingual DistilBERT model covering English, Hindi (Devanagari), and Romanized Hindi (Hinglish). It identifies names, addresses, organizations, and locations that lack rigid regular expression structures.

## 3. Span Resolution Policy
Conflicts between overlapping spans are resolved deterministically:
1. Exact duplicates are deduplicated favoring the highest confidence and deterministic source.
2. Structured identifiers (Aadhaar, PAN, Cards, API Keys) take precedence over broad contextual spans.
3. Partial overlaps are decided by priority rank in the PII Taxonomy.
4. Non-overlapping intervals are selected greedily and sorted by start offset.

## 4. Redaction Modes
- **Mode A (Masking)**: Replaces entities with canonical tags (e.g. `[PERSON] called [PHONE_NUMBER]`).
- **Mode B (Consistent Pseudonymization)**: Assigns stable scoped identifiers (e.g. `[PERSON_1] called [PERSON_1]`).
- **Mode C (Type-Preserving)**: Adds entity counters (e.g. `[PHONE_NUMBER_1]`).
- **Mode D (Selective & Partial)**: Configurable category selection and optional partial masking (e.g. `XXXXXXXXXXXX5678`).
