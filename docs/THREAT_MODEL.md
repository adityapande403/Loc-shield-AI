# Threat Model

## 1. System Objective & Assets
- **Asset**: Sensitive user data, Indian and international national identifiers (Aadhaar, PAN, Passport, DL), payment cards, credentials, API secrets, and personal contact information.
- **Goal**: Prevent sensitive data leakage to third-party LLMs, logs, and external analytical services.

## 2. Threat Actors & Attack Vectors

### Vector A: LLM Data Exfiltration
- **Threat**: Application sends prompt containing user PII to cloud LLM provider, leading to retention, training set ingestion, or compliance violations (DPDP Act, GDPR, HIPAA).
- **Mitigation**: LocalShield acts as a local interceptor, replacing PII with opaque tokens before prompt dispatch.

### Vector B: Adversarial Text Evasion
- **Threat**: Adversaries insert zero-width non-joiners (`\u200c`), zero-width spaces (`\u200b`), or homoglyphs to evade standard regex detectors.
- **Mitigation**: `normalize_with_offsets()` strips invisible control characters and maps spans back to the raw string, ensuring both the identifier and the evasion characters are redacted.

### Vector C: Untrusted Re-identification
- **Threat**: An unauthorized user attempts to reverse pseudonymized tokens.
- **Mitigation**: Encrypted mapping storage requires encryption keys, session valid tokens, and explicit authorization parameters.
