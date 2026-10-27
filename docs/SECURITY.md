# LocalShield AI Security Policy & Safeguards

## 1. Strict Offline Guarantee
LocalShield is architected to operate in strictly air-gapped or offline environments:
- Zero runtime network calls during detection, analysis, or redaction.
- No model weights are downloaded at inference time.
- All telemetry and background network requests are disabled by default.
- Verified by automated unit test `test_strict_offline_mode_no_network_calls` which monkeypatches `socket.socket.connect` to assert zero socket connections occur during inference.

## 2. Sensitive Data Handling in Logs and Exceptions
- **Zero Raw PII Logging**: The engine does not log raw text or extracted entities in log statements.
- **Safe Candidate Representation**: `PIICandidate.to_safe_dict()` strips raw text and leaves only category, offset, and confidence for diagnostic auditing.
- **Safe Error Responses**: The FastAPI service captures exceptions and returns sanitized errors without stack traces or request fragments.

## 3. Cryptographic Storage for Reversible Pseudonymization
- **Standard Cryptography**: Uses authenticated symmetric encryption (`Fernet` / AES-128-CBC with HMAC-SHA256 from the Python `cryptography` library). No proprietary or untested encryption algorithms are used.
- **Key Separation**: Encryption keys are stored separately from encrypted payload files.
- **Access Authorization**: The `reidentify()` API explicitly checks authorization flags and raises `PermissionError` if unauthorized.
- **TTL Expiration**: Mappings expire automatically after the configured time-to-live (default: 86,400 seconds).

## 4. Resource & Input Bounds
- Configurable maximum input length (`max_input_chars=1,000,000`).
- Configurable batch size limit (`max_batch_size=500`).
- Regex patterns use bounded quantifiers and strict lookaheads/lookbehinds to prevent Regular Expression Denial of Service (ReDoS).
