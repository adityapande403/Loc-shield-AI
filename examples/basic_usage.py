"""Example: Basic PII detection and redaction using LocalShield."""

from localshield import LocalShield, RedactionMode


def main():
    shield = LocalShield.from_pretrained("./models/localshield_multilingual")

    sample_text = (
        "Rahul Sharma (Phone: +91 9876543210, PAN: ABCPE1234F, Aadhaar: 2345 6789 0124) "
        "works at HDFC Bank in Mumbai. His server token is AKIAIOSFODNN7EXAMPLE."
    )

    print("--- 1. Original Text ---")
    print(sample_text)

    # 1. Detect
    print("\n--- 2. Detection Results ---")
    candidates = shield.detect(sample_text)
    for c in candidates:
        print(f"[{c.start}:{c.end}] {c.entity_type:<20} '{c.text}' (Source: {c.source}, Conf: {c.confidence})")

    # 2. Redact - Mode A (Mask)
    print("\n--- 3. Redacted (Mask Mode) ---")
    res_mask = shield.redact(sample_text, mode=RedactionMode.MASK)
    print(res_mask.redacted_text)

    # 3. Redact - Mode B (Pseudonymize)
    print("\n--- 4. Consistent Pseudonymization ---")
    repeated_text = "Rahul Sharma talked to Amit Kumar. Then Rahul Sharma emailed Priya."
    res_pseudo = shield.redact(repeated_text, mode=RedactionMode.PSEUDONYMIZE)
    print(res_pseudo.redacted_text)

    # 4. Risk Analysis
    print("\n--- 5. Security & Risk Analysis ---")
    analysis = shield.analyze(sample_text)
    print(f"Risk Score: {analysis.risk_score} / 1.0")
    for cat, stat in analysis.category_breakdown.items():
        print(f" - {cat}: {stat.count} detected (avg conf: {stat.confidence_avg})")


if __name__ == "__main__":
    main()
