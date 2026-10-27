"""Example: Reversible encrypted pseudonymization with LocalShield."""

from localshield import LocalShield, RedactionMode


def main():
    shield = LocalShield.from_pretrained("./models/localshield_multilingual")

    original_prompt = (
        "Customer Priya Patel called from Flat 402, Shanti Heights, Sector 62, Noida. "
        "Her account number is 9876543210123."
    )

    print("--- Original Sensitive Prompt ---")
    print(original_prompt)

    # 1. Redact with reversible pseudonymization
    result = shield.redact(
        original_prompt,
        mode=RedactionMode.PSEUDONYMIZE,
        enable_reversible=True,
    )

    print("\n--- Sanitized Prompt for External LLM ---")
    print(result.redacted_text)
    print(f"Session ID: {result.session_id}")

    # 2. Simulate LLM response containing the placeholders
    simulated_llm_response = (
        f"Summary: We have logged an inquiry for [PERSON_1] regarding address [ADDRESS_1] "
        f"and account [BANK_ACCOUNT_1]. A verification email has been sent."
    )
    print("\n--- External LLM Response Received ---")
    print(simulated_llm_response)

    # 3. Authorized re-identification
    restored = shield.redaction_engine.mapping_store.reidentify(
        simulated_llm_response,
        session_id=result.session_id,
        authorized=True,
    )

    print("\n--- Re-identified Response (Authorized Internal View) ---")
    print(restored)


if __name__ == "__main__":
    main()
