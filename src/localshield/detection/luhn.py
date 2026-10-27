"""Luhn algorithm (mod 10) implementation for credit and debit card validation."""

from __future__ import annotations


def validate_luhn(card_number_str: str) -> bool:
    """Validate a card number string using the standard Luhn (mod 10) algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 12 or len(digits) > 19:
        return False

    checksum = 0
    # Process from right to left; double every second digit starting from the second-to-last
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d

    return checksum % 10 == 0


def generate_luhn_check_digit(number_str: str) -> int:
    """Calculate the single Luhn check digit to append to number_str."""
    digits = [int(c) for c in number_str if c.isdigit()]
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 0:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d

    remainder = checksum % 10
    return 0 if remainder == 0 else 10 - remainder
