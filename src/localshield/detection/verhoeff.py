"""Verhoeff algorithm implementation for validating Indian Aadhaar numbers and generating test check-digits.

The Verhoeff algorithm is a checksum formula for error detection developed by Dutch
mathematician Jacobus Verhoeff. UIDAI uses this algorithm for the 12th check digit of Aadhaar.
"""

from __future__ import annotations

# The multiplication table (d)
VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# The permutation table (p)
VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# The inverse table (inv)
VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def validate_verhoeff(number_str: str) -> bool:
    """Validate a numeric string using the Verhoeff algorithm.

    Returns True if the check digit is mathematically correct.
    """
    clean_digits = [int(c) for c in number_str if c.isdigit()]
    if not clean_digits:
        return False

    c = 0
    for i, item in enumerate(reversed(clean_digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][item]]

    return c == 0


def generate_verhoeff_check_digit(number_str: str) -> int:
    """Generate the single-digit Verhoeff check digit for a string of digits."""
    clean_digits = [int(c) for c in number_str if c.isdigit()]
    c = 0
    for i, item in enumerate(reversed(clean_digits)):
        c = VERHOEFF_D[c][VERHOEFF_P[(i + 1) % 8][item]]

    return VERHOEFF_INV[c]
