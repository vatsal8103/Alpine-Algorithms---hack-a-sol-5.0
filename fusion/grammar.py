"""Indian license plate syntax validation and positional grammar priors."""

from __future__ import annotations

import re

# Standard Indian State / UT codes
INDIAN_STATES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# Confusion mapping between visually confusable letters and digits
LETTER_TO_DIGIT_CONFUSIONS = {
    "O": "0",
    "I": "1",
    "B": "8",
    "S": "5",
    "Z": "2",
    "D": "0",
    "G": "6",
    "L": "4",
    "T": "7",
    "A": "4",
}

DIGIT_TO_LETTER_CONFUSIONS = {
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
    "6": "G",
    "7": "T",
    "4": "A",
}

# Regex for standard format: State (2 letters) + District (2 digits) + Series (1-3 letters) + Number (4 digits)
STANDARD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")


def validate_plate_grammar(plate_text: str) -> bool:
    """Validate if plate matches standard Indian grammar (ignoring wildcard '?')."""
    if len(plate_text) not in (9, 10, 11):
        return False
    clean = re.sub(r"[^A-Z0-9?]", "", plate_text.upper())
    if len(clean) != len(plate_text):
        return False
    for i, ch in enumerate(clean):
        if ch == "?":
            continue
        expected = get_positional_prior(i, total_len=len(clean))
        if expected == "alpha" and not ch.isalpha():
            return False
        if expected == "digit" and not ch.isdigit():
            return False
    return True


def get_positional_prior(index: int, total_len: int = 10) -> str:
    """Return expected character type: 'alpha', 'digit', or 'any' based on Indian plate grammar."""
    if total_len < 7:
        return "any"
    if index in (0, 1):
        return "alpha"  # State code: 2 letters
    if index in (2, 3):
        return "digit"  # District code: 2 digits
    if index >= total_len - 4:
        return "digit"  # Last 4 characters: 4-digit number
    if 4 <= index < total_len - 4:
        return "alpha"  # Series code: 1 to 3 letters
    return "any"
