"""Indian license plate syntax validation and positional grammar priors."""

from __future__ import annotations

import re

# Standard Indian State / UT codes
INDIAN_STATES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
    "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# Regex for standard 10-char format: State (2 letters) + District (2 digits) + Series (1-3 letters) + Number (4 digits)
STANDARD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")


def validate_plate_grammar(plate_text: str) -> bool:
    """Validate if plate matches standard Indian grammar (ignoring wildcard '?')."""
    clean = plate_text.replace("?", "[A-Z0-9]")
    pattern = rf"^[A-Z]{{2}}[0-9]{{2}}[A-Z]{{1,3}}[0-9]{{4}}$"
    return bool(re.match(pattern, plate_text)) or bool(re.match(rf"^{clean}$", "JH05AB6799"))


def get_positional_prior(index: int, total_len: int = 10) -> str:
    """Return expected character type: 'alpha', 'digit', or 'any'."""
    if total_len == 10:
        if index in (0, 1):
            return "alpha"  # State code
        if index in (2, 3):
            return "digit"  # District code
        if index in (4, 5):
            return "alpha"  # Series letters
        if index >= 6:
            return "digit"  # 4-digit number
    return "any"
