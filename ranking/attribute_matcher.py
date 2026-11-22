"""Colour and body-type attribute scoring using similarity matrices."""

from __future__ import annotations

import json
from pathlib import Path

# Embedded default similarities if external files are missing
DEFAULT_COLOUR_SIM = {
    "white": {"white": 1.0, "silver": 0.75, "grey": 0.40},
    "silver": {"silver": 1.0, "white": 0.75, "grey": 0.70},
    "black": {"black": 1.0, "grey": 0.65},
    "red": {"red": 1.0, "maroon": 0.85, "orange": 0.65},
    "blue": {"blue": 1.0, "black": 0.30},
}

DEFAULT_BODY_SIM = {
    "hatchback": {"hatchback": 1.0, "sedan": 0.70, "suv": 0.40},
    "sedan": {"sedan": 1.0, "hatchback": 0.70, "suv": 0.45},
    "suv": {"suv": 1.0, "sedan": 0.45, "hatchback": 0.40},
    "two-wheeler": {"two-wheeler": 1.0, "auto-rickshaw": 0.25},
    "auto-rickshaw": {"auto-rickshaw": 1.0, "two-wheeler": 0.25},
    "truck": {"truck": 1.0, "bus": 0.45},
    "bus": {"bus": 1.0, "truck": 0.45},
}


def score_colour(observed: str, registered: str, table_path: str | None = None) -> float:
    """Compute colour compatibility score."""
    obs = observed.lower()
    reg = registered.lower()
    if obs == reg:
        return 1.0

    table = DEFAULT_COLOUR_SIM
    if table_path and Path(table_path).exists():
        with open(table_path, "r", encoding="utf-8") as f:
            table = json.load(f)

    return float(table.get(obs, {}).get(reg, 0.05))


def score_body_type(observed: str, registered: str, table_path: str | None = None) -> float:
    """Compute body-type compatibility score."""
    obs = observed.lower()
    reg = registered.lower()
    if obs == reg:
        return 1.0

    table = DEFAULT_BODY_SIM
    if table_path and Path(table_path).exists():
        with open(table_path, "r", encoding="utf-8") as f:
            table = json.load(f)

    return float(table.get(obs, {}).get(reg, 0.0))
