"""Colour and body-type attribute scoring using similarity matrices."""

from __future__ import annotations

import json
from pathlib import Path

# Embedded default similarities if external files are missing
DEFAULT_COLOUR_SIM = {
    "white": {"white": 1.0, "silver": 0.75, "grey": 0.40, "black": 0.0, "yellow": 0.20, "red": 0.05, "blue": 0.05, "green": 0.05, "brown": 0.05, "orange": 0.10},
    "silver": {"silver": 1.0, "white": 0.75, "grey": 0.70, "black": 0.10, "yellow": 0.15, "red": 0.05, "blue": 0.10, "green": 0.05, "brown": 0.10, "orange": 0.05},
    "grey": {"grey": 1.0, "silver": 0.70, "black": 0.65, "white": 0.40, "brown": 0.30, "blue": 0.25, "red": 0.10, "yellow": 0.05, "green": 0.10, "orange": 0.05},
    "black": {"black": 1.0, "grey": 0.65, "silver": 0.10, "brown": 0.35, "blue": 0.30, "white": 0.0, "red": 0.15, "green": 0.15, "yellow": 0.0, "orange": 0.0},
    "red": {"red": 1.0, "maroon": 0.85, "orange": 0.65, "brown": 0.40, "black": 0.15, "yellow": 0.20, "white": 0.05, "silver": 0.05, "grey": 0.10, "blue": 0.0},
    "blue": {"blue": 1.0, "black": 0.30, "grey": 0.25, "silver": 0.10, "green": 0.25, "white": 0.05, "red": 0.0, "yellow": 0.0, "brown": 0.10, "orange": 0.0},
    "yellow": {"yellow": 1.0, "orange": 0.70, "white": 0.20, "green": 0.30, "silver": 0.15, "red": 0.20, "grey": 0.05, "black": 0.0, "blue": 0.0, "brown": 0.10},
    "green": {"green": 1.0, "yellow": 0.30, "blue": 0.25, "black": 0.15, "grey": 0.10, "brown": 0.20, "white": 0.05, "silver": 0.05, "red": 0.0, "orange": 0.0},
    "brown": {"brown": 1.0, "black": 0.35, "grey": 0.30, "red": 0.40, "orange": 0.30, "maroon": 0.60, "silver": 0.10, "white": 0.05, "yellow": 0.10, "green": 0.20},
    "orange": {"orange": 1.0, "red": 0.65, "yellow": 0.70, "brown": 0.30, "white": 0.10, "silver": 0.05, "grey": 0.05, "black": 0.0, "blue": 0.0, "green": 0.0}
}

DEFAULT_BODY_SIM = {
    "hatchback": {"hatchback": 1.0, "sedan": 0.70, "suv": 0.40, "auto-rickshaw": 0.05, "two-wheeler": 0.0, "truck": 0.0, "bus": 0.0},
    "sedan": {"sedan": 1.0, "hatchback": 0.70, "suv": 0.45, "auto-rickshaw": 0.05, "two-wheeler": 0.0, "truck": 0.0, "bus": 0.0},
    "suv": {"suv": 1.0, "sedan": 0.45, "hatchback": 0.40, "truck": 0.25, "auto-rickshaw": 0.05, "two-wheeler": 0.0, "bus": 0.10},
    "two-wheeler": {"two-wheeler": 1.0, "auto-rickshaw": 0.25, "hatchback": 0.0, "sedan": 0.0, "suv": 0.0, "truck": 0.0, "bus": 0.0},
    "auto-rickshaw": {"auto-rickshaw": 1.0, "two-wheeler": 0.25, "hatchback": 0.10, "sedan": 0.05, "suv": 0.05, "truck": 0.05, "bus": 0.0},
    "truck": {"truck": 1.0, "bus": 0.45, "suv": 0.25, "auto-rickshaw": 0.05, "hatchback": 0.0, "sedan": 0.0, "two-wheeler": 0.0},
    "bus": {"bus": 1.0, "truck": 0.45, "suv": 0.15, "auto-rickshaw": 0.0, "hatchback": 0.0, "sedan": 0.0, "two-wheeler": 0.0}
}


def score_colour(observed: str, registered: str, table_path: str | None = None) -> float:
    """Compute colour compatibility score."""
    obs = observed.lower()
    reg = registered.lower()
    if obs == reg:
        return 1.0

    if table_path is None:
        table_path = "configs/colour_similarity.json"

    table = DEFAULT_COLOUR_SIM
    if table_path and Path(table_path).exists():
        try:
            with open(table_path, "r", encoding="utf-8") as f:
                table = json.load(f)
        except Exception:
            table = DEFAULT_COLOUR_SIM

    return float(table.get(obs, {}).get(reg, 0.05))


def score_body_type(observed: str, registered: str, table_path: str | None = None) -> float:
    """Compute body-type compatibility score."""
    obs = observed.lower()
    reg = registered.lower()
    if obs == reg:
        return 1.0

    if table_path is None:
        table_path = "configs/body_type_similarity.json"

    table = DEFAULT_BODY_SIM
    if table_path and Path(table_path).exists():
        try:
            with open(table_path, "r", encoding="utf-8") as f:
                table = json.load(f)
        except Exception:
            table = DEFAULT_BODY_SIM

    return float(table.get(obs, {}).get(reg, 0.0))
