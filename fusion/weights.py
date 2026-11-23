"""Frame weighting calculation based on sharpness, plate area, and mean OCR confidence."""

from __future__ import annotations

import numpy as np


def compute_frame_weights(
    sharpness_scores: list[float],
    plate_areas: list[float],
    mean_confs: list[float],
    w_sharp: float = 0.30,
    w_area: float = 0.20,
    w_conf: float = 1.20
) -> list[float]:
    """Compute normalized frame weights using power law formula."""
    if not sharpness_scores:
        return []

    raw_weights = []
    for s, a, c in zip(sharpness_scores, plate_areas, mean_confs):
        s_norm = max(1e-4, s)
        a_norm = max(1.0, a)
        c_norm = max(1e-2, c)
        raw = (s_norm ** w_sharp) * (a_norm ** w_area) * (c_norm ** w_conf)
        raw_weights.append(raw)

    total = sum(raw_weights)
    if total <= 0:
        n = len(raw_weights)
        return [1.0 / n] * n

    return [w / total for w in raw_weights]
