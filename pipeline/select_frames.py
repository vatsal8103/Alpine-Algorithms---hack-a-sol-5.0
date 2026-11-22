"""Frame quality ranking, best-N frame selection, and near-duplicate suppression."""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter / float(area1 + area2 - inter)


def select_best_frames(
    observations: list[dict[str, Any]],
    best_n: int = 12,
    dedup_iou: float = 0.85
) -> list[dict[str, Any]]:
    """Select top-N frames based on composite quality (sharpness * plate area * conf).
    
    Suppresses near-identical consecutive frames to maximize visual diversity.
    """
    if not observations:
        return []

    def quality_score(item: dict[str, Any]) -> float:
        sharpness = max(1.0, float(item.get("sharpness", 1.0)))
        bbox = item.get("bbox", [0, 0, 100, 100])
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        conf = float(item.get("conf", 0.8))
        return float(np.sqrt(sharpness) * np.sqrt(area) * conf)

    # Sort all candidate observations by quality descending
    sorted_obs = sorted(observations, key=quality_score, reverse=True)

    selected: list[dict[str, Any]] = []
    for obs in sorted_obs:
        if len(selected) >= best_n:
            break
        # Check IoU and frame distance against already selected frames
        is_duplicate = False
        for s in selected:
            # If same frame number, skip
            if obs["frame_no"] == s["frame_no"]:
                is_duplicate = True
                break
            # If consecutive frame and high bbox overlap, suppress
            if abs(obs["frame_no"] - s["frame_no"]) <= 1 and compute_iou(obs["bbox"], s["bbox"]) > dedup_iou:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(obs)

    # Fallback if filtering was too aggressive
    if len(selected) < min(3, len(observations)):
        selected = sorted_obs[:best_n]

    # Re-sort chronologically by frame_no
    selected.sort(key=lambda x: x["frame_no"])
    return selected
