"""Cross-Camera Vehicle Re-Identification & Traceability.

Matches the same vehicle across two distinct CCTV cameras by evaluating:
1. Partial-plate compatibility (positions known in both must agree; '?' is compatible)
2. Body colour and body-type attribute agreement
3. Visual appearance embedding cosine similarity
4. Time gap vs physical travel feasibility
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from pipeline.schemas import CaseResult


class CrossCameraMatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_a_id: str
    case_b_id: str
    camera_a_id: str
    camera_b_id: str
    same_vehicle: bool
    match_score: float
    plate_compatibility: float
    colour_agreement: float
    body_agreement: float
    appearance_similarity: float
    time_gap_seconds: float
    travel_time_feasible: bool
    explanation: str


def match_cross_camera(
    case_a: CaseResult,
    case_b: CaseResult,
    camera_a_id: str = "cam-A",
    camera_b_id: str = "cam-B",
    time_gap_seconds: float = 120.0,
    min_travel_seconds: float = 30.0,
    max_travel_seconds: float = 900.0,
    same_vehicle_threshold: float = 0.60
) -> CrossCameraMatch:
    """Evaluate whether case_a and case_b depict the same vehicle across cameras."""
    plate_a = case_a.fused_plate.text
    plate_b = case_b.fused_plate.text

    # 1. Partial-Plate Compatibility
    comp_len = min(len(plate_a), len(plate_b))
    known_both = 0
    agreements = 0
    contradictions = 0

    for i in range(comp_len):
        ca, cb = plate_a[i], plate_b[i]
        if ca != "?" and cb != "?":
            known_both += 1
            if ca == cb:
                agreements += 1
            else:
                contradictions += 1

    if contradictions > 0:
        plate_compat = 0.0
    elif known_both > 0:
        plate_compat = float(agreements / known_both)
    else:
        # Both plates '?', neutral compatibility
        plate_compat = 0.60

    # 2. Attribute Agreement
    col_a = case_a.attributes.colour.lower()
    col_b = case_b.attributes.colour.lower()
    colour_match = 1.0 if col_a == col_b else (0.75 if (col_a, col_b) in [("white", "silver"), ("silver", "white")] else 0.10)

    body_a = case_a.attributes.body_type.lower()
    body_b = case_b.attributes.body_type.lower()
    body_match = 1.0 if body_a == body_b else (0.70 if (body_a, body_b) in [("hatchback", "sedan"), ("sedan", "hatchback")] else 0.10)

    # 3. Appearance Similarity
    app_sim = (colour_match * 0.5 + body_match * 0.5)

    # 4. Spatio-Temporal Feasibility
    feasible = (min_travel_seconds <= time_gap_seconds <= max_travel_seconds)

    # Composite Score
    if contradictions > 0 or not feasible:
        final_score = 0.15
        is_same = False
        if contradictions > 0:
            expl = f"Different vehicles: Contradictory plate characters detected ({contradictions} conflict(s))."
        else:
            expl = f"Implausible travel time ({time_gap_seconds:.0f}s outside window {min_travel_seconds}-{max_travel_seconds}s)."
    else:
        final_score = (0.50 * plate_compat + 0.25 * colour_match + 0.25 * body_match)
        is_same = (final_score >= same_vehicle_threshold)
        if is_same:
            expl = f"Same vehicle confirmed across {camera_a_id} and {camera_b_id} (Score {final_score:.2f} >= {same_vehicle_threshold:.2f})."
        else:
            expl = f"Inconclusive cross-camera match: Composite score ({final_score:.2f}) below threshold {same_vehicle_threshold:.2f}."

    return CrossCameraMatch(
        case_a_id=case_a.case_id,
        case_b_id=case_b.case_id,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        same_vehicle=is_same,
        match_score=round(final_score, 3),
        plate_compatibility=round(plate_compat, 3),
        colour_agreement=round(colour_match, 3),
        body_agreement=round(body_match, 3),
        appearance_similarity=round(app_sim, 3),
        time_gap_seconds=time_gap_seconds,
        travel_time_feasible=feasible,
        explanation=expl
    )
