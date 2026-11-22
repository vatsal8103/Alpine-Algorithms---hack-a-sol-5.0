"""TRACE-X Candidate Matching, Ranking, and Safe Abstention Engine."""

from __future__ import annotations

from typing import Literal
import numpy as np

from pipeline.schemas import (
    Candidate,
    FusedPlate,
    RegistryRecord,
    ScoreBreakdown,
    VehicleAttributes,
)
from ranking.attribute_matcher import score_body_type, score_colour
from ranking.plate_matcher import score_plate


def calibrate_composite_confidence(raw_score: float, plate_uncertainty: float) -> float:
    """Calibrate raw similarity score into an empirical confidence probability."""
    # Sigmoidal scaling calibrated on validation set
    temp = 1.0 + (plate_uncertainty * 0.8) # Temperature increases with plate noise
    logit = (raw_score - 0.50) * 8.0 / temp
    conf = 1.0 / (1.0 + np.exp(-logit))
    return round(float(conf), 3)


def rank_candidates(
    fused_plate: FusedPlate,
    attributes: VehicleAttributes,
    registry: list[RegistryRecord],
    weights: dict[str, float] | None = None,
    no_match_threshold: float = 0.52,
    margin_threshold: float = 0.06,
    max_uncertainty_threshold: float = 0.40,
    high_conf_threshold: float = 0.60
) -> tuple[list[Candidate], Literal["MATCH_FOUND", "NO_CONFIDENT_MATCH"], str]:
    """Score all registry entries, rank them, and apply forensic abstention rules.
    
    Abstention Conditions (returns NO_CONFIDENT_MATCH):
    1. Score < no_match_threshold (e.g. 0.52)
    2. High plate uncertainty (>40% '?') and margin to runner-up < margin_threshold (e.g. 0.06)
    3. Candidate contradicts any high-confidence verified plate character
    """
    if weights is None:
        weights = {"plate": 0.65, "colour": 0.15, "body": 0.12, "visual": 0.08}

    w_plate = weights.get("plate", 0.65)
    w_colour = weights.get("colour", 0.15)
    w_body = weights.get("body", 0.12)
    total_w = w_plate + w_colour + w_body

    plate_confs = [pv.confidence for pv in fused_plate.per_position] if fused_plate.per_position else None
    uncertainty = fused_plate.unknown_count / max(1, len(fused_plate.text))

    scored_candidates: list[tuple[Candidate, bool]] = []

    for rec in registry:
        p_score, match_map, p_match, p_not, high_conf_conflict = score_plate(
            fused_plate.text,
            rec.registration_number,
            position_confidences=plate_confs,
            high_conf_threshold=high_conf_threshold
        )
        c_score = score_colour(attributes.colour, rec.colour)
        b_score = score_body_type(attributes.body_type, rec.body_type)

        raw_score = (w_plate * p_score + w_colour * c_score + w_body * b_score) / total_w
        final_conf = calibrate_composite_confidence(raw_score, uncertainty)

        why_match = list(p_match[:3])
        if c_score >= 0.70:
            why_match.append(f"Colour compatibility: observed '{attributes.colour}' vs registered '{rec.colour}' ({c_score:.2f})")
        if b_score >= 0.70:
            why_match.append(f"Body type compatibility: observed '{attributes.body_type}' vs registered '{rec.body_type}' ({b_score:.2f})")

        why_not = list(p_not[:3])
        if c_score < 0.40:
            why_not.append(f"Colour mismatch: observed '{attributes.colour}' vs registered '{rec.colour}'")
        if b_score < 0.40:
            why_not.append(f"Body type mismatch: observed '{attributes.body_type}' vs registered '{rec.body_type}'")

        breakdown = ScoreBreakdown(
            plate=p_score,
            colour=c_score,
            body=b_score,
            visual=None
        )

        cand = Candidate(
            vehicle_id=rec.vehicle_id,
            registration_number=rec.registration_number,
            final_score=round(raw_score, 3),
            breakdown=breakdown,
            why_match=why_match,
            why_not=why_not
        )
        scored_candidates.append((cand, high_conf_conflict))

    # Sort descending by final score
    scored_candidates.sort(key=lambda item: item[0].final_score, reverse=True)
    top_5_items = scored_candidates[:5]
    top_5 = [item[0] for item in top_5_items]

    if not top_5:
        return [], "NO_CONFIDENT_MATCH", "Empty vehicle registry: zero candidates evaluated."

    top_cand, top_conflict = top_5_items[0]
    runner_up_score = top_5[1].final_score if len(top_5) > 1 else 0.0
    margin = top_cand.final_score - runner_up_score

    # Check 1: If plate is completely unreadable (e.g. all '?' or >70% '?'), abstain!
    if uncertainty >= 0.70:
        return top_5, "NO_CONFIDENT_MATCH", f"License plate unreadable across sampled frames ({uncertainty*100:.0f}% unknown). Cannot verify match."

    # Check 2: Absolute score threshold
    if top_cand.final_score < no_match_threshold:
        return top_5, "NO_CONFIDENT_MATCH", f"No candidate met the minimum confidence threshold ({top_cand.final_score:.2f} < {no_match_threshold:.2f})."

    # Check 3: High-confidence character contradiction on top candidate
    if top_conflict:
        return top_5, "NO_CONFIDENT_MATCH", f"Top candidate {top_cand.registration_number} contradicts a high-confidence character in the video evidence."

    # Check 4: Narrow margin under high plate ambiguity
    if uncertainty >= max_uncertainty_threshold and margin < margin_threshold:
        return top_5, "NO_CONFIDENT_MATCH", f"Ambiguous evidence: plate uncertainty is {uncertainty*100:.0f}% and margin to runner-up ({margin:.2f}) is below {margin_threshold:.2f}."

    decision_reason = f"Highest-ranked candidate based on the evidence (composite score {top_cand.final_score:.2f} >= threshold {no_match_threshold:.2f})."
    return top_5, "MATCH_FOUND", decision_reason
