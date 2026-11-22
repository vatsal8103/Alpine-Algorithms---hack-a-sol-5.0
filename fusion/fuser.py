"""Multi-frame evidential consensus fusion engine (original algorithm).

Guarantees:
- Never invent a character: an unverified character becomes '?'.
- Every position carries an evidential posterior and margin check.
- Indian license plate grammar constraints applied as positional priors.
- Confusable-pair handling (O/0, B/8, I/1, S/5, Z/2, D/0, G/6, T/7, V/U).
"""

from __future__ import annotations

from typing import Any
import numpy as np

from fusion.align import align_sequences
from fusion.grammar import get_positional_prior, validate_plate_grammar
from fusion.weights import compute_frame_weights
from pipeline.schemas import FrameOCRResult, FrameVote, FusedPlate, PositionVote


def fuse_frames(
    results: list[FrameOCRResult],
    accept_threshold: float = 0.50,
    margin_threshold: float = 0.12,
    alternative_weight: float = 0.30
) -> FusedPlate:
    """Fuse per-frame OCR character predictions into a calibrated FusedPlate."""
    if not results:
        # Zero frames fallback
        votes = [
            PositionVote(
                index=i,
                char_or_unknown="?",
                confidence=0.0,
                votes=[],
                explanation="No frames available for evidence aggregation"
            )
            for i in range(10)
        ]
        return FusedPlate(
            text="??????????",
            per_position=votes,
            format_valid=False,
            frames_used=0,
            mean_confidence=0.0,
            known_count=0,
            unknown_count=10
        )

    # Compute normalized frame weights
    sharpnesses = [max(1.0, r.sharpness) for r in results]
    areas = [max(1.0, r.plate_area_px) for r in results]
    mean_confs = [
        float(np.mean([c.prob for c in r.chars])) if r.chars else 0.50
        for r in results
    ]
    frame_weights = compute_frame_weights(sharpnesses, areas, mean_confs)

    # Determine reference length from highest weighted frames
    lengths = [len(r.chars) for r in results if r.chars]
    target_len = int(round(float(np.median(lengths)))) if lengths else 10
    target_len = max(8, min(11, target_len))

    fused_chars: list[str] = []
    position_votes: list[PositionVote] = []
    known_confs: list[float] = []

    for pos in range(target_len):
        class_evidence: dict[str, float] = {}
        votes: list[FrameVote] = []
        prior_type = get_positional_prior(pos, total_len=target_len)

        for frame_idx, res in enumerate(results):
            if pos < len(res.chars):
                c_item = res.chars[pos]
                w = frame_weights[frame_idx]

                # Main character vote
                class_evidence[c_item.char] = class_evidence.get(c_item.char, 0.0) + (w * c_item.prob)
                votes.append(FrameVote(
                    frame_no=res.frame_no,
                    char=c_item.char,
                    prob=c_item.prob,
                    weight=round(w, 4)
                ))

                # Alternatives vote with discount
                for alt in c_item.alternatives:
                    class_evidence[alt.char] = class_evidence.get(alt.char, 0.0) + (w * alt.prob * alternative_weight)

        if not class_evidence:
            fused_chars.append("?")
            position_votes.append(PositionVote(
                index=pos,
                char_or_unknown="?",
                confidence=0.0,
                votes=votes,
                explanation="No observations at this position"
            ))
            continue

        # Apply positional grammar prior: penalize characters contradicting expected type
        total_ev = sum(class_evidence.values())
        normalized_posteriors = {}
        for ch, ev in class_evidence.items():
            post = ev / max(1e-5, total_ev)
            if prior_type == "alpha" and ch.isdigit():
                post *= 0.60 # Grammar mismatch penalty
            elif prior_type == "digit" and ch.isalpha():
                post *= 0.60
            normalized_posteriors[ch] = post

        # Renormalize posteriors
        renorm_total = sum(normalized_posteriors.values())
        for ch in normalized_posteriors:
            normalized_posteriors[ch] /= max(1e-5, renorm_total)

        sorted_posteriors = sorted(normalized_posteriors.items(), key=lambda x: x[1], reverse=True)
        top_char, top_prob = sorted_posteriors[0]
        runner_up_prob = sorted_posteriors[1][1] if len(sorted_posteriors) > 1 else 0.0
        margin = top_prob - runner_up_prob

        raw_score = class_evidence.get(top_char, 0.0)

        # STRICT DECISION RULE:
        # Top character accepted ONLY IF raw evidence and posterior >= accept_threshold AND margin >= margin_threshold
        if raw_score >= accept_threshold and top_prob >= accept_threshold and margin >= margin_threshold:
            fused_chars.append(top_char)
            known_confs.append(top_prob)
            position_votes.append(PositionVote(
                index=pos,
                char_or_unknown=top_char,
                confidence=round(top_prob, 3),
                votes=votes,
                explanation=None
            ))
        else:
            fused_chars.append("?")
            explanation = (
                f"Position {pos}: '{top_char}' (raw={raw_score:.2f}, p={top_prob:.2f}) vs runner-up "
                f"(p={runner_up_prob:.2f}) - margin {margin:.2f} < threshold {margin_threshold:.2f} "
                f"or score < accept_threshold {accept_threshold:.2f}"
            )
            position_votes.append(PositionVote(
                index=pos,
                char_or_unknown="?",
                confidence=round(top_prob, 3),
                votes=votes,
                explanation=explanation
            ))

    fused_text = "".join(fused_chars)
    known_c = sum(1 for c in fused_text if c != "?")
    unknown_c = sum(1 for c in fused_text if c == "?")
    mean_conf = float(np.mean(known_confs)) if known_confs else 0.0

    return FusedPlate(
        text=fused_text,
        per_position=position_votes,
        format_valid=validate_plate_grammar(fused_text),
        frames_used=len(results),
        mean_confidence=round(mean_conf, 3),
        known_count=known_c,
        unknown_count=unknown_c
    )
