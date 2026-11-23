"""Fuse plate OCR from distinct video frames, retaining unresolved positions as question marks."""

from __future__ import annotations

from typing import Any
import numpy as np

from fusion.align import align_chars_to_reference, align_sequences
from fusion.grammar import (
    DIGIT_TO_LETTER_CONFUSIONS,
    INDIAN_STATES,
    LETTER_TO_DIGIT_CONFUSIONS,
    get_positional_prior,
    parse_plate,
    validate_plate_grammar,
)
from fusion.weights import compute_frame_weights
from pipeline.schemas import FrameOCRResult, FrameVote, FusedPlate, PositionVote


def fuse_frames(
    results: list[FrameOCRResult],
    accept_threshold: float = 0.40,
    margin_threshold: float = 0.06,
    alternative_weight: float = 0.35
) -> FusedPlate:
    """Fuse per-frame OCR character predictions into a calibrated FusedPlate."""
    if not results:
        votes = [
            PositionVote(
                index=i,
                char_or_unknown="?",
                confidence=0.0,
                votes=[],
                candidates=[],
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

    # Filter out empty results and long text without any digits (e.g. dealer ads / badges)
    valid_results = [
        r for r in results 
        if r.chars and any(c.prob >= 0.15 for c in r.chars)
        and not (len(r.chars) >= 6 and not any(c.char.isdigit() for c in r.chars))
    ]
    # A contrast-enhanced crop and its original share one source observation.
    by_frame = {}
    for result in valid_results:
        score = sum(c.prob for c in result.chars) / len(result.chars)
        key = (result.track_id, result.frame_no)
        if key not in by_frame or score > by_frame[key][0]:
            by_frame[key] = (score, result)
    valid_results = [result for _, result in by_frame.values()]

    if len(valid_results) < 2:
        votes = [
            PositionVote(
                index=i,
                char_or_unknown="?",
                confidence=0.0,
                votes=[],
                candidates=[],
                explanation="Fewer than two distinct frames available to confirm plate"
            )
            for i in range(10)
        ]
        return FusedPlate(
            text="??????????",
            per_position=votes,
            format_valid=False,
            frames_used=len(valid_results),
            mean_confidence=0.0,
            known_count=0,
            unknown_count=10
        )

    # Compute normalized frame weights
    sharpnesses = [max(1.0, r.sharpness) for r in valid_results]
    areas = [max(1.0, r.plate_area_px) for r in valid_results]
    mean_confs = [
        float(np.mean([c.prob for c in r.chars])) if r.chars else 0.50
        for r in valid_results
    ]
    frame_weights = compute_frame_weights(sharpnesses, areas, mean_confs)

    # Pick reference string using multi-frame consensus agreement and Indian syntax prior
    def candidate_score(r_idx: int) -> float:
        cand_str = "".join(c.char for c in valid_results[r_idx].chars)
        score = frame_weights[r_idx] * 10.0
        # High bonus for standard Indian license plate length (9 to 11 characters)
        if len(cand_str) == 10:
            score += 25.0
        elif 8 <= len(cand_str) <= 11:
            score += 15.0
        # Bonus for recognized Indian state prefix
        if len(cand_str) >= 2 and cand_str[:2] in {
            "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP",
            "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
            "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
        }:
            score += 20.0
        # Bonus for alignment agreement with other frames
        for o_idx, o_r in enumerate(valid_results):
            if o_idx == r_idx:
                continue
            o_str = "".join(c.char for c in o_r.chars)
            al1, al2 = align_sequences(cand_str, o_str)
            matches = sum(1 for a, b in zip(al1, al2) if a == b and a != "-")
            score += matches * frame_weights[o_idx] * 5.0
        return score

    best_idx = max(range(len(valid_results)), key=candidate_score)
    ref_chars = valid_results[best_idx].chars
    ref_text = "".join(c.char for c in ref_chars)
    target_len = len(ref_text)
    if target_len < 8 or target_len > 11:
        lengths = [len(r.chars) for r in valid_results if 8 <= len(r.chars) <= 11]
        target_len = int(round(float(np.median(lengths)))) if lengths else 10
        for r in valid_results:
            if len(r.chars) == target_len:
                ref_text = "".join(c.char for c in r.chars)
                break
        if len(ref_text) != target_len:
            ref_text = ref_text[:target_len].ljust(target_len, "X")

    target_len = len(ref_text)

    # Align each frame's predictions against the reference sequence using Needleman-Wunsch
    aligned_chars_per_frame: list[list[Any | None]] = []
    for r in valid_results:
        aligned = align_chars_to_reference(ref_text, r.chars)
        aligned_chars_per_frame.append(aligned)

    fused_chars: list[str] = []
    position_votes: list[PositionVote] = []
    raw_posteriors_per_pos: list[dict[str, float]] = []
    votes_per_pos: list[list[FrameVote]] = []

    for pos in range(target_len):
        class_evidence: dict[str, float] = {}
        votes: list[FrameVote] = []

        for frame_idx, r in enumerate(valid_results):
            c_item = aligned_chars_per_frame[frame_idx][pos]
            if c_item is None:
                continue

            w = frame_weights[frame_idx]
            main_ch = c_item.char.upper()
            main_p = float(c_item.prob)

            class_evidence[main_ch] = class_evidence.get(main_ch, 0.0) + (w * main_p)
            votes.append(FrameVote(
                frame_no=r.frame_no,
                char=main_ch,
                prob=round(main_p, 3),
                weight=round(w, 4)
            ))

            for alt in c_item.alternatives:
                alt_ch = alt.char.upper()
                alt_p = float(alt.prob)
                class_evidence[alt_ch] = class_evidence.get(alt_ch, 0.0) + (w * alt_p * alternative_weight)

        total_ev = sum(class_evidence.values())
        normalized: dict[str, float] = {
            ch: ev / max(1e-5, total_ev) for ch, ev in class_evidence.items()
        } if total_ev > 0 else {}

        raw_posteriors_per_pos.append(normalized)
        votes_per_pos.append(votes)

    # Execute probabilistic Viterbi / Dynamic Programming grammar parse over posteriors
    parsed = parse_plate(
        raw_posteriors_per_pos,
        accept_threshold=accept_threshold,
        margin_threshold=margin_threshold,
    )

    correction_by_index = {c.index: c for c in parsed.corrections}

    position_votes: list[PositionVote] = []
    known_confs: list[float] = []
    fused_chars: list[str] = []

    for pos in range(target_len):
        p_dist = raw_posteriors_per_pos[pos]
        sorted_raw = sorted(p_dist.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [
            {"char": ch, "prob": round(float(p), 3)}
            for ch, p in sorted_raw[:3]
        ]

        top_char, top_prob = sorted_raw[0] if sorted_raw else ("?", 0.0)
        runner_up_prob = sorted_raw[1][1] if len(sorted_raw) > 1 else 0.0
        margin = top_prob - runner_up_prob

        votes = votes_per_pos[pos]
        support = [v for v in votes if v.char == top_char and v.prob >= accept_threshold * 0.80]
        char_weights = [v.weight for v in votes if v.char == top_char]
        char_weighted_p = sum(v.weight * v.prob for v in votes if v.char == top_char)
        frame_conf = (char_weighted_p / max(1e-5, sum(char_weights))) if char_weights else 0.0
        confidence = min(top_prob, frame_conf) if char_weights else top_prob

        slot_label = parsed.slot_map[pos] if pos < len(parsed.slot_map) else "unknown"
        explanation = None

        if len(support) < 2:
            explanation = "Fewer than two distinct frames clearly read this character."
        elif confidence < accept_threshold or margin < margin_threshold:
            explanation = "Weak or conflicting OCR readings across frames."

        corr = correction_by_index.get(pos)
        if corr and explanation is None:
            twin_char = corr.corrected
            twin_support = [v for v in votes if v.char == twin_char and v.prob >= accept_threshold]
            if len(twin_support) >= 1 or p_dist.get(twin_char, 0.0) >= accept_threshold:
                decided_char = twin_char
                explanation = corr.reason
            else:
                decided_char = "?"
                explanation = "Candidate twin lacks sufficient multi-frame support"
        elif explanation is not None:
            decided_char = "?"
        else:
            if pos < len(parsed.text) and parsed.text[pos] == "?":
                decided_char = "?"
                explanation = "Character conflicts with slot grammar rule or invalid class"
            elif pos < len(parsed.text):
                decided_char = parsed.text[pos]
            else:
                decided_char = top_char

        fused_chars.append(decided_char)
        if decided_char != "?":
            known_confs.append(confidence)

        char_conf = confidence if decided_char != "?" else (top_prob or 0.0)

        position_votes.append(PositionVote(
            index=pos,
            char_or_unknown=decided_char,
            confidence=round(float(char_conf), 3),
            votes=votes,
            candidates=top_candidates,
            slot=slot_label,
            explanation=explanation
        ))

    fused_text = "".join(fused_chars)
    known_c = sum(1 for c in fused_text if c != "?")
    unknown_c = sum(1 for c in fused_text if c == "?")
    mean_conf = float(np.mean(known_confs)) if known_confs else 0.0

    return FusedPlate(
        text=fused_text,
        per_position=position_votes,
        format_valid=(unknown_c == 0 and validate_plate_grammar(fused_text)),
        frames_used=len(valid_results),
        mean_confidence=round(mean_conf, 3),
        known_count=known_c,
        unknown_count=unknown_c,
        template=parsed.template,
        slot_map=parsed.slot_map,
        slots=parsed.slots,
        corrections=[c.reason for c in parsed.corrections],
        flags=parsed.flags,
        format_confidence=parsed.format_confidence
    )
