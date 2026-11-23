"""TRACE-X Evidential Fusion between Fast Multi-Frame OCR and VLM Predictions.

Solves the core forensic problem:
1. When fast OCR produces '?' in ambiguous positions (e.g. 'MP13?U2058'),
   the VLM confirms or resolves the question marks to concrete characters ('Z').
2. When fast OCR and VLM agree on characters (e.g. 'M', 'P', '1', '3', 'U', '2', '0', '5', '8'),
   the evidential confidence is boosted and averaged.
3. When fast OCR is completely unreadable ('??????????' or 'NOT FOUND'), VLM populates
   the plate ONLY IF it satisfies Indian MoRTH registration grammar.
4. If VLM reading is invalid (e.g. English words, wrong state code, shop signs) or vehicle has no plate,
   enforces Forensic Safe Abstention ('NOT FOUND').
"""

from __future__ import annotations

import re
from typing import Any
import numpy as np

from fusion.grammar import is_valid_indian_plate, validate_plate_grammar
from pipeline.schemas import CharAlternative, FrameVote, FusedPlate, PositionVote, VehicleAttributes


def normalize_vlm_to_10_slots(vlm_text: str) -> str:
    """Normalize VLM plate reading into a clean uppercase string without dummy padding."""
    if not vlm_text:
        return ""
    clean = re.sub(r"[^A-Za-z0-9]", "", vlm_text).upper()
    clean = re.sub(r"^(IND|INO|IMD|IRD|IHD|1ND)", "", clean)
    if clean in {"NONE", "UNREADABLE", "NOPLATE", "NOTFOUND", "UNKNOWN", "THELICENSE"}:
        return ""
    if len(clean) < 7 or len(clean) > 11:
        return ""
    return clean


def fuse_ocr_and_vlm(
    fused_plate: FusedPlate | None,
    vlm_text: str,
    vlm_confidence: float = 0.88
) -> tuple[FusedPlate, dict[str, Any]]:
    """Fuse multi-frame OCR FusedPlate with VLM reading, strictly enforcing MoRTH plate syntax."""
    clean_vlm = normalize_vlm_to_10_slots(vlm_text)
    vlm_valid = is_valid_indian_plate(clean_vlm) if clean_vlm else False

    # If VLM provided no valid plate or text failed MoRTH grammar
    if not clean_vlm or not vlm_valid:
        # Check if fast OCR had a valid plate
        if fused_plate and fused_plate.text not in ("NOT FOUND", "??????????") and fused_plate.format_valid:
            return fused_plate, {
                "resolved_chars": [],
                "agreed_chars": [],
                "corrected_chars": [],
                "old_text": fused_plate.text,
                "new_text": fused_plate.text,
                "old_conf": fused_plate.mean_confidence,
                "new_conf": fused_plate.mean_confidence,
                "status": "vlm_rejected_prior_valid_kept"
            }
        # Safe abstention: honestly report NOT FOUND
        abstain_plate = FusedPlate(
            text="NOT FOUND",
            per_position=[],
            format_valid=False,
            frames_used=fused_plate.frames_used if fused_plate else 0,
            mean_confidence=0.0,
            known_count=0,
            unknown_count=0
        )
        return abstain_plate, {
            "resolved_chars": [],
            "agreed_chars": [],
            "corrected_chars": [],
            "old_text": fused_plate.text if fused_plate else "NOT FOUND",
            "new_text": "NOT FOUND",
            "old_conf": fused_plate.mean_confidence if fused_plate else 0.0,
            "new_conf": 0.0,
            "status": "abstain_not_found"
        }

    resolved_chars: list[str] = []
    agreed_chars: list[str] = []
    corrected_chars: list[str] = []

    # Case A: Previous fused plate has per-position votes with matching length
    target_len = len(clean_vlm)
    if fused_plate and fused_plate.per_position and len(fused_plate.per_position) == target_len:
        new_votes: list[PositionVote] = []
        for i in range(target_len):
            old_pv = fused_plate.per_position[i]
            old_ch = old_pv.char_or_unknown
            old_conf = float(old_pv.confidence or 0.0)
            vlm_ch = clean_vlm[i]

            if old_ch == "?":
                # Question mark resolved by VLM!
                new_ch = vlm_ch
                new_conf = vlm_confidence
                explanation = f"Question mark resolved by Qwen2-VL as '{new_ch}'"
                resolved_chars.append(f"Pos #{i+1}: '?' -> '{new_ch}'")
                new_votes.append(PositionVote(
                    index=i,
                    char_or_unknown=new_ch,
                    confidence=round(new_conf, 3),
                    votes=old_pv.votes or [FrameVote(frame_no=1, char=new_ch, prob=round(new_conf, 3), weight=1.0)],
                    candidates=[{"char": new_ch, "prob": new_conf}],
                    explanation=explanation
                ))
            elif old_ch == vlm_ch:
                # Both models agree! Boost confidence
                new_conf = min(0.99, (old_conf + vlm_confidence) / 2.0 + 0.05)
                explanation = f"Cross-verified: Fast OCR ({round(old_conf*100)}%) and Qwen2-VL agreement"
                agreed_chars.append(f"Pos #{i+1}: '{old_ch}' (conf: {round(old_conf*100)}% -> {round(new_conf*100)}%)")
                new_votes.append(PositionVote(
                    index=i,
                    char_or_unknown=old_ch,
                    confidence=round(new_conf, 3),
                    votes=old_pv.votes,
                    candidates=old_pv.candidates,
                    explanation=explanation
                ))
            else:
                # Disagreement: take VLM if OCR had low confidence
                if old_conf < 0.65:
                    new_ch = vlm_ch
                    new_conf = vlm_confidence * 0.92
                    explanation = f"Corrected to Qwen2-VL '{new_ch}' (Fast OCR had low confidence {round(old_conf*100)}% on '{old_ch}')"
                    corrected_chars.append(f"Pos #{i+1}: '{old_ch}' ({round(old_conf*100)}%) -> '{new_ch}'")
                    new_votes.append(PositionVote(
                        index=i,
                        char_or_unknown=new_ch,
                        confidence=round(new_conf, 3),
                        votes=old_pv.votes,
                        candidates=[{"char": new_ch, "prob": new_conf}, {"char": old_ch, "prob": old_conf}],
                        explanation=explanation
                    ))
                else:
                    new_votes.append(old_pv)

        # Eliminate any accidental question marks remaining
        for pos_idx in range(len(new_votes)):
            if new_votes[pos_idx].char_or_unknown == "?":
                replacement_ch = clean_vlm[pos_idx]
                new_votes[pos_idx].char_or_unknown = replacement_ch
                new_votes[pos_idx].confidence = max(0.50, new_votes[pos_idx].confidence)
                new_votes[pos_idx].explanation = f"Resolved to '{replacement_ch}' by Qwen2-VL"
                if not new_votes[pos_idx].votes:
                    new_votes[pos_idx].votes = [FrameVote(frame_no=1, char=replacement_ch, prob=round(new_votes[pos_idx].confidence, 3), weight=1.0)]
                if not new_votes[pos_idx].candidates:
                    new_votes[pos_idx].candidates = [{"char": replacement_ch, "prob": round(new_votes[pos_idx].confidence, 3)}]

        new_text = "".join(v.char_or_unknown for v in new_votes)
        mean_c = round(float(np.mean([v.confidence for v in new_votes])), 3)

        updated_fused = FusedPlate(
            text=new_text,
            per_position=new_votes,
            format_valid=is_valid_indian_plate(new_text),
            frames_used=max(1, fused_plate.frames_used),
            mean_confidence=mean_c,
            known_count=len(new_text),
            unknown_count=0,
            template=fused_plate.template or "STANDARD",
            corrections=fused_plate.corrections + ([f"VLM resolved {len(resolved_chars)} missing characters"] if resolved_chars else []),
            flags=fused_plate.flags
        )
        return updated_fused, {
            "resolved_chars": resolved_chars,
            "agreed_chars": agreed_chars,
            "corrected_chars": corrected_chars,
            "old_text": fused_plate.text,
            "new_text": new_text,
            "old_conf": fused_plate.mean_confidence,
            "new_conf": mean_c
        }

    # Case B: Previously unreadable plate, different length, or no prior position votes
    new_votes = []
    for i, ch in enumerate(clean_vlm):
        conf = vlm_confidence
        expl = f"Identified by Qwen2-VL as '{ch}'"
        new_votes.append(PositionVote(
            index=i,
            char_or_unknown=ch,
            confidence=round(conf, 3),
            votes=[FrameVote(frame_no=1, char=ch, prob=round(conf, 3), weight=1.0)],
            candidates=[{"char": ch, "prob": round(conf, 3)}],
            explanation=expl
        ))

    mean_c = round(float(np.mean([v.confidence for v in new_votes])), 3)

    updated_fused = FusedPlate(
        text=clean_vlm,
        per_position=new_votes,
        format_valid=True,
        frames_used=max(1, fused_plate.frames_used if fused_plate else 1),
        mean_confidence=mean_c,
        known_count=len(clean_vlm),
        unknown_count=0,
        template="STANDARD"
    )
    return updated_fused, {
        "resolved_chars": [f"Directly resolved to '{clean_vlm}' via Qwen2-VL"],
        "agreed_chars": [],
        "corrected_chars": [],
        "old_text": fused_plate.text if fused_plate else "NOT FOUND",
        "new_text": clean_vlm,
        "old_conf": fused_plate.mean_confidence if fused_plate else 0.0,
        "new_conf": mean_c
    }
