"""Probabilistic Indian License Plate Grammar Parser and Positional Viterbi Decoder.

Loads single source of truth from configs/plate_grammar.yaml.
Parses per-position character posterior distributions against Indian plate templates
(Standard, Delhi Category-Coded, Bharat BH Series, Special/Unrecognised).
Enforces CMVR 1989 rules:
- Strictly never invents a character unsupported by OCR evidence (alternatives).
- Penalizes I and O in series slots (CMVR Rule 51(8)).
- Validates 36 active State/UT codes and flags grandfathered legacy codes (TS, UA, OR, CT, DN).
- Detects Delhi RTO category letters (DL 8C AF 1234).
- Rejects number 0000.
- Replaces unsupported invalid slot characters with '?'.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Path to single source of truth
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "plate_grammar.yaml"


def _load_grammar_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_GRAMMAR_CFG = _load_grammar_config()

# State definitions
STATE_CODES: dict[str, dict[str, Any]] = _GRAMMAR_CFG.get("state_codes", {})
CURRENT_STATES = {k for k, v in STATE_CODES.items() if v.get("status") == "current"}
LEGACY_STATES = {k for k, v in STATE_CODES.items() if v.get("status") == "legacy"}
ALL_VALID_STATES = CURRENT_STATES | LEGACY_STATES

# Backwards compatibility export
INDIAN_STATES = ALL_VALID_STATES

# Noise tokens
NOISE_TOKENS = _GRAMMAR_CFG.get("noise_tokens", ["IND", "INDIA", "BHARAT"])

# Delhi category letters
DELHI_CATEGORIES = _GRAMMAR_CFG.get("delhi_category_letters", {
    "C": "Car / Private 4-Wheeler",
    "S": "Two-Wheeler",
    "P": "Passenger Bus",
    "R": "Auto-rickshaw",
    "T": "Taxi",
    "V": "Van",
    "Y": "Private Hire",
    "E": "Electric Vehicle"
})

# Confusion maps
CONFUSION_MAP = _GRAMMAR_CFG.get("confusion_map", {})
LETTER_TO_DIGIT_MAP = CONFUSION_MAP.get("letter_to_digit", {
    "O": {"to": "0", "penalty": 0.85},
    "I": {"to": "1", "penalty": 0.85},
    "Z": {"to": "2", "penalty": 0.80},
    "S": {"to": "5", "penalty": 0.80},
    "B": {"to": "8", "penalty": 0.80},
    "G": {"to": "6", "penalty": 0.75},
    "Q": {"to": "0", "penalty": 0.70},
    "D": {"to": "0", "penalty": 0.75},
    "T": {"to": "7", "penalty": 0.70},
    "A": {"to": "4", "penalty": 0.70},
})
DIGIT_TO_LETTER_MAP = CONFUSION_MAP.get("digit_to_letter", {
    "0": {"to": "O", "penalty": 0.85},
    "1": {"to": "I", "penalty": 0.85},
    "2": {"to": "Z", "penalty": 0.80},
    "5": {"to": "S", "penalty": 0.80},
    "8": {"to": "B", "penalty": 0.80},
    "6": {"to": "G", "penalty": 0.75},
    "4": {"to": "A", "penalty": 0.70},
    "7": {"to": "T", "penalty": 0.70},
})

# Backwards compatibility simple dicts
LETTER_TO_DIGIT_CONFUSIONS = {k: v["to"] for k, v in LETTER_TO_DIGIT_MAP.items()}
DIGIT_TO_LETTER_CONFUSIONS = {k: v["to"] for k, v in DIGIT_TO_LETTER_MAP.items()}

# Priors
PRIORS = _GRAMMAR_CFG.get("priors", {
    "invalid_class_penalty": 0.001,
    "unsupported_replacement_penalty": 0.0,
    "series_I_penalty": 0.05,
    "series_O_penalty": 0.05,
    "number_0000_penalty": 0.0001,
    "unknown_state_penalty": 0.01,
    "legacy_state_penalty": 0.95,
    "state_candidate_posterior_margin": 0.20,
})

# Regex for fast check
STANDARD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$")


@dataclass
class PositionCorrection:
    index: int
    original: str
    corrected: str
    reason: str
    support_prob: float


@dataclass
class ParsedSlot:
    id: str
    text: str
    start: int
    end: int
    allowed_class: str


@dataclass
class ParsedPlate:
    text: str
    raw_text: str
    template: str
    slot_map: list[str]
    slots: dict[str, str]
    corrections: list[PositionCorrection]
    flags: list[str]
    format_confidence: float
    is_valid: bool
    candidate_states: list[str] = field(default_factory=list)


def clean_noise(text: str) -> str:
    """Strip known noise tokens like IND, BHARAT, and non-alphanumeric symbols."""
    clean = text.upper().strip()
    for token in NOISE_TOKENS:
        clean = re.sub(rf"\b{token}\b", "", clean)
        if clean.startswith(token) and len(clean) > len(token) + 6:
            clean = clean[len(token):]
    clean = re.sub(r"[^A-Z0-9?↑/▲]", "", clean)
    return clean


def get_positional_prior(index: int, total_len: int = 10) -> str:
    """Return expected character type: 'alpha', 'digit', or 'any' for standard plate layout."""
    if total_len < 7:
        return "any"
    if index in (0, 1):
        return "alpha"
    if index in (2, 3):
        return "digit"
    if index >= total_len - 4:
        return "digit"
    if 4 <= index < total_len - 4:
        return "alpha"
    return "any"


def _generate_template_segmentations(total_len: int) -> list[tuple[str, list[tuple[str, str, int]]]]:
    """Generate all valid slot segmentations for templates matching total_len.
    
    Returns list of: (template_name, [(slot_id, slot_class, slot_len), ...])
    """
    valid_segmentations = []

    # 1. STANDARD: state (2, letter) + rto (1..2, digit) + series (1..3, letter) + number (4, digit)
    for rto_len in (2, 1):
        for series_len in (2, 1, 3):
            for num_len in (4,):
                if 2 + rto_len + series_len + num_len == total_len:
                    valid_segmentations.append((
                        "STANDARD",
                        [
                            ("state", "letter", 2),
                            ("rto", "digit", rto_len),
                            ("series", "letter", series_len),
                            ("number", "digit", num_len),
                        ]
                    ))

    # 2. DELHI: state (2, letter, DL) + rto (1..2, digit) + delhi_class (1, delhi_class) + series (1..2, letter) + number (4, digit)
    for rto_len in (1, 2):
        for series_len in (2, 1):
            for num_len in (4,):
                if 2 + rto_len + 1 + series_len + num_len == total_len:
                    valid_segmentations.append((
                        "DELHI",
                        [
                            ("state", "letter", 2),
                            ("rto", "digit", rto_len),
                            ("delhi_class", "delhi_class", 1),
                            ("series", "letter", series_len),
                            ("number", "digit", num_len),
                        ]
                    ))

    # 3. BHARAT (BH): bh_year (2, digit) + bh_tag (2, letter, BH) + number (4, digit) + series (1..2, letter)
    for series_len in (2, 1):
        if 2 + 2 + 4 + series_len == total_len:
            valid_segmentations.append((
                "BHARAT",
                [
                    ("bh_year", "digit", 2),
                    ("bh_tag", "letter", 2),
                    ("number", "digit", 4),
                    ("series", "letter", series_len),
                ]
            ))

    return valid_segmentations


def parse_plate(
    posteriors: list[dict[str, float]],
    accept_threshold: float = 0.50,
    margin_threshold: float = 0.15,
) -> ParsedPlate:
    """Dynamic programming / Viterbi parser over per-position character posteriors.
    
    posteriors: list of dicts {char: probability} for each aligned character position.
    Strictly forbids inventing characters absent from posteriors (evidence-based only).
    """
    total_len = len(posteriors)
    if total_len == 0:
        return ParsedPlate(
            text="", raw_text="", template="UNRECOGNISED", slot_map=[],
            slots={}, corrections=[], flags=["empty_input"],
            format_confidence=0.0, is_valid=False
        )

    # Derive raw top-1 characters
    raw_chars = []
    for p_dist in posteriors:
        if not p_dist:
            raw_chars.append("?")
        else:
            best_c = max(p_dist.items(), key=lambda x: x[1])[0]
            raw_chars.append(best_c)
    raw_text = "".join(raw_chars)

    # Check for special formats (Army, Diplomatic, Temporary, Trade)
    if any(c in raw_text for c in ("↑", "/", "▲")):
        return ParsedPlate(
            text=raw_text, raw_text=raw_text, template="UNRECOGNISED_FORMAT",
            slot_map=["special"] * total_len, slots={"raw": raw_text},
            corrections=[], flags=["special_military_format", "unrecognised_civilian_format", "unrecognised_format"],
            format_confidence=0.90, is_valid=False
        )
    if any(tok in raw_text for tok in ("CD", "CC", "UN")) and total_len <= 8:
        return ParsedPlate(
            text=raw_text, raw_text=raw_text, template="UNRECOGNISED_FORMAT",
            slot_map=["special"] * total_len, slots={"raw": raw_text},
            corrections=[], flags=["special_diplomatic_format", "unrecognised_civilian_format", "unrecognised_format"],
            format_confidence=0.85, is_valid=False
        )

    segmentations = _generate_template_segmentations(total_len)
    if not segmentations:
        return ParsedPlate(
            text=raw_text, raw_text=raw_text, template="UNRECOGNISED",
            slot_map=["unknown"] * total_len, slots={"raw": raw_text},
            corrections=[], flags=["unrecognised_length"],
            format_confidence=0.10, is_valid=False
        )

    best_score = -1.0
    best_result: dict[str, Any] | None = None

    for template_name, slots_spec in segmentations:
        current_pos = 0
        seg_score = 1.0
        seg_chars: list[str] = []
        seg_slot_map: list[str] = []
        seg_corrections: list[PositionCorrection] = []
        seg_flags: list[str] = []
        seg_slots_dict: dict[str, str] = {}
        candidate_states: list[str] = []

        for slot_id, slot_class, slot_len in slots_spec:
            slot_chars = []
            slot_start = current_pos
            slot_end = current_pos + slot_len

            for pos_idx in range(slot_start, slot_end):
                p_dist = posteriors[pos_idx] if pos_idx < len(posteriors) else {}
                seg_slot_map.append(slot_id)

                if not p_dist:
                    seg_chars.append("?")
                    slot_chars.append("?")
                    seg_score *= PRIORS["invalid_class_penalty"]
                    continue

                sorted_candidates = sorted(p_dist.items(), key=lambda x: x[1], reverse=True)
                top_char, top_p = sorted_candidates[0]

                char_chosen = top_char
                char_score = top_p
                correction_applied = False

                if slot_class == "digit":
                    if top_char.isdigit():
                        char_chosen = top_char
                        char_score = top_p
                    else:
                        twin_cand = LETTER_TO_DIGIT_MAP.get(top_char)
                        if twin_cand:
                            target_digit = twin_cand["to"]
                            twin_p = p_dist.get(target_digit, 0.0)
                            if twin_p > 0.0:
                                char_chosen = target_digit
                                char_score = twin_p * twin_cand["penalty"]
                                correction_applied = True
                                seg_corrections.append(PositionCorrection(
                                    index=pos_idx,
                                    original=top_char,
                                    corrected=target_digit,
                                    reason=f"Position {pos_idx}: {top_char} to {target_digit} by digit-slot rule (support: {twin_p:.2f})",
                                    support_prob=twin_p
                                ))
                            else:
                                char_chosen = "?"
                                char_score = PRIORS["invalid_class_penalty"]
                        else:
                            char_chosen = "?"
                            char_score = PRIORS["invalid_class_penalty"]

                elif slot_class == "letter":
                    if top_char.isalpha():
                        char_chosen = top_char
                        char_score = top_p
                        if slot_id in ("series", "bh_series"):
                            if top_char == "I":
                                char_score *= PRIORS["series_I_penalty"]
                                seg_flags.append(f"series_I_penalized_pos_{pos_idx}")
                            elif top_char == "O":
                                char_score *= PRIORS["series_O_penalty"]
                                seg_flags.append(f"series_O_penalized_pos_{pos_idx}")
                    else:
                        twin_cand = DIGIT_TO_LETTER_MAP.get(top_char)
                        if twin_cand:
                            target_letter = twin_cand["to"]
                            twin_p = p_dist.get(target_letter, 0.0)
                            if twin_p > 0.0:
                                char_chosen = target_letter
                                char_score = twin_p * twin_cand["penalty"]
                                correction_applied = True
                                seg_corrections.append(PositionCorrection(
                                    index=pos_idx,
                                    original=top_char,
                                    corrected=target_letter,
                                    reason=f"Position {pos_idx}: {top_char} to {target_letter} by letter-slot rule (support: {twin_p:.2f})",
                                    support_prob=twin_p
                                ))
                            else:
                                char_chosen = "?"
                                char_score = PRIORS["invalid_class_penalty"]
                        else:
                            char_chosen = "?"
                            char_score = PRIORS["invalid_class_penalty"]

                elif slot_class == "delhi_class":
                    if top_char in DELHI_CATEGORIES:
                        char_chosen = top_char
                        char_score = top_p * 1.20
                    else:
                        valid_alt = None
                        for c, p in sorted_candidates:
                            if c in DELHI_CATEGORIES and p > 0.10:
                                valid_alt = (c, p)
                                break
                        if valid_alt:
                            char_chosen = valid_alt[0]
                            char_score = valid_alt[1] * 0.80
                            seg_corrections.append(PositionCorrection(
                                index=pos_idx,
                                original=top_char,
                                corrected=char_chosen,
                                reason=f"Position {pos_idx}: {top_char} to {char_chosen} by Delhi vehicle class rule",
                                support_prob=valid_alt[1]
                            ))
                        else:
                            char_chosen = "?"
                            char_score = PRIORS["invalid_class_penalty"]

                seg_chars.append(char_chosen)
                slot_chars.append(char_chosen)
                seg_score *= max(1e-6, char_score)

            slot_val = "".join(slot_chars)
            seg_slots_dict[slot_id] = slot_val

            # Slot-specific semantic constraints
            if slot_id == "state":
                if template_name == "DELHI":
                    if slot_val == "DL":
                        seg_score *= 1.35
                    else:
                        seg_score *= 0.01
                elif slot_val in CURRENT_STATES:
                    seg_score *= 1.2
                elif slot_val in LEGACY_STATES:
                    seg_score *= PRIORS["legacy_state_penalty"]
                    seg_flags.append(f"legacy_state_code_{slot_val}")
                else:
                    p0 = posteriors[slot_start] if slot_start < len(posteriors) else {}
                    p1 = posteriors[slot_start + 1] if slot_start + 1 < len(posteriors) else {}
                    c0_cands = [c for c, _ in sorted(p0.items(), key=lambda x: x[1], reverse=True)[:3]]
                    c1_cands = [c for c, _ in sorted(p1.items(), key=lambda x: x[1], reverse=True)[:3]]

                    matching_states = []
                    for c0 in c0_cands:
                        for c1 in c1_cands:
                            code = c0 + c1
                            if code in ALL_VALID_STATES:
                                prob_joint = p0.get(c0, 0.0) * p1.get(c1, 0.0)
                                matching_states.append((code, prob_joint))

                    if matching_states:
                        matching_states.sort(key=lambda x: x[1], reverse=True)
                        best_match_code, best_match_p = matching_states[0]
                        candidate_states = [m[0] for m in matching_states]

                        if len(matching_states) == 1 or (len(matching_states) > 1 and (best_match_p - matching_states[1][1] >= PRIORS["state_candidate_posterior_margin"])):
                            seg_chars[slot_start] = best_match_code[0]
                            seg_chars[slot_start + 1] = best_match_code[1]
                            seg_slots_dict["state"] = best_match_code
                            seg_flags.append(f"state_candidate_proposed_{best_match_code}")
                            seg_score *= 0.85
                        else:
                            seg_chars[slot_start] = "?"
                            seg_chars[slot_start + 1] = "?"
                            seg_slots_dict["state"] = "??"
                            seg_flags.append("unknown_state_ambiguous")
                            seg_score *= PRIORS["unknown_state_penalty"]
                    else:
                        seg_chars[slot_start] = "?"
                        seg_chars[slot_start + 1] = "?"
                        seg_slots_dict["state"] = "??"
                        seg_flags.append("unknown_state_no_valid_candidates")
                        seg_score *= PRIORS["unknown_state_penalty"]

            elif slot_id == "bh_tag":
                if slot_val == "BH":
                    seg_score *= 1.3
                else:
                    seg_chars[slot_start] = "?"
                    seg_chars[slot_start + 1] = "?"
                    seg_flags.append("invalid_bh_tag")
                    seg_score *= 0.001

            elif slot_id == "number":
                if slot_val.replace("?", "") == "0" * len(slot_val.replace("?", "")):
                    seg_score *= PRIORS["number_0000_penalty"]
                    seg_flags.append("invalid_serial_0000_rejected")

            current_pos = slot_end

        norm_score = seg_score ** (1.0 / max(1, total_len))

        if norm_score > best_score:
            best_score = norm_score
            best_result = {
                "template": template_name,
                "chars": seg_chars,
                "slot_map": seg_slot_map,
                "slots": seg_slots_dict,
                "corrections": seg_corrections,
                "flags": seg_flags,
                "score": norm_score,
                "candidate_states": candidate_states,
            }

    if not best_result:
        return ParsedPlate(
            text=raw_text, raw_text=raw_text, template="UNRECOGNISED",
            slot_map=["unknown"] * total_len, slots={"raw": raw_text},
            corrections=[], flags=["no_valid_parse"],
            format_confidence=0.0, is_valid=False
        )

    fused_str = "".join(best_result["chars"])
    has_unknowns = "?" in fused_str
    is_valid = (not has_unknowns) and ("invalid_serial_0000_rejected" not in best_result["flags"])

    return ParsedPlate(
        text=fused_str,
        raw_text=raw_text,
        template=best_result["template"],
        slot_map=best_result["slot_map"],
        slots=best_result["slots"],
        corrections=best_result["corrections"],
        flags=best_result["flags"],
        format_confidence=round(float(min(1.0, best_result["score"] * 1.2)), 3),
        is_valid=is_valid,
        candidate_states=best_result["candidate_states"]
    )


def parse_plate_text(plate_text: str, allowed_confusion_twins: bool = True) -> ParsedPlate:
    """Convenience wrapper to parse a deterministic plate string.
    
    Converts each character into a posterior distribution.
    If a character is confusable (e.g. 'O' in a digit slot), includes the twin in posteriors
    only if allowed_confusion_twins is True.
    """
    clean = clean_noise(plate_text)
    posteriors = []
    for ch in clean:
        dist = {ch: 0.95}
        if allowed_confusion_twins:
            if ch in LETTER_TO_DIGIT_MAP:
                dist[LETTER_TO_DIGIT_MAP[ch]["to"]] = 0.35
            elif ch in DIGIT_TO_LETTER_MAP:
                dist[DIGIT_TO_LETTER_MAP[ch]["to"]] = 0.35
        posteriors.append(dist)
    return parse_plate(posteriors)


def is_valid_indian_plate(plate_text: str) -> bool:
    """Strict verification of Indian vehicle registration plate syntax (CMVR 1989 / MoRTH).
    
    Returns True only if:
    1. The string is non-empty and between 8 and 11 alphanumeric characters.
    2. Does not contain '?' or noise tokens / words.
    3. Matches Standard, Delhi, or Bharat series templates with recognized state codes.
    """
    clean = clean_noise(plate_text)
    if not clean or len(clean) < 8 or len(clean) > 11 or "?" in clean:
        return False
    if not (any(c.isalpha() for c in clean) and any(c.isdigit() for c in clean)):
        return False

    # Check Bharat (BH) series: [2 digits year]BH[4 digits number][1-2 letters series]
    m_bh = re.match(r"^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$", clean)
    if m_bh:
        num, ser = m_bh.group(2), m_bh.group(3)
        return num != "0000" and not ("I" in ser or "O" in ser)

    # Check Standard series: [2 letters State][1-2 digits RTO][0-3 letters Series][1-4 digits Number]
    m_std = re.match(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{1,4})$", clean)
    if m_std:
        st, rto, ser, num = m_std.groups()
        if st in ALL_VALID_STATES and num != "0000":
            return not ("I" in ser or "O" in ser)

    # Check Delhi category series: DL[1-2 digits][C/S/P/R/T/V/Y/E][1-2 letters][4 digits]
    m_dl = re.match(r"^DL([0-9]{1,2})([CSPRT VYE])([A-Z]{1,2})([0-9]{4})$", clean)
    if m_dl:
        num, ser = m_dl.group(4), m_dl.group(3)
        return num != "0000" and not ("I" in ser or "O" in ser)

    return False


def validate_plate_grammar(plate_text: str) -> bool:
    """Validate whether plate text matches Indian legal grammar rules."""
    clean = clean_noise(plate_text)
    if not clean:
        return False
    if not is_valid_indian_plate(clean):
        return False
    parsed = parse_plate_text(clean, allowed_confusion_twins=False)
    return parsed.is_valid
