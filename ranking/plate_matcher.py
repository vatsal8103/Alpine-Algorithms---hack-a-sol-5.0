"""Positional plate scoring with confusable pair penalty reduction and match map generation."""

from __future__ import annotations

CONFUSION_PAIRS = {
    ("O", "0"), ("0", "O"),
    ("B", "8"), ("8", "B"),
    ("I", "1"), ("1", "I"),
    ("S", "5"), ("5", "S"),
    ("Z", "2"), ("2", "Z"),
    ("D", "0"), ("0", "D"),
    ("G", "6"), ("6", "G"),
    ("T", "7"), ("7", "T"),
    ("V", "U"), ("U", "V"),
    ("Q", "0"), ("0", "Q"),
}


def score_plate(
    fused_text: str,
    registry_plate: str,
    position_confidences: list[float] | None = None,
    confusable_penalty: float = 0.35,
    mismatch_penalty: float = 1.00,
    gap_penalty: float = 0.40,
    high_conf_threshold: float = 0.60
) -> tuple[float, list[dict[str, str]], list[str], list[str], bool]:
    """Positional comparison between fused plate (with '?') and candidate plate.
    
    Returns:
        (plate_score, match_map, why_match, why_not, has_high_conf_conflict)
    """
    fused_clean = fused_text.upper().strip()
    reg_clean = registry_plate.upper().strip()

    max_len = max(len(fused_clean), len(reg_clean))
    if max_len == 0:
        return 0.0, [], [], ["Empty plate strings"], False

    confs = position_confidences or [0.85] * len(fused_clean)
    while len(confs) < len(fused_clean):
        confs.append(0.85)

    earned_points = 0.0
    max_possible = float(max_len)
    match_map: list[dict[str, str]] = []
    why_match: list[str] = []
    why_not: list[str] = []
    has_high_conf_conflict = False

    comp_len = min(len(fused_clean), len(reg_clean))

    for idx in range(comp_len):
        f_char = fused_clean[idx]
        r_char = reg_clean[idx]
        c_conf = confs[idx] if idx < len(confs) else 0.80

        if f_char == "?":
            # Wildcard / neutral: missing evidence does NOT earn positive match credit
            # Neither rewards nor penalizes candidate
            match_map.append({"pos": str(idx), "fused": "?", "cand": r_char, "status": "wildcard"})
        elif f_char == r_char:
            earned_points += (1.0 * c_conf)
            match_map.append({"pos": str(idx), "fused": f_char, "cand": r_char, "status": "match"})
            why_match.append(f"Position {idx}: verified match '{f_char}' (conf: {c_conf:.2f})")
        elif (f_char, r_char) in CONFUSION_PAIRS:
            earned_points += max(0.0, (1.0 - confusable_penalty) * c_conf)
            match_map.append({"pos": str(idx), "fused": f_char, "cand": r_char, "status": "confusable"})
            why_not.append(f"Position {idx}: confusable pair '{f_char}' vs '{r_char}'")
        else:
            penalty = mismatch_penalty * c_conf
            earned_points -= penalty
            match_map.append({"pos": str(idx), "fused": f_char, "cand": r_char, "status": "mismatch"})
            why_not.append(f"Position {idx}: mismatch '{f_char}' vs '{r_char}' (penalty: -{penalty:.2f})")
            if c_conf >= high_conf_threshold:
                has_high_conf_conflict = True

    # Length mismatch gap penalty
    len_delta = abs(len(fused_clean) - len(reg_clean))
    if len_delta > 0:
        earned_points -= (len_delta * gap_penalty)
        why_not.append(f"Plate length mismatch ({len(fused_clean)} chars observed vs {len(reg_clean)} in registry)")

    unknown_count = sum(1 for c in fused_clean if c == "?")
    if unknown_count > 0:
        if unknown_count >= len(fused_clean):
            why_not.append("All plate character slots unreadable in CCTV footage")
        else:
            why_not.append(f"{unknown_count} plate character slot(s) unreadable in CCTV footage ('?')")

    normalized_score = max(0.0, min(1.0, earned_points / max_possible))
    return round(normalized_score, 3), match_map, why_match, why_not, has_high_conf_conflict
