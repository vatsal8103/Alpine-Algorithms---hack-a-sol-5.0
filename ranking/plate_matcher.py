"""Positional plate scoring with Needleman-Wunsch alignment, confusable pair penalty reduction and match map generation."""

from __future__ import annotations

from fusion.align import align_sequences

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
    ("P", "D"), ("D", "P"),
    ("E", "C"), ("C", "E"),
    ("F", "C"), ("C", "F"),
    ("G", "C"), ("C", "G"),
    ("I", "J"), ("J", "I"),
    ("J", "U"), ("U", "J"),
    ("W", "U"), ("U", "W"),
    ("A", "K"), ("K", "A"),
    ("B", "0"), ("0", "B"),
    ("8", "3"), ("3", "8"),
    ("4", "A"), ("A", "4"),
    ("1", "L"), ("L", "1"),
    ("L", "4"), ("4", "L"),
    ("H", "J"), ("J", "H"),
    ("D", "O"), ("O", "D"),
    ("C", "O"), ("O", "C"),
}


def score_plate(
    fused_text: str,
    registry_plate: str,
    position_confidences: list[float] | None = None,
    confusable_penalty: float = 0.25,
    mismatch_penalty: float = 0.90,
    gap_penalty: float = 0.25,
    high_conf_threshold: float = 0.70
) -> tuple[float, list[dict[str, str]], list[str], list[str], bool]:
    """Positional comparison between fused plate (with '?') and candidate plate using alignment.
    
    Returns:
        (plate_score, match_map, why_match, why_not, has_high_conf_conflict)
    """
    fused_clean = fused_text.upper().strip()
    reg_clean = registry_plate.upper().strip()

    max_len = max(len(fused_clean), len(reg_clean))
    if max_len == 0:
        return 0.0, [], [], ["Empty plate strings"], False

    confs = position_confidences or [1.0] * len(fused_clean)
    while len(confs) < len(fused_clean):
        confs.append(1.0)

    # Use Needleman-Wunsch sequence alignment to handle single-digit RTOs, insertions, or occlusions
    a1, a2 = align_sequences(fused_clean, reg_clean)

    earned_points = 0.0
    match_map: list[dict[str, str]] = []
    why_match: list[str] = []
    why_not: list[str] = []
    has_high_conf_conflict = False

    fused_idx = 0
    for idx, (c1, c2) in enumerate(zip(a1, a2)):
        c_conf = confs[fused_idx] if (c1 != "-" and fused_idx < len(confs)) else 0.80

        if c1 == "?" or c2 == "?":
            # Wildcard / neutral: missing evidence does NOT earn positive match credit
            match_map.append({"pos": str(idx), "fused": "?", "cand": c2, "status": "wildcard"})
            if c1 != "-":
                fused_idx += 1
        elif c1 == "-" or c2 == "-":
            # Gap penalty for inserted/deleted characters
            earned_points -= gap_penalty
            match_map.append({"pos": str(idx), "fused": c1, "cand": c2, "status": "gap"})
            why_not.append(f"Position {idx}: alignment gap '{c1}' vs '{c2}'")
            if c1 != "-":
                fused_idx += 1
        elif c1 == c2:
            earned_points += (1.0 * c_conf)
            match_map.append({"pos": str(idx), "fused": c1, "cand": c2, "status": "match"})
            why_match.append(f"Position {idx}: verified match '{c1}' (conf: {c_conf:.2f})")
            fused_idx += 1
        elif (c1, c2) in CONFUSION_PAIRS:
            earned_points += max(0.0, (1.0 - confusable_penalty) * c_conf)
            match_map.append({"pos": str(idx), "fused": c1, "cand": c2, "status": "confusable"})
            why_match.append(f"Position {idx}: confusable pair '{c1}' vs '{c2}'")
            fused_idx += 1
        else:
            penalty = mismatch_penalty * c_conf
            earned_points -= penalty
            match_map.append({"pos": str(idx), "fused": c1, "cand": c2, "status": "mismatch"})
            why_not.append(f"Position {idx}: mismatch '{c1}' vs '{c2}' (penalty: -{penalty:.2f})")
            if c_conf >= high_conf_threshold:
                has_high_conf_conflict = True
            fused_idx += 1

    unknown_count = sum(1 for c in fused_clean if c == "?")
    if unknown_count > 0:
        if unknown_count >= len(fused_clean):
            why_not.append("All plate character slots unreadable in CCTV footage")
        else:
            why_not.append(f"{unknown_count} plate character slot(s) unreadable in CCTV footage ('?')")

    normalized_score = max(0.0, min(1.0, earned_points / float(max_len)))
    return round(normalized_score, 3), match_map, why_match, why_not, has_high_conf_conflict
