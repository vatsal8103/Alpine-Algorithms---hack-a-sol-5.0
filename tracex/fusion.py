"""Multi-frame character fusion (team-authored algorithm).

The fusion is a confidence-weighted, grammar-aware vote.  It deliberately
has three abstention gates per character position:

1. **Support** - the total evidence weight at that position must exceed a
   floor.  A single very weak OCR reading cannot fill a character.
2. **Vote share** - the winning character must beat the alternatives by a
   margin.  If frames disagree, the position becomes ``?``.
3. **Calibrated confidence** - the per-character confidence must exceed the
   configured floor.

Characters are only ever copied from an observed reading (or an explicit
investigator hint).  No registry lookup, language model or prior is used to
fill a position.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .config import FusionConfig
from .ocr import OCRLine
from .plate import (
    UNKNOWN,
    AlignedReading,
    align_reading,
    candidate_patterns,
    class_pattern,
    clean_text,
    is_bh_series,
    is_valid_plate,
    pattern_string,
    plate_likeness,
)
from .schemas import CharAlternative, FusedPlate, PlateObservation


def observation_from_ocr(
    frame_index: int,
    timestamp: float,
    line: OCRLine,
    quality: float,
) -> PlateObservation:
    return PlateObservation(
        frame_index=frame_index,
        timestamp=timestamp,
        text=clean_text(line.text),
        confidence=float(max(0.0, min(1.0, line.confidence))),
        char_confidences=[float(max(0.0, min(1.0, c))) for c in line.char_confidences],
        quality=float(max(0.0, min(1.0, quality))),
        source="ocr",
        backend=line.backend,
    )


def observation_from_hint(hint: str, confidence: float = 0.95) -> PlateObservation:
    text = clean_text(hint)
    return PlateObservation(
        frame_index=-1,
        timestamp=0.0,
        text=text,
        confidence=confidence,
        char_confidences=[confidence] * len(text),
        quality=1.0,
        source="investigator",
        backend="hint",
    )


def _weight(obs: PlateObservation) -> float:
    base = max(0.02, min(1.0, obs.confidence)) ** 1.35
    quality = 0.55 + 0.45 * max(0.0, min(1.0, obs.quality))
    source = 1.25 if obs.source == "investigator" else 1.0
    return max(0.01, base * quality * source)


def _align_for_length(obs: PlateObservation, length: int) -> tuple[AlignedReading, str]:
    """Choose the best grammar hypothesis for this reading and length."""

    text = clean_text(obs.text)
    patterns: list[tuple[str, int, bool]] = []
    if is_bh_series(text):
        patterns.append((class_pattern(length, bh=True), 2, True))
    patterns.extend(candidate_patterns(length))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, int, bool]] = []
    for p in patterns:
        if p[0] not in seen and "?" not in p[0]:
            seen.add(p[0])
            unique.append(p)

    best: AlignedReading | None = None
    best_score = -1.0
    for pattern, _rto, _bh in unique:
        aligned = align_reading(
            text,
            length,
            pattern=pattern,
            char_confidences=obs.char_confidences,
            line_confidence=obs.confidence,
            class_correction=True,
        )
        # Prefer alignment quality, then plate-likeness, then longer patterns.
        score = 0.70 * aligned.match_score + 0.30 * plate_likeness(text)
        if aligned.known_chars < 4:
            score -= 0.15
        if score > best_score:
            best_score = score
            best = aligned
    if best is None:
        best = align_reading(text, length, char_confidences=obs.char_confidences, line_confidence=obs.confidence)
    return best, best.pattern


def _choose_length(observations: list[PlateObservation], cfg: FusionConfig) -> int:
    """Pick the canonical plate length by aligning every reading to every hypothesis."""

    observed_lengths = [len(clean_text(o.text)) for o in observations if clean_text(o.text)]
    if not observed_lengths:
        return 10
    # A short investigator hint such as "JH05" or "6799" is a fragment, not a
    # complete 4-character plate.  Align it to the common 10-character layout.
    if max(observed_lengths) <= 7:
        return 10
    candidates = set(observed_lengths)
    candidates.update({9, 10})
    candidates = {L for L in candidates if cfg.min_plate_len <= L <= cfg.max_plate_len}
    if not candidates:
        candidates = {max(cfg.min_plate_len, min(cfg.max_plate_len, 10))}

    weighted_median = _weighted_median(observed_lengths, [_weight(o) for o in observations if clean_text(o.text)])
    best_length = 10
    best_score = -1e9
    for length in sorted(candidates):
        total = 0.0
        total_weight = 0.0
        for obs in observations:
            if not clean_text(obs.text):
                continue
            aligned, _ = _align_for_length(obs, length)
            w = _weight(obs)
            total += w * (0.65 * aligned.match_score + 0.35 * plate_likeness(obs.text))
            total_weight += w
        if total_weight <= 0:
            continue
        score = total / total_weight
        # Mild prior: most Indian plates are 10 characters (2+2+2+4).
        if length == 10:
            score += 0.035
        elif length == 9:
            score += 0.012
        score -= 0.035 * abs(length - weighted_median)
        if score > best_score:
            best_score = score
            best_length = length
    return best_length


def _weighted_median(values: list[int], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float(sorted(values)[len(values) // 2])
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return float(value)
    return float(pairs[-1][0])


def fuse_plate(observations: list[PlateObservation], cfg: FusionConfig) -> FusedPlate:
    """Fuse per-frame readings into a plate with ``?`` for unknown positions."""

    observations = [
        o
        for o in observations
        if clean_text(o.text) and sum(1 for c in clean_text(o.text) if c != UNKNOWN) >= cfg.min_chars_per_reading
    ]
    warnings: list[str] = []
    if not observations:
        length = 10
        return FusedPlate(
            text=UNKNOWN * length,
            pattern=class_pattern(length),
            char_confidences=[0.0] * length,
            support=[0] * length,
            alternatives=[[] for _ in range(length)],
            known_chars=0,
            unknown_positions=list(range(length)),
            mean_confidence=0.0,
            n_frames_observed=0,
            n_frames_used=0,
            warnings=["No plate reading passed the minimum character count."],
        )

    length = _choose_length(observations, cfg)
    bh_mode = any(is_bh_series(clean_text(o.text)) for o in observations)
    global_pattern = class_pattern(length, bh=bh_mode) if bh_mode else class_pattern(length)

    votes: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(length)]
    conf_sum: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(length)]
    support_count: list[defaultdict[str, int]] = [defaultdict(int) for _ in range(length)]
    observed_weight = [0.0] * length
    frame_weight = [0.0] * length
    aligned_rows: list[tuple[PlateObservation, AlignedReading]] = []
    corrections: list[dict] = []

    for obs in observations:
        aligned, _pattern = _align_for_length(obs, length)
        aligned_rows.append((obs, aligned))
        base_w = _weight(obs)
        for i, ch in enumerate(aligned.chars):
            frame_weight[i] += base_w
            if ch == UNKNOWN:
                continue
            char_conf = max(0.05, min(1.0, aligned.confidences[i] if i < len(aligned.confidences) else obs.confidence))
            w = base_w * char_conf
            votes[i][ch] += w
            conf_sum[i][ch] += w * char_conf
            support_count[i][ch] += 1
            observed_weight[i] += w
        for correction in aligned.corrections:
            corrections.append(
                {
                    **correction,
                    "frame_index": obs.frame_index,
                    "source": obs.source,
                }
            )

    chars: list[str] = []
    char_confidences: list[float] = []
    support: list[int] = []
    alternatives: list[list[CharAlternative]] = []
    unknown_positions: list[int] = []

    for i in range(length):
        if observed_weight[i] < cfg.min_support_weight or not votes[i]:
            chars.append(UNKNOWN)
            char_confidences.append(0.0)
            support.append(0)
            alternatives.append([])
            unknown_positions.append(i)
            continue
        ranked = sorted(votes[i].items(), key=lambda kv: kv[1], reverse=True)
        top_char, top_w = ranked[0]
        vote_share = top_w / max(1e-9, observed_weight[i])
        mean_conf = conf_sum[i][top_char] / max(1e-9, top_w)
        n_support = support_count[i][top_char]
        n_total = sum(support_count[i].values())
        agreement = n_support / max(1, n_total)
        support_factor = 0.85 + 0.15 * (1.0 - math.exp(-n_support / 2.0))
        confidence = mean_conf * (vote_share ** 0.55) * (0.90 + 0.10 * agreement) * support_factor
        confidence = max(0.0, min(1.0, confidence))

        alts: list[CharAlternative] = []
        for alt_char, alt_w in ranked[:3]:
            alts.append(
                CharAlternative(
                    char=alt_char,
                    weight=float(alt_w / max(1e-9, observed_weight[i])),
                    confidence=float(conf_sum[i][alt_char] / max(1e-9, alt_w)),
                    support=int(support_count[i][alt_char]),
                )
            )
        alternatives.append(alts)

        if vote_share < cfg.min_vote_share or confidence < cfg.min_char_confidence:
            chars.append(UNKNOWN)
            char_confidences.append(0.0)
            support.append(0)
            unknown_positions.append(i)
            if len(ranked) > 1 and vote_share < 0.75:
                warnings.append(
                    f"Position {i + 1} left unknown: readings disagree "
                    f"({', '.join(f'{c}:{w:.2f}' for c, w in ranked[:3])})."
                )
        else:
            chars.append(top_char)
            char_confidences.append(confidence)
            support.append(n_support)

    text = "".join(chars)
    known_chars = sum(1 for c in text if c != UNKNOWN)
    mean_confidence = (
        sum(c for c in char_confidences if c > 0) / known_chars if known_chars else 0.0
    )

    # Deduplicate corrections, keeping the highest-confidence occurrence.
    deduped: dict[tuple[int, str, str], dict] = {}
    for c in corrections:
        key = (int(c["position"]), str(c["observed"]), str(c["corrected"]))
        if key not in deduped or c.get("confidence", 0) > deduped[key].get("confidence", 0):
            deduped[key] = c
    corrections = sorted(deduped.values(), key=lambda c: (c["position"], -float(c.get("confidence", 0))))

    frames_observed = {o.frame_index for o in observations if o.frame_index >= 0}
    frames_used = {
        o.frame_index
        for o, aligned in aligned_rows
        if o.frame_index >= 0 and aligned.known_chars > 0
    }
    if len(frames_used) == 1:
        warnings.append("Only one frame contributed readable characters; treat the result as weak evidence.")
    if known_chars < 4:
        warnings.append("Fewer than four characters were recovered; registry matching may be ambiguous.")
    if text and not is_valid_plate(text):
        warnings.append("Fused plate does not fully satisfy the Indian plate grammar; unknown positions were preserved.")
    if any(c["source"] == "investigator" for c in corrections):
        warnings.append("An investigator-provided hint was used as evidence.")

    return FusedPlate(
        text=text,
        pattern=pattern_string(text) if text else global_pattern,
        char_confidences=[round(c, 4) for c in char_confidences],
        support=support,
        alternatives=alternatives,
        known_chars=known_chars,
        unknown_positions=unknown_positions,
        mean_confidence=round(float(mean_confidence), 4),
        n_frames_observed=len(frames_observed),
        n_frames_used=len(frames_used),
        class_corrections=corrections,
        warnings=warnings,
    )
