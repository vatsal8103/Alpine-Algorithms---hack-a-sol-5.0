"""Registry matching and ranking (team-authored algorithm).

For every registry entry we compute:

* a **partial-plate compatibility score** over only the query positions that
  were actually recovered (``?`` positions are neutral: they neither support
  nor contradict a candidate);
* **attribute compatibility** for colour, body type and optional make/model;
* a **confidence** that combines the match score with evidence sufficiency
  and explicit penalties for high-confidence plate conflicts.

The shortlist is the top-5 by confidence.  If the best candidate does not
clear the confidence/score/margin gates, the decision is
``NO_CONFIDENT_MATCH`` and the ranked list is returned only as *near misses*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .attributes import body_similarity, color_similarity, plate_style_similarity
from .config import MatchConfig
from .plate import CONFUSION, UNKNOWN, clean_text, confusion_match
from .registry import Registry
from .schemas import FusedPlate, MatchOutcome, MatchResult, RegistryEntry, VehicleAttributes


@dataclass
class PlateSimilarity:
    score: float = 0.0
    known_chars: int = 0
    unknown_chars: int = 0
    known_weight: float = 0.0
    matches: int = 0
    mismatches: int = 0
    high_conf_mismatches: int = 0
    confusion_matches: int = 0
    mismatch_positions: list[tuple[int, str, str, float]] = field(default_factory=list)
    exact: bool = False
    length_delta: int = 0
    reasons: list[str] = field(default_factory=list)


def _pad_query(query: str, conf: list[float], target_len: int) -> tuple[str, list[float], int]:
    """Pad a shorter query with ``?`` at the best offset."""

    diff = target_len - len(query)
    if diff <= 0:
        return query, conf, 0
    best = None
    best_score = -1.0
    for offset in range(diff + 1):
        padded = UNKNOWN * offset + query + UNKNOWN * (diff - offset)
        padded_conf = [0.0] * offset + conf + [0.0] * (diff - offset)
        # Score by how many known characters land in class-compatible slots
        # is handled by the caller; here use a simple count of non-unknown
        # overlap with the target length (all offsets tie).  Prefer the centre
        # as a deterministic tie-break.
        centre = abs(offset - diff / 2)
        score = -centre
        if score > best_score:
            best_score = score
            best = (padded, padded_conf, offset)
    assert best is not None
    return best


def compare_plate(
    query: str,
    candidate: str,
    query_conf: list[float] | None,
    cfg: MatchConfig,
) -> PlateSimilarity:
    q = clean_text(query)
    c = clean_text(candidate)
    conf = list(query_conf or [])
    if len(conf) < len(q):
        conf.extend([0.5] * (len(q) - len(conf)))
    conf = [max(0.0, min(1.0, v)) for v in conf[: len(q)]]

    length_delta = abs(len(q) - len(c))
    if len(q) < len(c):
        q, conf, _offset = _pad_query(q, conf, len(c))
    elif len(c) < len(q):
        # Registry plates should be complete; pad the candidate defensively.
        c = c + UNKNOWN * (len(q) - len(c))
        length_delta = len(q) - len(c) + length_delta

    sim = PlateSimilarity()
    weighted_sum = 0.0
    weight_total = 0.0
    for i, (qc, cc) in enumerate(zip(q, c)):
        if qc == UNKNOWN:
            continue
        if cc == UNKNOWN:
            continue
        qconf = conf[i] if i < len(conf) else 0.5
        weight = max(0.05, qconf)
        if qc == cc:
            char_sim = 1.0
            sim.matches += 1
        elif confusion_match(qc, cc):
            char_sim = cfg.confusion_similarity
            sim.confusion_matches += 1
        else:
            char_sim = 0.0
            sim.mismatches += 1
            if qconf >= cfg.high_conf_threshold:
                sim.high_conf_mismatches += 1
            sim.mismatch_positions.append((i, qc, cc, qconf))
        weighted_sum += weight * char_sim
        weight_total += weight
        sim.known_chars += 1
        sim.known_weight += weight

    sim.unknown_chars = len(q) - sim.known_chars
    sim.score = weighted_sum / weight_total if weight_total > 0 else 0.0
    sim.exact = sim.unknown_chars == 0 and sim.mismatches == 0 and sim.confusion_matches == 0 and len(q) == len(c)
    if length_delta:
        sim.score *= max(0.55, 1.0 - 0.08 * length_delta)
        sim.reasons.append(f"Query and registry plate lengths differ by {length_delta}; alignment used.")
    if sim.exact:
        sim.reasons.append(f"Exact plate match on all {len(q)} characters.")
    else:
        sim.reasons.append(
            f"Plate agrees on {sim.matches}/{sim.known_chars} known characters"
            + (f"; {sim.unknown_chars} position(s) unknown in the query." if sim.unknown_chars else ".")
        )
        if sim.confusion_matches:
            sim.reasons.append(
                f"{sim.confusion_matches} character(s) are OCR-confusion equivalents and scored partially."
            )
        for pos, qc, cc, qconf in sim.mismatch_positions[:4]:
            sim.reasons.append(
                f"Conflict at position {pos + 1}: query '{qc}' (conf {qconf:.2f}) vs registry '{cc}'."
            )
        if len(sim.mismatch_positions) > 4:
            sim.reasons.append(f"{len(sim.mismatch_positions) - 4} further plate conflicts omitted.")
    return sim


def _make_model_score(attrs: VehicleAttributes, entry: RegistryEntry, cfg: MatchConfig) -> tuple[float | None, str | None]:
    if attrs.make_model_confidence < cfg.high_conf_threshold:
        return None, None
    q_make = (attrs.make or "unknown").strip().lower()
    q_model = (attrs.model or "unknown").strip().lower()
    c_make = (entry.make or "unknown").strip().lower()
    c_model = (entry.model or "unknown").strip().lower()
    if q_make in ("", "unknown"):
        return None, None
    if q_make == c_make and (q_model == c_model or q_model == "unknown"):
        return 1.0, None
    if q_make == c_make:
        return 0.65, f"Make agrees ({entry.make}) but model differs ({attrs.model} vs {entry.model})."
    if q_model not in ("", "unknown") and q_model == c_model:
        return 0.40, f"Model agrees ({entry.model}) but make differs ({attrs.make} vs {entry.make})."
    return 0.0, f"Make/model conflict: query {attrs.make} {attrs.model} vs registry {entry.make} {entry.model}."


def evaluate_candidate(
    entry: RegistryEntry,
    fused: FusedPlate,
    attrs: VehicleAttributes,
    cfg: MatchConfig,
) -> MatchResult:
    plate = compare_plate(fused.text, entry.plate, fused.char_confidences, cfg)
    reasons = list(plate.reasons)
    conflicts: list[str] = []

    color_score: float | None = None
    if attrs.color_confidence >= 0.0 and attrs.color != "unknown":
        color_score = color_similarity(attrs.color, entry.color)
        if color_score is None:
            reasons.append("Colour unknown; ignored in scoring.")
        elif color_score >= 0.999:
            reasons.append(f"Colour matches: {attrs.color} (query confidence {attrs.color_confidence:.2f}).")
        elif color_score >= 0.5:
            reasons.append(f"Colour is compatible: query {attrs.color} vs registry {entry.color}.")
        else:
            conflicts.append(f"Colour conflict: query {attrs.color} vs registry {entry.color}.")
            reasons.append(conflicts[-1])

    body_score: float | None = None
    if attrs.body_type_confidence > 0 and attrs.body_type != "unknown":
        body_score = body_similarity(attrs.body_type, entry.body_type)
        if body_score is None:
            reasons.append("Body type unknown; ignored in scoring.")
        elif body_score >= 0.999:
            reasons.append(f"Body type matches: {attrs.body_type} (query confidence {attrs.body_type_confidence:.2f}).")
        elif body_score >= 0.5:
            reasons.append(f"Body type is compatible: query {attrs.body_type} vs registry {entry.body_type}.")
        else:
            conflicts.append(f"Body type conflict: query {attrs.body_type} vs registry {entry.body_type}.")
            reasons.append(conflicts[-1])

    style_score: float | None = None
    if attrs.plate_style_confidence >= cfg.min_plate_style_conf and attrs.plate_style != "unknown":
        style_score = plate_style_similarity(attrs.plate_style, entry.plate_style)
        if style_score is None:
            reasons.append("Plate style unknown; ignored in scoring.")
        elif style_score >= 0.999:
            reasons.append(f"Plate style matches: {attrs.plate_style} (commercial/private/EV cue).")
        elif style_score >= 0.3:
            reasons.append(f"Plate style is weakly compatible: query {attrs.plate_style} vs registry {entry.plate_style}.")
        else:
            conflicts.append(f"Plate style conflict: query {attrs.plate_style} vs registry {entry.plate_style}.")
            reasons.append(conflicts[-1])

    mm_score, mm_note = _make_model_score(attrs, entry, cfg)
    if mm_score is not None:
        if mm_score >= 0.999:
            reasons.append(f"Make/model matches: {entry.make} {entry.model}.")
        elif mm_score > 0:
            reasons.append(mm_note or f"Make/model partially agrees: {entry.make} {entry.model}.")
        else:
            conflicts.append(mm_note or "Make/model conflict.")
            reasons.append(conflicts[-1])

    # State / RTO evidence is useful and cheap to explain.
    q = clean_text(fused.text)
    if len(q) >= 4 and len(entry.plate) >= 4 and q[:2] != UNKNOWN * 2:
        if q[:2] == entry.plate[:2]:
            reasons.append(f"State code {q[:2]} is consistent with the registry entry.")
        elif all(ch != UNKNOWN for ch in q[:2]):
            conflicts.append(f"State code conflict: query {q[:2]} vs registry {entry.plate[:2]}.")
    if len(q) >= 4 and all(ch != UNKNOWN for ch in q[2:4]) and q[2:4] == entry.plate[2:4]:
        reasons.append(f"RTO code {q[2:4]} is consistent with the registry entry.")

    # Evidence reliability (0-1): how much of the plate/attributes is known.
    plate_rel = min(1.0, plate.known_chars / 8.0) * max(0.0, min(1.0, fused.mean_confidence))
    color_rel = attrs.color_confidence if color_score is not None else 0.0
    body_rel = attrs.body_type_confidence if body_score is not None else 0.0
    style_rel = attrs.plate_style_confidence if style_score is not None else 0.0
    mm_rel = attrs.make_model_confidence if mm_score is not None else 0.0

    w_plate = cfg.weight_plate * (0.30 + 0.70 * plate_rel)
    w_color = cfg.weight_color * color_rel
    w_body = cfg.weight_body * body_rel
    w_style = cfg.weight_plate_style * style_rel
    w_mm = cfg.weight_make_model * mm_rel
    total_w = w_plate + w_color + w_body + w_style + w_mm
    if total_w <= 0:
        w_plate, total_w = 1.0, 1.0
    score = (
        w_plate * plate.score
        + w_color * (color_score or 0.0)
        + w_body * (body_score or 0.0)
        + w_style * (style_score or 0.0)
        + w_mm * (mm_score or 0.0)
    ) / total_w

    rejected = False
    rejection_reason: str | None = None
    if plate.high_conf_mismatches >= cfg.max_high_conf_conflicts:
        rejected = True
        rejection_reason = (
            f"{plate.high_conf_mismatches} high-confidence plate conflicts "
            f"(limit {cfg.max_high_conf_conflicts})."
        )
    elif plate.known_chars >= 4 and plate.score < 0.25:
        rejected = True
        rejection_reason = f"Plate compatibility {plate.score:.2f} is too low."
    elif plate.high_conf_mismatches == 1:
        score *= 0.62
        conflicts.append(
            f"One high-confidence plate conflict; score reduced (position "
            f"{plate.mismatch_positions[0][0] + 1})."
        )

    evidence = 0.45 + 0.55 * (
        0.55 * plate_rel + 0.20 * color_rel + 0.15 * body_rel + 0.10 * style_rel
    )
    coverage = 0.35 + 0.65 * min(1.0, plate.known_chars / 8.0)
    confidence = score * evidence * coverage

    if plate.unknown_chars == 0 and not plate.exact:
        confidence = min(confidence, 0.44)
        conflicts.append("The query plate is fully read but does not exactly match this registry plate.")
    if plate.high_conf_mismatches == 1:
        confidence *= 0.55
    if plate.known_chars < cfg.min_known_chars:
        confidence = min(confidence, 0.35)
        reasons.append(f"Only {plate.known_chars} plate character(s) recovered; confidence capped.")
    if plate.score < 0.5 and (color_score is None or color_score < 0.5) and (body_score is None or body_score < 0.5):
        confidence = min(confidence, 0.35)
    confidence = max(0.0, min(0.99, confidence))

    if not rejected and confidence < cfg.min_confidence:
        rejection_reason = f"Confidence {confidence:.2f} is below the shortlist threshold."

    return MatchResult(
        rank=0,
        vehicle=entry,
        score=round(float(score), 4),
        confidence=round(float(confidence), 4),
        plate_score=round(float(plate.score), 4),
        color_score=None if color_score is None else round(float(color_score), 4),
        body_score=None if body_score is None else round(float(body_score), 4),
        plate_style_score=None if style_score is None else round(float(style_score), 4),
        make_model_score=None if mm_score is None else round(float(mm_score), 4),
        known_chars=plate.known_chars,
        unknown_chars=plate.unknown_chars,
        exact_plate=plate.exact,
        reasons=reasons,
        conflicts=conflicts,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def match_registry(
    fused: FusedPlate,
    attrs: VehicleAttributes,
    registry: Registry,
    cfg: MatchConfig,
) -> MatchOutcome:
    evaluated = [evaluate_candidate(entry, fused, attrs, cfg) for entry in registry.entries]
    evaluated.sort(key=lambda m: (m.rejected, -m.confidence, -m.score, -m.known_chars, m.vehicle.plate))
    eligible = [m for m in evaluated if not m.rejected]
    rejected = [m for m in evaluated if m.rejected]

    top = eligible[0] if eligible else None
    second = eligible[1] if len(eligible) > 1 else None
    top_conf = top.confidence if top else 0.0
    runner_conf = second.confidence if second else 0.0
    margin = top_conf - runner_conf

    thresholds = {
        "min_confidence": cfg.min_confidence,
        "min_score": cfg.min_score,
        "min_known_chars": float(cfg.min_known_chars),
        "min_margin": cfg.min_margin,
        "max_high_conf_conflicts": float(cfg.max_high_conf_conflicts),
        "min_plate_style_conf": float(cfg.min_plate_style_conf),
    }

    decision = "SHORTLIST"
    reason = ""
    if top is None:
        decision = "NO_CONFIDENT_MATCH"
        reason = "Every registry entry was rejected by the plate-conflict filter."
    elif top.confidence < cfg.min_confidence:
        decision = "NO_CONFIDENT_MATCH"
        reason = (
            f"Best candidate {top.vehicle.plate} has confidence {top.confidence:.2f}, "
            f"below the {cfg.min_confidence:.2f} threshold."
        )
    elif top.score < cfg.min_score:
        decision = "NO_CONFIDENT_MATCH"
        reason = (
            f"Best candidate {top.vehicle.plate} has a match score of {top.score:.2f}, "
            f"below the {cfg.min_score:.2f} threshold."
        )
    elif top.known_chars < cfg.min_known_chars:
        decision = "NO_CONFIDENT_MATCH"
        reason = (
            f"Only {top.known_chars} plate character(s) recovered; at least "
            f"{cfg.min_known_chars} are required for a confident match."
        )
    elif top.plate_score < 0.5 and (top.color_score is None or top.color_score < 0.5) and (
        top.body_score is None or top.body_score < 0.5
    ):
        decision = "NO_CONFIDENT_MATCH"
        reason = "Plate and attribute evidence are both too weak to identify a vehicle."
    elif second is not None and margin < cfg.min_margin and top.confidence < 0.70:
        decision = "NO_CONFIDENT_MATCH"
        reason = (
            f"The top two candidates are within {margin:.2f} confidence "
            f"({top.vehicle.plate} {top.confidence:.2f} vs {second.vehicle.plate} "
            f"{second.confidence:.2f}); the evidence is ambiguous."
        )
    else:
        reason = (
            f"Top candidate {top.vehicle.plate} scored {top.confidence:.2f} "
            f"with a {margin:.2f} margin over the runner-up."
        )
        if top.conflicts:
            reason += " It is the best fit despite the listed conflicts."

    if decision == "SHORTLIST":
        matches = eligible[: cfg.top_k]
        for i, m in enumerate(matches, start=1):
            m.rank = i
        near = rejected[:3]
    else:
        matches = []
        # Keep the shortlist transparent but clearly labelled as near misses.
        near = eligible[: cfg.top_k]
        for i, m in enumerate(near, start=1):
            m.rank = i
        for m in rejected[:2]:
            if m not in near:
                near.append(m)
        near = near[: cfg.top_k]

    return MatchOutcome(
        decision=decision,
        decision_reason=reason,
        matches=matches,
        near_misses=near,
        top_confidence=round(float(top_conf), 4),
        runner_up_confidence=round(float(runner_conf), 4),
        margin=round(float(margin), 4),
        thresholds=thresholds,
    )
