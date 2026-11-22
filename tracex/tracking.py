"""Cross-clip vehicle association (bonus) and a small IoU tracker fallback.

The association is deliberately conservative: two clips are called a likely
same-vehicle pair only when the recovered plate evidence agrees and at least
one attribute is consistent.  It is a hypothesis for investigators, not an
identity claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .attributes import body_similarity, color_similarity
from .config import TrackingConfig
from .detection import Detection
from .plate import CONFUSION, UNKNOWN, clean_text, confusion_match
from .schemas import AnalysisResult, TrackingResult


@dataclass
class VehicleSignature:
    job_id: str
    plate: str
    plate_confidences: list[float]
    color: str
    color_confidence: float
    body_type: str
    body_confidence: float
    make: str
    model: str
    make_model_confidence: float
    histogram: list[float] = field(default_factory=list)


def build_signature(result: AnalysisResult) -> VehicleSignature:
    return VehicleSignature(
        job_id=result.job_id,
        plate=result.fused_plate.text,
        plate_confidences=list(result.fused_plate.char_confidences),
        color=result.attributes.color,
        color_confidence=result.attributes.color_confidence,
        body_type=result.attributes.body_type,
        body_confidence=result.attributes.body_type_confidence,
        make=result.attributes.make,
        model=result.attributes.model,
        make_model_confidence=result.attributes.make_model_confidence,
        histogram=list(result.attributes.color_histogram),
    )


def _plate_between(a: VehicleSignature, b: VehicleSignature, threshold: float) -> tuple[float, int, int, list[str]]:
    qa = clean_text(a.plate)
    qb = clean_text(b.plate)
    length = min(len(qa), len(qb))
    if length == 0:
        return 0.0, 0, 0, ["No plate characters in either clip."]
    weighted = 0.0
    total = 0.0
    known_both = 0
    high_conflicts = 0
    reasons: list[str] = []
    for i in range(length):
        ca, cb = qa[i], qb[i]
        if ca == UNKNOWN or cb == UNKNOWN:
            continue
        conf = min(
            a.plate_confidences[i] if i < len(a.plate_confidences) else 0.5,
            b.plate_confidences[i] if i < len(b.plate_confidences) else 0.5,
        )
        weight = max(0.05, conf)
        if ca == cb:
            sim = 1.0
        elif confusion_match(ca, cb):
            sim = 0.65
        else:
            sim = 0.0
            if conf >= threshold:
                high_conflicts += 1
            reasons.append(f"Plate conflict at position {i + 1}: '{ca}' vs '{cb}' (confidence {conf:.2f}).")
        weighted += weight * sim
        total += weight
        known_both += 1
    score = weighted / total if total > 0 else 0.0
    if known_both:
        reasons.insert(0, f"{known_both} plate position(s) known in both clips; agreement score {score:.2f}.")
    if high_conflicts:
        reasons.append(f"{high_conflicts} high-confidence plate conflict(s).")
    return score, known_both, high_conflicts, reasons


def _hist_similarity(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-9:
        return None
    return float(max(0.0, min(1.0, float(np.dot(va, vb) / denom))))


def compare_signatures(a: VehicleSignature, b: VehicleSignature, cfg: TrackingConfig) -> TrackingResult:
    plate_score, known_both, high_conflicts, reasons = _plate_between(a, b, cfg.same_vehicle_threshold)
    conflicts: list[str] = []

    color_score = color_similarity(a.color, b.color)
    body_score = body_similarity(a.body_type, b.body_type)
    hist_score = _hist_similarity(a.histogram, b.histogram)

    color_rel = min(a.color_confidence, b.color_confidence) if color_score is not None else 0.0
    body_rel = min(a.body_confidence, b.body_confidence) if body_score is not None else 0.0
    hist_rel = 1.0 if hist_score is not None else 0.0
    plate_rel = min(1.0, known_both / max(1, cfg.min_plate_known))

    w_plate = cfg.weight_plate * (0.25 + 0.75 * plate_rel)
    w_color = cfg.weight_color * color_rel
    w_body = cfg.weight_body * body_rel
    w_hist = cfg.weight_appearance * hist_rel
    total = w_plate + w_color + w_body + w_hist
    if total <= 0:
        w_plate, total = 1.0, 1.0
    score = (
        w_plate * plate_score
        + w_color * (color_score or 0.0)
        + w_body * (body_score or 0.0)
        + w_hist * (hist_score or 0.0)
    ) / total

    evidence = 0.35 + 0.65 * min(1.0, (0.65 * plate_rel + 0.20 * color_rel + 0.15 * body_rel))
    confidence = max(0.0, min(0.99, score * evidence))
    if known_both < cfg.min_plate_known:
        confidence = min(confidence, 0.45)
        reasons.append(f"Only {known_both} plate position(s) are known in both clips; confidence capped.")
    if high_conflicts >= 2:
        confidence = min(confidence, 0.25)
        conflicts.append("Two or more high-confidence plate conflicts; the clips are treated as different/inconclusive.")
    elif high_conflicts == 1:
        confidence *= 0.70
        conflicts.append("One high-confidence plate conflict reduces the association confidence.")

    if color_score is not None:
        if color_score >= 0.8:
            reasons.append(f"Colour agrees: {a.color} vs {b.color}.")
        elif color_score < 0.4:
            conflicts.append(f"Colour differs: {a.color} vs {b.color}.")
    if body_score is not None:
        if body_score >= 0.8:
            reasons.append(f"Body type agrees: {a.body_type} vs {b.body_type}.")
        elif body_score < 0.4:
            conflicts.append(f"Body type differs: {a.body_type} vs {b.body_type}.")
    if hist_score is not None:
        reasons.append(f"Appearance histogram cosine similarity: {hist_score:.2f}.")

    same = confidence >= cfg.same_vehicle_threshold and high_conflicts < 2
    decision = "LIKELY_SAME_VEHICLE" if same else "DIFFERENT_OR_INCONCLUSIVE"
    if same:
        reasons.insert(0, "Highest-ranked hypothesis based on plate, colour, body and appearance evidence.")
    else:
        reasons.insert(0, "Not enough consistent evidence to associate the two clips.")

    return TrackingResult(
        same_vehicle=bool(same),
        confidence=round(float(confidence), 4),
        decision=decision,
        reasons=reasons,
        conflicts=conflicts,
        plate_similarity=round(float(plate_score), 4),
        color_similarity=None if color_score is None else round(float(color_score), 4),
        body_similarity=None if body_score is None else round(float(body_score), 4),
        histogram_similarity=None if hist_score is None else round(float(hist_score), 4),
        known_both=known_both,
        job_id_a=a.job_id,
        job_id_b=b.job_id,
    )


def compare_results(a: AnalysisResult, b: AnalysisResult, cfg: TrackingConfig) -> TrackingResult:
    return compare_signatures(build_signature(a), build_signature(b), cfg)


class SimpleIoUTracker:
    """Fallback tracker when ByteTrack / Ultralytics is unavailable.

    This is intentionally tiny: greedy IoU association with a max-age.  It is
    only used to keep a vehicle identity stable across frames for evidence
    grouping; it is not trained.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 8):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 1
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[Detection]) -> list[Detection]:
        assigned: set[int] = set()
        for det in detections:
            best_id = None
            best_iou = self.iou_threshold
            for track_id, track in self._tracks.items():
                if track_id in assigned:
                    continue
                score = self._iou(det.box, track["box"])
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            assigned.add(best_id)
            det.track_id = best_id
            self._tracks[best_id] = {"box": det.box, "age": 0}
        for track_id in list(self._tracks):
            if track_id not in assigned:
                self._tracks[track_id]["age"] += 1
                if self._tracks[track_id]["age"] > self.max_age:
                    del self._tracks[track_id]
        return detections
