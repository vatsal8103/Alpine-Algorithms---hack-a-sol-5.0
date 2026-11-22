"""Vehicle and number-plate detection.

Primary path: Ultralytics YOLO (COCO vehicle classes + optional custom plate
weights).  Fallback path: a classical contour/morphology plate localiser so
the demo still runs without model downloads.  The fallback is explicitly
marked in the report; it is not presented as a trained detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .io_utils import clip_box, crop
from .video import frame_quality

# COCO class id -> TRACE-X body type.  SUV / auto-rickshaw are not COCO
# classes; they are handled by the attribute stage (custom classifier or hint).
COCO_VEHICLE_MAP = {
    1: "two-wheeler",   # bicycle
    2: "car",
    3: "two-wheeler",   # motorcycle
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    class_id: int = -1
    source: str = "yolo"
    track_id: int | None = None


@dataclass
class PlateCandidate:
    box: tuple[float, float, float, float]
    score: float
    source: str = "heuristic"


class VehicleDetector:
    """Lazy Ultralytics wrapper with an explicit availability status."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._model = None
        self._tried = False
        self.status = "not loaded"

    def _load(self) -> None:
        if self._tried:
            return
        self._tried = True
        if not self.cfg.enabled:
            self.status = "disabled by config"
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.status = f"unavailable (ultralytics import failed: {exc})"
            return
        model_name = self.cfg.vehicle_model
        model_path = Path(model_name)
        if not model_path.exists() and not self.cfg.allow_download and model_path.suffix == ".pt":
            self.status = f"unavailable (weights not found: {model_name})"
            return
        try:
            self._model = YOLO(model_name)
            self.status = f"ready ({model_name})"
        except Exception as exc:  # pragma: no cover - network/GPU dependent
            self.status = f"unavailable (model load failed: {exc})"

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._load()
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                source=image,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                imgsz=int(self.cfg.imgsz),
                device=self.cfg.device,
                verbose=False,
            )
        except Exception:
            return []
        return self._parse_results(results, image)

    def track(self, image: np.ndarray) -> list[Detection]:
        """Detect + ByteTrack.  Falls back to plain detection if tracking fails.

        ByteTrack is pretrained (Ultralytics ``bytetrack.yaml``); we never
        train or fine-tune the tracker.
        """

        self._load()
        if self._model is None or not self.cfg.use_tracking:
            return self.detect(image)
        try:
            results = self._model.track(
                source=image,
                persist=True,
                conf=float(self.cfg.conf),
                iou=float(self.cfg.iou),
                imgsz=int(self.cfg.imgsz),
                device=self.cfg.device,
                tracker=self.cfg.tracker,
                verbose=False,
            )
        except Exception:
            return self.detect(image)
        return self._parse_results(results, image)

    def _parse_results(self, results, image: np.ndarray) -> list[Detection]:
        if not results:
            return []
        h, w = image.shape[:2]
        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return []
        ids = None
        try:
            if getattr(boxes, "id", None) is not None:
                ids = boxes.id.cpu().numpy()
        except Exception:
            ids = None
        for idx, (xyxy, cls, conf) in enumerate(
            zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())
        ):
            class_id = int(cls)
            if class_id not in COCO_VEHICLE_MAP:
                continue
            x1, y1, x2, y2 = clip_box(tuple(float(v) for v in xyxy[:4]), w, h)
            track_id = int(ids[idx]) if ids is not None and idx < len(ids) else None
            detections.append(
                Detection(
                    label=COCO_VEHICLE_MAP[class_id],
                    confidence=float(conf),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    class_id=class_id,
                    source="yolo+bytetrack" if track_id is not None else "yolo",
                    track_id=track_id,
                )
            )
        detections.sort(key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) * d.confidence, reverse=True)
        return detections[:10]


class PlateDetector:
    """YOLO plate model when supplied, otherwise a classical localiser."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._model = None
        self._tried = False
        self.status = "heuristic localiser"
        if cfg.plate_model:
            self._load_plate_model()

    def _load_plate_model(self) -> None:
        if self._tried:
            return
        self._tried = True
        path = Path(self.cfg.plate_model or "")
        if not path.exists():
            self.status = f"plate weights not found ({path}); using heuristic"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(path))
            self.status = f"ready ({path.name})"
        except Exception as exc:  # pragma: no cover
            self.status = f"plate model failed ({exc}); using heuristic"

    def detect(self, image: np.ndarray, vehicle_box: tuple[float, float, float, float] | None = None) -> list[PlateCandidate]:
        if self._model is not None:
            return self._detect_yolo(image, vehicle_box)
        return self._detect_heuristic(image, vehicle_box)

    def _detect_yolo(
        self, image: np.ndarray, vehicle_box: tuple[float, float, float, float] | None
    ) -> list[PlateCandidate]:
        roi = image
        ox, oy = 0.0, 0.0
        if vehicle_box is not None:
            roi = crop(image, vehicle_box, pad=0.03)
            ox, oy = vehicle_box[0], vehicle_box[1]
        try:
            results = self._model.predict(source=roi, conf=0.20, imgsz=640, verbose=False)  # type: ignore[union-attr]
        except Exception:
            return self._detect_heuristic(image, vehicle_box)
        if not results or results[0].boxes is None:
            return []
        candidates: list[PlateCandidate] = []
        for xyxy, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
            candidates.append(
                PlateCandidate(
                    box=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    score=float(conf),
                    source="yolo-plate",
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:4]

    def _detect_heuristic(
        self, image: np.ndarray, vehicle_box: tuple[float, float, float, float] | None
    ) -> list[PlateCandidate]:
        h, w = image.shape[:2]
        if vehicle_box is None:
            vehicle_box = (0.02 * w, 0.02 * h, 0.98 * w, 0.98 * h)
        x1, y1, x2, y2 = clip_box(vehicle_box, w, h)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        # Work at a bounded resolution for speed; map boxes back afterwards.
        max_side = 960
        scale = min(1.0, max_side / max(roi.shape[:2]))
        work = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else roi
        wh, ww = work.shape[:2]
        raw_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw_gray)

        candidates: list[PlateCandidate] = []
        candidates.extend(_rectangle_candidates(raw_gray, ww, wh, scale, x1, y1))

        masks: list[tuple[np.ndarray, str]] = []
        for kernel_w in (9, 15):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
            for m, name in ((blackhat, "dark-text"), (tophat, "light-text")):
                _, th = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                masks.append((th, name))
                masks.append((cv2.bitwise_not(th), name + "-inv"))

        for mask, name in masks:
            # Connect characters into a plate-shaped blob, then find contours.
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
            merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
            merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bx, by, bw, bh = cv2.boundingRect(contour)
                if bw < 42 or bh < 12:
                    continue
                aspect = bw / max(1.0, float(bh))
                if aspect < 1.6 or aspect > 7.5:
                    continue
                area_ratio = (bw * bh) / float(ww * wh)
                if area_ratio < 0.0004 or area_ratio > 0.35:
                    continue
                rect = cv2.contourArea(contour) / max(1.0, float(bw * bh))
                if rect < 0.22:
                    continue
                patch = gray[by:by + bh, bx:bx + bw]
                if patch.size == 0:
                    continue
                gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
                edge_density = float((np.abs(gx) > 55).mean())
                text_score = min(1.0, edge_density / 0.09)
                if text_score < 0.15:
                    continue
                aspect_score = math.exp(-0.5 * ((math.log(aspect / 3.2)) / 0.55) ** 2)
                area_score = min(1.0, area_ratio / 0.018)
                cx = (bx + bw / 2) / max(1.0, ww)
                cy = (by + bh / 2) / max(1.0, wh)
                position_score = (1.0 - abs(cx - 0.5)) * (0.55 + 0.45 * cy)
                score = (
                    0.34 * aspect_score
                    + 0.18 * area_score
                    + 0.18 * min(1.0, rect)
                    + 0.20 * text_score
                    + 0.10 * position_score
                )
                # Map back to original image coordinates.
                ox = x1 + bx / scale
                oy = y1 + by / scale
                ow = bw / scale
                oh = bh / scale
                candidates.append(
                    PlateCandidate(
                        box=(ox, oy, ox + ow, oy + oh),
                        score=float(max(0.0, min(1.0, score))),
                        source=f"heuristic-{name}",
                    )
                )

        candidates = _dedupe_candidates(candidates, iou_threshold=0.45)
        if not candidates:
            # Last resort: lower-central window.  OCR still has to pass the
            # plate-grammar likelihood gate, so this cannot fabricate a plate.
            fw, fh = (x2 - x1), (y2 - y1)
            candidates.append(
                PlateCandidate(
                    box=(x1 + 0.15 * fw, y1 + 0.55 * fh, x1 + 0.85 * fw, y1 + 0.92 * fh),
                    score=0.05,
                    source="fallback-window",
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:4]

    def crop_and_refine(
        self, image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.14
    ) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """Return a padded plate crop, its refined box and a quality score."""

        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        pw = (x2 - x1) * pad
        ph = (y2 - y1) * pad
        bx1, by1, bx2, by2 = clip_box((x1 - pw, y1 - ph, x2 + pw, y2 + ph), w, h)
        plate = image[by1:by2, bx1:bx2].copy()
        if plate.size == 0:
            return plate, (bx1, by1, bx2, by2), 0.0
        refined = _tighten_plate(plate)
        if refined is not None:
            rx1, ry1, rx2, ry2 = refined
            if (rx2 - rx1) >= 0.45 * (bx2 - bx1) and (ry2 - ry1) >= 0.35 * (by2 - by1):
                plate = plate[ry1:ry2, rx1:rx2].copy()
                bx1, by1, bx2, by2 = bx1 + rx1, by1 + ry1, bx1 + rx2, by1 + ry2
        q = frame_quality(image, (bx1, by1, bx2, by2))["quality"]
        return plate, (bx1, by1, bx2, by2), float(q)


def _rectangle_candidates(
    gray: np.ndarray,
    ww: int,
    wh: int,
    scale: float,
    ox: float,
    oy: float,
) -> list[PlateCandidate]:
    """Find plate-shaped rectangles from strong edges.

    Number plates have a conspicuous rectangular border.  This works for
    white, yellow and green plates and complements the black-hat text blob
    detector (which struggles when the plate is brighter than the vehicle).
    """

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[PlateCandidate] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 0.05 * ww or bw > 0.85 * ww or bh < 0.015 * wh or bh > 0.30 * wh:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect < 1.7 or aspect > 7.2:
            continue
        rectangularity = cv2.contourArea(contour) / max(1.0, float(bw * bh))
        if rectangularity < 0.40:
            continue
        patch = gray[by:by + bh, bx:bx + bw]
        if patch.size == 0:
            continue
        lo, hi = np.percentile(patch, [10, 90])
        contrast = float(hi - lo)
        edge_density = float((cv2.Canny(patch, 50, 150) > 0).mean())
        text_score = min(1.0, 0.55 * contrast / 90.0 + 0.45 * edge_density / 0.10)
        if text_score < 0.12:
            continue
        aspect_score = math.exp(-0.5 * ((math.log(aspect / 3.6)) / 0.48) ** 2)
        area_ratio = (bw * bh) / float(ww * wh)
        area_score = min(1.0, area_ratio / 0.020)
        cx = (bx + bw / 2.0) / max(1.0, ww)
        cy = (by + bh / 2.0) / max(1.0, wh)
        position_score = (1.0 - abs(cx - 0.5)) * (0.45 + 0.55 * cy)
        score = (
            0.30 * rectangularity
            + 0.26 * aspect_score
            + 0.20 * text_score
            + 0.14 * position_score
            + 0.10 * area_score
        )
        x = ox + bx / scale
        y = oy + by / scale
        width = bw / scale
        height = bh / scale
        candidates.append(
            PlateCandidate(
                box=(x, y, x + width, y + height),
                score=float(max(0.0, min(1.0, score))),
                source="edge-rectangle",
            )
        )
    return candidates


def _tighten_plate(plate: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tighten a crop to the text blob, ignoring plate borders/screws."""

    if plate.size == 0:
        return None
    h, w = plate.shape[:2]
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    boxes: list[tuple[int, int, int, int]] = []
    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, th = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        local: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 0.25 * h or bh > 0.98 * h:
                continue
            if bw < 0.03 * w or bw > 0.35 * w:
                continue
            local.append((x, y, bw, bh))
        if len(local) >= 4:
            boxes = local
            break
    if len(boxes) < 4:
        return None
    # Keep a margin around the detected text.  Cropping flush to the glyphs
    # clips the first/last character when the contour is slightly eroded.
    mx = max(3, int(0.06 * w))
    my = max(3, int(0.12 * h))
    x1 = max(0, min(b[0] for b in boxes) - mx)
    y1 = max(0, min(b[1] for b in boxes) - my)
    x2 = min(w, max(b[0] + b[2] for b in boxes) + mx)
    y2 = min(h, max(b[1] + b[3] for b in boxes) + my)
    if x2 - x1 < 0.5 * w or y2 - y1 < 0.3 * h:
        return None
    return x1, y1, x2, y2


def _dedupe_candidates(candidates: list[PlateCandidate], iou_threshold: float = 0.45) -> list[PlateCandidate]:
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[PlateCandidate] = []
    for cand in candidates:
        if all(_iou(cand.box, k.box) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


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


def fallback_vehicle_box(image: np.ndarray) -> tuple[float, float, float, float]:
    h, w = image.shape[:2]
    return (0.05 * w, 0.08 * h, 0.95 * w, 0.96 * h)
