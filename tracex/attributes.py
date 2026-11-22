"""Vehicle colour, body type and optional make/model extraction.

Colour is extracted from the lower-central body panel (number plate and
windscreen excluded), clustered with k-means and mapped to a small,
registry-friendly vocabulary.  Body type comes from the COCO detector when
available, or from an optional custom classifier / investigator hint.  When
neither exists the value is ``unknown`` rather than guessed.

The optional make/model path is a plug-in: provide a fine-tuned weights file
and label map, or a reference gallery of vehicle images.  Without one, the
field stays ``unknown`` and matching ignores it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import AttributeConfig
from .io_utils import crop

COLOR_FAMILIES = {
    "white": {"white", "pearl", "ivory"},
    "silver": {"silver", "grey", "gray"},
    "black": {"black"},
    "red": {"red", "maroon"},
    "blue": {"blue", "navy"},
    "green": {"green"},
    "yellow": {"yellow", "gold"},
    "orange": {"orange"},
    "brown": {"brown"},
}

_ADJACENT = {
    ("white", "silver"): 0.72,
    ("silver", "white"): 0.72,
    ("silver", "grey"): 0.85,
    ("grey", "silver"): 0.85,
    ("black", "grey"): 0.55,
    ("grey", "black"): 0.55,
    ("red", "maroon"): 0.90,
    ("maroon", "red"): 0.90,
    ("red", "orange"): 0.42,
    ("orange", "red"): 0.42,
    ("yellow", "orange"): 0.60,
    ("orange", "yellow"): 0.60,
    ("blue", "green"): 0.30,
    ("green", "blue"): 0.30,
    ("brown", "orange"): 0.45,
    ("orange", "brown"): 0.45,
}

BODY_TYPES = [
    "hatchback", "sedan", "car", "suv", "two-wheeler",
    "auto-rickshaw", "truck", "bus", "van", "unknown",
]

_BODY_COMPAT = {
    ("car", "suv"): 0.55,
    ("suv", "car"): 0.55,
    ("car", "hatchback"): 0.80,
    ("hatchback", "car"): 0.80,
    ("car", "sedan"): 0.80,
    ("sedan", "car"): 0.80,
    ("hatchback", "sedan"): 0.50,
    ("sedan", "hatchback"): 0.50,
    ("hatchback", "suv"): 0.40,
    ("suv", "hatchback"): 0.40,
    ("sedan", "suv"): 0.40,
    ("suv", "sedan"): 0.40,
    ("truck", "bus"): 0.45,
    ("bus", "truck"): 0.45,
    ("van", "car"): 0.60,
    ("car", "van"): 0.60,
    ("van", "suv"): 0.60,
    ("suv", "van"): 0.60,
}

PLATE_STYLES = ["white", "yellow", "green", "unknown"]

_PLATE_STYLE_COMPAT = {
    ("white", "yellow"): 0.35,
    ("yellow", "white"): 0.35,
    ("green", "white"): 0.20,
    ("white", "green"): 0.20,
    ("yellow", "green"): 0.10,
    ("green", "yellow"): 0.10,
}


def plate_style_similarity(query: str, candidate: str) -> float | None:
    q = (query or "unknown").strip().lower()
    c = (candidate or "unknown").strip().lower()
    if q not in PLATE_STYLES or c not in PLATE_STYLES:
        return None
    if q == "unknown" or c == "unknown":
        return None
    if q == c:
        return 1.0
    return _PLATE_STYLE_COMPAT.get((q, c), 0.0)


def canonical_color(name: str | None) -> str:
    value = (name or "unknown").strip().lower()
    value = {"gray": "grey", "grayish": "grey", "silverish": "silver"}.get(value, value)
    for canonical, family in COLOR_FAMILIES.items():
        if value in family:
            return canonical
    return value if value in COLOR_FAMILIES else "unknown"


def color_similarity(query: str, candidate: str) -> float | None:
    q = canonical_color(query)
    c = canonical_color(candidate)
    if q == "unknown" or c == "unknown":
        return None
    if q == c:
        return 1.0
    if canonical_color(q) == canonical_color(c):
        return 0.85
    return _ADJACENT.get((q, c), 0.0)


def body_similarity(query: str, candidate: str) -> float | None:
    q = (query or "unknown").strip().lower()
    c = (candidate or "unknown").strip().lower()
    if q in ("", "unknown") or c in ("", "unknown"):
        return None
    if q == c:
        return 1.0
    return _BODY_COMPAT.get((q, c), 0.0)


def _classify_hsv(h: int, s: int, v: int) -> str:
    """Map one OpenCV HSV triple to a registry colour name."""

    if v < 42:
        return "black"
    if s < 26:
        if v > 198:
            return "white"
        if v > 118:
            return "silver"
        return "grey"
    if s < 52 and v > 196:
        return "white"
    hue = h * 2.0  # OpenCV hue is 0..179
    if hue < 12 or hue >= 168:
        return "maroon" if v < 82 else "red"
    if hue < 24:
        return "brown" if v < 92 else "orange"
    if hue < 34:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 140:
        return "blue"
    if hue < 168:
        return "purple"
    return "unknown"


@dataclass
class ColorResult:
    color: str = "unknown"
    confidence: float = 0.0
    distribution: dict[str, float] = field(default_factory=dict)
    histogram: list[float] = field(default_factory=list)
    evidence: str = ""


def extract_color(
    image: np.ndarray,
    vehicle_box: tuple[float, float, float, float],
    plate_box: tuple[float, float, float, float] | None = None,
) -> ColorResult:
    """Dominant body colour from the lower-central vehicle panel."""

    if image is None or image.size == 0:
        return ColorResult()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = vehicle_box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Body panel: avoid windows (top), shadow (bottom) and frame edges.
    bx1 = int(max(0, x1 + 0.10 * bw))
    bx2 = int(min(w, x1 + 0.90 * bw))
    by1 = int(max(0, y1 + 0.34 * bh))
    by2 = int(min(h, y1 + 0.93 * bh))
    if bx2 <= bx1 or by2 <= by1:
        bx1, by1, bx2, by2 = 0, 0, w, h
    panel = image[by1:by2, bx1:bx2].copy()
    if panel.size == 0:
        return ColorResult()

    mask = np.ones(panel.shape[:2], dtype=np.uint8) * 255
    if plate_box is not None:
        px1, py1, px2, py2 = plate_box
        pad_x = 0.15 * max(1.0, px2 - px1)
        pad_y = 0.30 * max(1.0, py2 - py1)
        rx1 = int(max(0, px1 - pad_x - bx1))
        ry1 = int(max(0, py1 - pad_y - by1))
        rx2 = int(min(panel.shape[1], px2 + pad_x - bx1))
        ry2 = int(min(panel.shape[0], py2 + pad_y - by1))
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 0

    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if pixels.shape[0] < 50:
        return ColorResult()
    if pixels.shape[0] > 6000:
        idx = np.linspace(0, pixels.shape[0] - 1, 6000).astype(np.int64)
        pixels = pixels[idx]

    data = pixels.astype(np.float32)
    k = 3
    cv2.setRNGSeed(42)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    try:
        _compactness, labels, centers = cv2.kmeans(
            data, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        return ColorResult()
    labels = labels.reshape(-1)
    total = float(len(labels))
    counts = np.bincount(labels, minlength=k).astype(float)
    merged: dict[str, dict[str, float]] = {}
    for idx in range(k):
        frac = counts[idx] / total
        if frac < 0.06:
            continue
        hh, ss, vv = [int(round(v)) for v in centers[idx]]
        name = _classify_hsv(hh, ss, vv)
        if name == "unknown":
            continue
        bucket = merged.setdefault(name, {"fraction": 0.0, "h": 0.0, "s": 0.0, "v": 0.0})
        bucket["fraction"] += frac
        bucket["h"] += hh * frac
        bucket["s"] += ss * frac
        bucket["v"] += vv * frac

    if not merged:
        return ColorResult()
    ranked = sorted(merged.items(), key=lambda kv: kv[1]["fraction"], reverse=True)
    top_name, top = ranked[0]
    second_frac = ranked[1][1]["fraction"] if len(ranked) > 1 else 0.0
    top_frac = top["fraction"]
    # Purity and separation, with a small bonus for chromatic colours where
    # the HSV classification is more reliable.
    separation = max(0.0, top_frac - second_frac)
    chromatic = 1.0 if top_name not in ("white", "silver", "grey", "black") else 0.9
    confidence = min(0.97, 1.20 * top_frac * (0.55 + 0.45 * separation) * chromatic)
    if top_frac < 0.34:
        confidence *= 0.65
    distribution = {name: round(bucket["fraction"], 4) for name, bucket in ranked}

    # Compact HSV histogram for cross-camera appearance comparison.
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 4], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    hist = hist / max(1e-9, float(hist.sum()))
    evidence = (
        f"k-means body panel: {top_name} {top_frac:.0%} of pixels "
        f"(runner-up {second_frac:.0%})"
    )
    return ColorResult(
        color=top_name,
        confidence=float(max(0.0, confidence)),
        distribution=distribution,
        histogram=[float(v) for v in hist],
        evidence=evidence,
    )


def fuse_colors(results: list[tuple[ColorResult, float]]) -> tuple[ColorResult, list[str]]:
    """Weighted colour vote across frames."""

    evidence: list[str] = []
    if not results:
        return ColorResult(), evidence
    votes: dict[str, float] = {}
    conf_sum: dict[str, float] = {}
    hist_sum = np.zeros(256, dtype=np.float64)
    hist_weight = 0.0
    for result, frame_weight in results:
        if result.color == "unknown":
            continue
        w = max(0.01, result.confidence) * max(0.05, frame_weight)
        votes[result.color] = votes.get(result.color, 0.0) + w
        conf_sum[result.color] = conf_sum.get(result.color, 0.0) + w * result.confidence
        if result.histogram:
            hist = np.array(result.histogram, dtype=np.float64)
            if hist.shape[0] == hist_sum.shape[0]:
                hist_sum += hist * w
                hist_weight += w
    if not votes:
        return ColorResult(), evidence
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ranked[0]
    total_w = sum(votes.values())
    share = top_w / max(1e-9, total_w)
    mean_conf = conf_sum[top] / max(1e-9, top_w)
    confidence = mean_conf * (0.55 + 0.45 * share)
    if share < 0.6:
        confidence *= 0.8
    evidence.append(
        f"colour fused over {len(results)} frames: {top} "
        f"({share:.0%} of evidence weight)"
    )
    hist_list: list[float] = []
    if hist_weight > 0:
        hist = hist_sum / hist_weight
        hist = hist / max(1e-9, float(hist.sum()))
        hist_list = [float(v) for v in hist]
    return (
        ColorResult(
            color=top,
            confidence=float(max(0.0, min(0.97, confidence))),
            distribution={name: round(weight / total_w, 4) for name, weight in ranked},
            histogram=hist_list,
            evidence=evidence[0],
        ),
        evidence,
    )


@dataclass
class PlateStyleResult:
    style: str = "unknown"
    confidence: float = 0.0
    evidence: str = ""


def extract_plate_style(plate_crop: np.ndarray) -> PlateStyleResult:
    """Classify the plate background as white / yellow / green.

    Commercial vehicles use yellow plates, private vehicles white and EVs
    green.  This is weak evidence, so it gets a small matching weight.  The
    classifier looks at the bright background pixels, not the characters.
    """

    if plate_crop is None or plate_crop.size == 0:
        return PlateStyleResult()
    h, w = plate_crop.shape[:2]
    # Trim the outer border, which may contain a coloured frame or dirt.
    x1, y1 = int(0.06 * w), int(0.06 * h)
    x2, y2 = max(x1 + 1, int(0.94 * w)), max(y1 + 1, int(0.94 * h))
    inner = plate_crop[y1:y2, x1:x2]
    if inner.size == 0:
        return PlateStyleResult()
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)

    # Background = the brighter half of pixels.  This suppresses dark text.
    threshold = float(np.percentile(vv, 55))
    bg = vv >= threshold
    if int(bg.sum()) < 30:
        return PlateStyleResult()
    h_bg, s_bg, v_bg = hh[bg], ss[bg], vv[bg]
    total = float(bg.sum())

    yellow = ((h_bg >= 15) & (h_bg <= 40) & (s_bg >= 55) & (v_bg >= 70)).sum() / total
    green = ((h_bg >= 35) & (h_bg <= 95) & (s_bg >= 45) & (v_bg >= 45)).sum() / total
    white = ((s_bg <= 45) & (v_bg >= 135)).sum() / total
    scores = {"white": white, "yellow": yellow, "green": green}
    style, frac = max(scores.items(), key=lambda kv: kv[1])
    second = sorted(scores.values(), reverse=True)[1]
    if frac < 0.30:
        return PlateStyleResult("unknown", 0.0, "plate background colour is ambiguous")
    confidence = min(0.95, frac * (0.65 + 0.35 * (frac - second) * 2.0))
    return PlateStyleResult(
        style=style,
        confidence=float(max(0.0, confidence)),
        evidence=f"plate background: {style} ({frac:.0%} of bright pixels)",
    )


def fuse_plate_styles(results: list[tuple[PlateStyleResult, float]]) -> tuple[PlateStyleResult, list[str]]:
    evidence: list[str] = []
    votes: dict[str, float] = {}
    conf_sum: dict[str, float] = {}
    for result, frame_weight in results:
        if result.style == "unknown":
            continue
        w = max(0.01, result.confidence) * max(0.05, frame_weight)
        votes[result.style] = votes.get(result.style, 0.0) + w
        conf_sum[result.style] = conf_sum.get(result.style, 0.0) + w * result.confidence
    if not votes:
        return PlateStyleResult(), evidence
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ranked[0]
    total_w = sum(votes.values())
    share = top_w / max(1e-9, total_w)
    mean_conf = conf_sum[top] / max(1e-9, top_w)
    confidence = mean_conf * (0.55 + 0.45 * share)
    evidence.append(f"plate style fused over {len(results)} frames: {top} ({share:.0%} weight)")
    return PlateStyleResult(top, float(max(0.0, min(0.95, confidence))), evidence[0]), evidence


@dataclass
class BodyResult:
    body_type: str = "unknown"
    confidence: float = 0.0
    evidence: str = ""


def body_from_detection(label: str, confidence: float, box: tuple[float, float, float, float], image: np.ndarray) -> BodyResult:
    label = (label or "unknown").lower()
    if label in ("hatchback", "sedan", "car", "suv", "two-wheeler", "auto-rickshaw", "truck", "bus", "van"):
        # COCO has no SUV class.  A tall, large car is *weakly* re-labelled as
        # SUV only when the geometry is strongly SUV-like; otherwise keep car.
        if label == "car":
            h, w = image.shape[:2]
            bw = max(1.0, box[2] - box[0])
            bh = max(1.0, box[3] - box[1])
            aspect = bh / bw
            if aspect > 0.66 and bw > 0.22 * w:
                return BodyResult("suv", min(0.62, 0.42 + 0.25 * confidence), "tall car bounding box suggests SUV")
        return BodyResult(label, float(max(0.0, min(1.0, confidence))), f"detector class {label}")
    return BodyResult()


def fuse_body_types(results: list[tuple[BodyResult, float]]) -> tuple[BodyResult, list[str]]:
    evidence: list[str] = []
    votes: dict[str, float] = {}
    conf_sum: dict[str, float] = {}
    for result, frame_weight in results:
        if result.body_type == "unknown":
            continue
        w = max(0.01, result.confidence) * max(0.05, frame_weight)
        votes[result.body_type] = votes.get(result.body_type, 0.0) + w
        conf_sum[result.body_type] = conf_sum.get(result.body_type, 0.0) + w * result.confidence
    if not votes:
        return BodyResult(), evidence
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ranked[0]
    total_w = sum(votes.values())
    share = top_w / max(1e-9, total_w)
    mean_conf = conf_sum[top] / max(1e-9, top_w)
    confidence = mean_conf * (0.55 + 0.45 * share)
    evidence.append(f"body type fused over {len(results)} frames: {top} ({share:.0%} weight)")
    return BodyResult(top, float(max(0.0, min(0.95, confidence))), evidence[0]), evidence


class AppearanceEmbedder:
    """Dependency-light appearance descriptor used by the reference gallery."""

    def embed(self, image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
        patch = crop(image, box, pad=0.02)
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / max(1e-9, float(hist.sum()))
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor(
            _winSize=(128, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        feat = hog.compute(gray).flatten()
        feat = feat / max(1e-9, float(np.linalg.norm(feat)))
        vec = np.concatenate([hist, feat]).astype(np.float32)
        return vec / max(1e-9, float(np.linalg.norm(vec)))


class MakeModelRecognizer:
    """Optional make/model plug-in.

    Priority:
    1. fine-tuned TorchScript / state-dict classifier if ``weights`` and
       ``labels`` are supplied;
    2. reference gallery (``gallery.json`` + images) using an appearance
       embedding;
    3. ``unknown``.
    """

    def __init__(self, gallery_dir: str | Path | None = None, weights: str | Path | None = None, labels: str | Path | None = None):
        self.embedder = AppearanceEmbedder()
        self.gallery: list[dict] = []
        self._model = None
        self._labels: list[str] = []
        self._torch = None
        if gallery_dir:
            self._load_gallery(Path(gallery_dir))
        if weights and labels:
            self._load_classifier(Path(weights), Path(labels))

    def _load_gallery(self, root: Path) -> None:
        meta = root / "gallery.json"
        if not meta.exists():
            return
        try:
            entries = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return
        for entry in entries:
            path = root / entry.get("image", "")
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            box = entry.get("box") or [0, 0, image.shape[1], image.shape[0]]
            vec = self.embedder.embed(image, tuple(box))
            if vec is None:
                continue
            self.gallery.append(
                {
                    "vector": vec,
                    "make": entry.get("make", "unknown"),
                    "model": entry.get("model", "unknown"),
                    "vehicle_id": entry.get("vehicle_id", ""),
                    "image": str(path),
                }
            )

    def _load_classifier(self, weights: Path, labels: Path) -> None:
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore

            self._torch = torch
            self._labels = [line.strip() for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
            model = torchvision.models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(self._labels))
            state = torch.load(str(weights), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._model = model
        except Exception:
            self._model = None

    def predict(self, image: np.ndarray, box: tuple[float, float, float, float]) -> tuple[str, str, float, str]:
        if self._model is not None and self._torch is not None:
            try:
                patch = crop(image, box, pad=0.02)
                if patch.size == 0:
                    return "unknown", "unknown", 0.0, "empty crop"
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, (224, 224))
                tensor = self._torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                mean = self._torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = self._torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                with self._torch.no_grad():
                    logits = self._model((tensor - mean) / std)
                    probs = self._torch.softmax(logits, dim=1)[0]
                    conf, idx = self._torch.max(probs, dim=0)
                label = self._labels[int(idx)]
                parts = label.replace("_", " ").split(" ", 1)
                make = parts[0]
                model = parts[1] if len(parts) > 1 else "unknown"
                return make, model, float(conf), f"classifier {label} ({float(conf):.2f})"
            except Exception:
                pass
        if self.gallery:
            vec = self.embedder.embed(image, box)
            if vec is not None:
                best = max(self.gallery, key=lambda e: float(np.dot(vec, e["vector"])))
                sim = float(np.dot(vec, best["vector"]))
                if sim >= 0.62:
                    return (
                        best["make"],
                        best["model"],
                        sim,
                        f"reference gallery match (cosine {sim:.2f}, vehicle {best['vehicle_id']})",
                    )
        return "unknown", "unknown", 0.0, "make/model model not configured"


@dataclass
class FrameAttributes:
    color: ColorResult = field(default_factory=ColorResult)
    body: BodyResult = field(default_factory=BodyResult)
    plate_style: PlateStyleResult = field(default_factory=PlateStyleResult)
    make: str = "unknown"
    model: str = "unknown"
    make_model_confidence: float = 0.0
    make_model_evidence: str = ""
    frame_weight: float = 0.5


@dataclass
class FusedAttributes:
    color: ColorResult = field(default_factory=ColorResult)
    body: BodyResult = field(default_factory=BodyResult)
    plate_style: PlateStyleResult = field(default_factory=PlateStyleResult)
    make: str = "unknown"
    model: str = "unknown"
    make_model_confidence: float = 0.0
    make_model_evidence: str = ""
    evidence: list[str] = field(default_factory=list)


class AttributeExtractor:
    def __init__(self, cfg: AttributeConfig, make_model: MakeModelRecognizer | None = None):
        self.cfg = cfg
        self.make_model = make_model or MakeModelRecognizer()

    def extract_frame(
        self,
        image: np.ndarray,
        vehicle_box: tuple[float, float, float, float],
        plate_box: tuple[float, float, float, float] | None,
        detector_label: str,
        detector_confidence: float,
        quality: float,
        plate_crop: np.ndarray | None = None,
    ) -> FrameAttributes:
        color = extract_color(image, vehicle_box, plate_box)
        body = body_from_detection(detector_label, detector_confidence, vehicle_box, image)
        plate_style = extract_plate_style(plate_crop) if plate_crop is not None else PlateStyleResult()
        make, model, mm_conf, mm_evidence = self.make_model.predict(image, vehicle_box)
        return FrameAttributes(
            color=color,
            body=body,
            plate_style=plate_style,
            make=make,
            model=model,
            make_model_confidence=mm_conf,
            make_model_evidence=mm_evidence,
            frame_weight=float(max(0.05, quality)),
        )

    def fuse(self, frames: list[FrameAttributes]) -> FusedAttributes:
        color, color_evidence = fuse_colors([(f.color, f.frame_weight) for f in frames])
        body, body_evidence = fuse_body_types([(f.body, f.frame_weight) for f in frames])
        style, style_evidence = fuse_plate_styles([(f.plate_style, f.frame_weight) for f in frames])
        # Make/model: highest-confidence frame wins; no averaging of labels.
        best_mm = ("unknown", "unknown", 0.0, "make/model model not configured")
        for f in frames:
            if f.make_model_confidence > best_mm[2]:
                best_mm = (f.make, f.model, f.make_model_confidence, f.make_model_evidence)
        evidence: list[str] = []
        evidence.extend(color_evidence)
        evidence.extend(body_evidence)
        evidence.extend(style_evidence)
        if best_mm[2] > 0:
            evidence.append(best_mm[3])
        return FusedAttributes(
            color=color,
            body=body,
            plate_style=style,
            make=best_mm[0],
            model=best_mm[1],
            make_model_confidence=best_mm[2],
            make_model_evidence=best_mm[3],
            evidence=evidence,
        )
