"""Plate OCR backends.

Two backends are provided:

``PaddleOCRBackend``
    Production path.  PaddleOCR's detector + recogniser handle perspective,
    low contrast and multi-line plates.  Per-character confidences are
    initialised from the recogniser's line score and then refined by the
    multi-frame fusion stage (cross-frame agreement is the strongest
    character-level signal).

``TemplateOCRBackend``
    Dependency-free fallback and deterministic demo/test backend.  It
    segments characters with classical morphology and matches them against
    rendered font templates.  It gives genuine per-character scores, but it is
    **not** a replacement for PaddleOCR on real CCTV.

Both return :class:`OCRLine` objects with a text string and one confidence
per character.  ``read_plate`` runs several preprocessing variants and picks
the most plate-like result.
"""

from __future__ import annotations

import functools
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import OCRConfig
from .plate import clean_text, plate_likeness

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_SIZE = 32


@dataclass
class OCRLine:
    text: str
    confidence: float
    char_confidences: list[float] = field(default_factory=list)
    backend: str = "unknown"
    alternatives: list[list[tuple[str, float]]] = field(default_factory=list)

    @property
    def known_chars(self) -> int:
        return sum(1 for c in self.text if c.isalnum())


def _font_candidates() -> list[str]:
    candidates: list[str] = []
    if os.name == "nt":
        root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        names = [
            "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
            "verdana.ttf", "verdanab.ttf", "tahoma.ttf", "tahomabd.ttf",
            "segoeui.ttf", "segoeuib.ttf",
        ]
        candidates = [str(root / n) for n in names]
    else:
        roots = [
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/truetype/liberation",
            "/usr/share/fonts/truetype/freefont",
            "/System/Library/Fonts/Supplemental",
            "/Library/Fonts",
        ]
        names = [
            "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
            "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
            "FreeSans.ttf", "FreeSansBold.ttf",
            "Arial.ttf", "Arial Bold.ttf",
        ]
        for root in roots:
            candidates.extend(str(Path(root) / n) for n in names)
    return [p for p in candidates if Path(p).exists()]


def _normalise_char_image(gray: np.ndarray) -> np.ndarray | None:
    """Crop ink, resize by height, centre on a 32x32 canvas."""

    if gray is None or gray.size == 0:
        return None
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # Ink is bright, background dark.  Normalise polarity.
    if float(gray.mean()) > 127:
        gray = 255 - gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 4:
        return None
    ink = gray[y:y + h, x:x + w]
    target_h = 26
    target_w = max(3, int(round(w * target_h / h)))
    target_w = min(target_w, 30)
    ink = cv2.resize(ink, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CHAR_SIZE, CHAR_SIZE), dtype=np.float32)
    x0 = (CHAR_SIZE - target_w) // 2
    y0 = (CHAR_SIZE - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = ink.astype(np.float32) / 255.0
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    return canvas


class _TemplateBank:
    """Rendered A-Z/0-9 templates, cached once per process."""

    _instance: "_TemplateBank | None" = None

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.label_to_indices: dict[str, list[int]] = {c: [] for c in ALPHABET}
        templates: list[np.ndarray] = []
        fonts = _font_candidates()
        # Use a subset to keep startup fast but retain regular/bold variation.
        for font_path in fonts[:6]:
            for size in (64, 84):
                try:
                    font = ImageFont.truetype(font_path, size)
                except Exception:
                    continue
                for ch in ALPHABET:
                    image = Image.new("L", (160, 160), 0)
                    draw = ImageDraw.Draw(image)
                    try:
                        draw.text((80, 80), ch, font=font, fill=255, anchor="mm")
                    except Exception:
                        draw.text((50, 40), ch, font=font, fill=255)
                    arr = np.array(image)
                    norm = _normalise_char_image(arr)
                    if norm is None:
                        continue
                    idx = len(templates)
                    templates.append(norm)
                    self.labels.append(ch)
                    self.label_to_indices[ch].append(idx)
        if not templates:
            raise RuntimeError(
                "No usable system fonts were found for the template OCR backend. "
                "Install PaddleOCR or add a TTF font (see README)."
            )
        self.templates = np.stack(templates).reshape(len(templates), -1).astype(np.float32)
        # Pre-normalise for cosine correlation.
        self.templates -= self.templates.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(self.templates, axis=1, keepdims=True)
        self.templates = self.templates / np.maximum(norms, 1e-6)
        self._label_index = {c: np.array(v, dtype=np.int32) for c, v in self.label_to_indices.items() if v}

    @classmethod
    def get(cls) -> "_TemplateBank":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def match(self, char_image: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        norm = _normalise_char_image(char_image)
        if norm is None:
            return []
        q = norm.reshape(-1)
        q = q - q.mean()
        q = q / max(1e-6, float(np.linalg.norm(q)))
        corr = self.templates @ q  # cosine correlation, shape (N,)
        # Add a binary IoU term to be tolerant of stroke thickness.
        q_bin = (norm > 0.5).reshape(-1)
        scores = np.full(len(ALPHABET), -1.0, dtype=np.float32)
        for i, ch in enumerate(ALPHABET):
            idxs = self._label_index.get(ch)
            if idxs is None or len(idxs) == 0:
                continue
            sub = self.templates[idxs]
            c = float(np.max(corr[idxs]))
            # Reconstruct approximate binary templates from stored floats is
            # not possible after normalisation; correlation alone is stable.
            scores[i] = max(scores[i], c)
        order = np.argsort(scores)[::-1][:top_k]
        out: list[tuple[str, float]] = []
        for idx in order:
            if scores[idx] <= -0.5:
                continue
            out.append((ALPHABET[int(idx)], float(max(0.0, scores[idx]))))
        return out


class TemplateOCRBackend:
    name = "template"

    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self._bank = _TemplateBank.get()

    def read(self, image: np.ndarray) -> list[OCRLine]:
        if image is None or image.size == 0:
            return []
        lines: list[OCRLine] = []
        for variant_name, variant in _segmentation_variants(image):
            line = self._read_variant(variant)
            if line is not None:
                line.backend = f"template:{variant_name}"
                lines.append(line)
        return lines

    def _read_variant(self, image: np.ndarray) -> OCRLine | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        if h < 12 or w < 30:
            return None
        # Upscale small plates; downscale huge crops.
        target_h = 120
        scale = target_h / max(1, h)
        if scale > 1:
            gray = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
        elif h > 220:
            gray = cv2.resize(gray, (int(w * 220 / h), 220), interpolation=cv2.INTER_AREA)
        # Padding prevents characters touching the crop edge from being
        # discarded as a border artefact.
        gray = cv2.copyMakeBorder(gray, 10, 10, 12, 12, cv2.BORDER_REPLICATE)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        H, W = gray.shape[:2]

        box_sets: list[list[tuple[int, int, int, int]]] = []
        best_boxes: list[tuple[int, int, int, int]] = []
        for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            _, th = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes: list[tuple[int, int, int, int]] = []
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                if bh < 0.34 * H or bh > 0.98 * H:
                    continue
                if bw < max(3, 0.035 * H) or bw > 0.55 * H:
                    continue
                if y < 0.02 * H or y + bh > 1.02 * H:
                    continue
                boxes.append((x, y, bw, bh))
            boxes = _merge_boxes(boxes, overlap=0.25)
            boxes = _filter_char_heights(boxes)
            if len(boxes) > len(best_boxes):
                best_boxes = boxes
            if len(boxes) >= 4:
                box_sets.append(boxes)
            if len(boxes) >= 7:
                break
        if len(best_boxes) >= 4:
            box_sets.append(best_boxes)
        projection = _projection_boxes(gray)
        if len(projection) >= 4:
            box_sets.append(projection)

        best_line: OCRLine | None = None
        best_score = -1.0
        seen: set[str] = set()
        for boxes in box_sets:
            line = self._recognise_boxes(gray, boxes)
            if line is None or line.text in seen:
                continue
            seen.add(line.text)
            score = (
                0.55 * plate_likeness(line.text)
                + 0.25 * _length_score(line.text)
                + 0.20 * max(0.0, min(1.0, line.confidence))
            )
            if score > best_score:
                best_score = score
                best_line = line
        return best_line

    def _recognise_boxes(
        self, gray: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> OCRLine | None:
        H, W = gray.shape[:2]
        boxes = sorted(boxes, key=lambda b: b[0])
        if len(boxes) > 12:
            heights = sorted(b[3] for b in boxes)
            med = heights[len(heights) // 2]
            boxes = sorted(boxes, key=lambda b: abs(b[3] - med))[:11]
            boxes = sorted(boxes, key=lambda b: b[0])
        if len(boxes) < 4:
            return None

        chars: list[str] = []
        confs: list[float] = []
        alternatives: list[list[tuple[str, float]]] = []
        for (x, y, bw, bh) in boxes:
            pad = max(1, int(0.08 * bh))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
            patch = gray[y0:y1, x0:x1]
            ranked = self._bank.match(patch, top_k=3)
            if not ranked:
                chars.append("?")
                confs.append(0.0)
                alternatives.append([])
                continue
            top_char, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = max(0.0, top_score - second_score)
            # Calibrated character confidence: absolute match + margin.
            conf = max(0.0, min(1.0, 0.68 * top_score + 0.32 * min(1.0, margin * 2.0)))
            if top_score < 0.32 or margin < 0.025:
                chars.append("?")
                confs.append(conf * 0.5)
            else:
                chars.append(top_char)
                confs.append(conf)
            alternatives.append([(c, round(s, 4)) for c, s in ranked])

        text = "".join(chars)
        if sum(1 for c in text if c != "?") < 4:
            return None
        confidence = float(np.mean([c for c in confs if c > 0])) if any(c > 0 for c in confs) else 0.0
        return OCRLine(
            text=text,
            confidence=confidence,
            char_confidences=confs,
            backend="template",
            alternatives=alternatives,
        )


class PaddleOCRBackend:
    name = "paddle"

    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PaddleOCR is not installed. Install paddlepaddle + paddleocr, "
                "or run with --ocr template."
            ) from exc
        kwargs = {
            "use_angle_cls": bool(cfg.use_angle_cls),
            "lang": cfg.lang,
            "show_log": False,
        }
        try:
            self._ocr = PaddleOCR(**kwargs)
        except TypeError:
            # PaddleOCR 3.x removed show_log.
            kwargs.pop("show_log", None)
            self._ocr = PaddleOCR(**kwargs)
        self._predict = hasattr(self._ocr, "predict")

    def read(self, image: np.ndarray) -> list[OCRLine]:
        if image is None or image.size == 0:
            return []
        try:
            if self._predict:
                raw = self._ocr.predict(image)
                lines = _parse_paddle_v3(raw)
            else:
                raw = self._ocr.ocr(image, cls=bool(self.cfg.use_angle_cls))
                lines = _parse_paddle_v2(raw)
        except Exception:
            return []
        return lines


def _parse_paddle_v2(raw) -> list[OCRLine]:
    lines: list[OCRLine] = []
    if not raw:
        return lines
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if not page:
            continue
        entries = page if isinstance(page, list) else [page]
        for entry in entries:
            try:
                box, rec = entry[0], entry[1]
                if isinstance(rec, (list, tuple)):
                    text, score = rec[0], float(rec[1])
                else:
                    text, score = str(rec), 0.0
            except Exception:
                continue
            text = clean_text(str(text))
            if sum(1 for c in text if c.isalnum()) < 3:
                continue
            lines.append(
                OCRLine(
                    text=text,
                    confidence=max(0.0, min(1.0, score)),
                    char_confidences=[max(0.0, min(1.0, score))] * len(text),
                    backend="paddle",
                )
            )
    return lines


def _parse_paddle_v3(raw) -> list[OCRLine]:
    lines: list[OCRLine] = []
    if not raw:
        return lines
    for item in raw:
        if isinstance(item, dict):
            texts = item.get("rec_texts") or item.get("rec_text") or []
            scores = item.get("rec_scores") or item.get("rec_score") or []
            if isinstance(texts, str):
                texts, scores = [texts], [scores]
            for text, score in zip(texts, scores):
                text = clean_text(str(text))
                if sum(1 for c in text if c.isalnum()) < 3:
                    continue
                score = float(score)
                lines.append(
                    OCRLine(
                        text=text,
                        confidence=max(0.0, min(1.0, score)),
                        char_confidences=[max(0.0, min(1.0, score))] * len(text),
                        backend="paddle",
                    )
                )
        else:
            lines.extend(_parse_paddle_v2([item]))
    return lines


def _segmentation_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [("raw", image)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))
    return variants


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    """Merge heavily overlapping components (broken strokes)."""

    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        did_merge = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            ix1, iy1 = max(x, mx), max(y, my)
            ix2, iy2 = min(x + w, mx + mw), min(y + h, my + mh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / max(1, min(w * h, mw * mh)) > overlap:
                nx1, ny1 = min(x, mx), min(y, my)
                nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                did_merge = True
                break
        if not did_merge:
            merged.append(box)
    return merged


def _filter_char_heights(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = sorted(b[3] for b in boxes)
    median = heights[len(heights) // 2]
    return [b for b in boxes if 0.52 * median <= b[3] <= 1.55 * median]


def _projection_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Segment characters by vertical ink projection.

    Contour segmentation can merge adjacent glyphs when the plate is blurred.
    Projection segmentation uses the gaps between characters and then splits
    any run that is clearly wider than one glyph.  It is only a layout
    heuristic: each slot still has to pass the template match to become a
    character, otherwise it stays ``?``.
    """

    if gray is None or gray.size == 0:
        return []
    h, w = gray.shape[:2]
    if h < 20 or w < 40:
        return []
    # Text is dark on a bright plate in almost all Indian plate styles.
    bright_background = float(np.median(gray)) > 127
    polarity = cv2.THRESH_BINARY_INV if bright_background else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    # Remove the long horizontal plate border; it would bridge every glyph.
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 12), 1))
    )
    mask = cv2.subtract(mask, horizontal)

    projection = (mask > 0).sum(axis=0).astype(np.float32)
    if float(projection.max()) <= 0:
        return []
    threshold = max(1.0, 0.045 * h)
    active = projection > threshold
    runs: list[tuple[int, int]] = []
    start = None
    for x, flag in enumerate(active):
        if flag and start is None:
            start = x
        elif not flag and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    if not runs:
        return []
    # Bridge tiny gaps inside a glyph (e.g. a broken stem).
    max_gap = max(2, int(0.012 * w))
    merged: list[tuple[int, int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    merged = [(a, b) for a, b in merged if 0.012 * w <= (b - a) <= 0.22 * w]
    if len(merged) < 4:
        return []
    widths = sorted(b - a for a, b in merged)
    median_width = widths[len(widths) // 2]

    boxes: list[tuple[int, int, int, int]] = []
    for a, b in merged:
        width = b - a
        pieces = 1
        if width > 1.55 * median_width:
            pieces = max(2, int(round(width / max(1.0, median_width))))
        step = width / pieces
        for k in range(pieces):
            x0 = int(round(a + k * step))
            x1 = int(round(a + (k + 1) * step))
            if x1 - x0 < 2:
                continue
            column = mask[:, x0:x1]
            rows = np.where(column.any(axis=1))[0]
            if rows.size < 4:
                continue
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            if y1 - y0 < 0.25 * h:
                continue
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


class AutoOCR:
    """PaddleOCR when available, template OCR otherwise."""

    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.warnings: list[str] = []
        self.backend = None
        mode = (cfg.backend or "auto").lower()
        if mode in ("auto", "paddle"):
            try:
                self.backend = PaddleOCRBackend(cfg)
                self.name = "paddle"
                return
            except Exception as exc:
                if mode == "paddle":
                    raise
                self.warnings.append(f"PaddleOCR unavailable ({exc}); using template OCR fallback.")
        self.backend = TemplateOCRBackend(cfg)
        self.name = "template"

    def read(self, image: np.ndarray) -> list[OCRLine]:
        return self.backend.read(image)  # type: ignore[union-attr]


def _length_score(text: str) -> float:
    n = sum(1 for c in text if c.isalnum())
    if 9 <= n <= 11:
        return 1.0
    if n == 8:
        return 0.8
    if n == 7:
        return 0.65
    return 0.35


def read_plate(crop: np.ndarray, ocr: AutoOCR, cfg: OCRConfig) -> OCRLine | None:
    """Run OCR on several variants and return the most plate-like line."""

    if crop is None or crop.size == 0:
        return None
    variants: list[tuple[str, np.ndarray]] = [("raw", crop)]
    h, w = crop.shape[:2]
    if min(h, w) < 80:
        scale = max(2.0, 120.0 / max(1, h))
        up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        variants.append(("upscaled", up))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    best: OCRLine | None = None
    best_score = -1.0
    seen: set[str] = set()
    for variant_name, variant in variants:
        for line in ocr.read(variant):
            text = clean_text(line.text)
            if text in seen:
                continue
            seen.add(text)
            known = sum(1 for c in text if c.isalnum())
            if known < 4:
                continue
            score = (
                0.45 * plate_likeness(text)
                + 0.30 * _length_score(text)
                + 0.25 * max(0.0, min(1.0, line.confidence))
            )
            if score > best_score:
                best_score = score
                best = OCRLine(
                    text=text,
                    confidence=line.confidence,
                    char_confidences=line.char_confidences,
                    backend=f"{line.backend}:{variant_name}",
                    alternatives=line.alternatives,
                )
    if best is None:
        return None
    if best.confidence < cfg.min_line_conf:
        return None
    return best


def build_ocr(cfg: OCRConfig) -> AutoOCR:
    return AutoOCR(cfg)
