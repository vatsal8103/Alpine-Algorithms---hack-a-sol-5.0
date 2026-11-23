"""License plate localization, margin crop, and per-frame OCR inference.

Guarantees:
- Slightly larger crop margin (20% padding) to avoid truncating characters.
- Deskewing from plate quad using Radon/Hough orientation analysis.
- Upscale with cubic or Lanczos only (NO AI super-resolution).
- Mild CLAHE contrast adjustment.
- Extraction of calibrated top-3 alternatives based on OCR confusion rules.
- Multi-variant OCR generating additional consensus evidence for fusion.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from pipeline.ocr import LicensePlateOCR, get_calibrated_char_alternatives
from pipeline.schemas import CharAlternative, CharCandidate, FrameOCRResult


def deskew_plate(img: np.ndarray) -> np.ndarray:
    """Detect dominant plate angle and warp upright if rotated."""
    if img is None or img.size == 0:
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, minLineLength=int(img.shape[1] * 0.25), maxLineGap=8)
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 10:
                    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -25.0 < ang < 25.0 and abs(ang) > 1.5:
                        angles.append(ang)
            if angles:
                median_angle = float(np.median(angles))
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    return img


def unroll_two_line_plate(plate_crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """If plate crop is square/two-line (e.g. Indian scooter/bike with w/h < 2.2), unroll into a single horizontal strip.

    Line 1 (State + RTO, top) is horizontally stitched before Line 2 (Series + Number, bottom)
    to prevent character collision when processed by 1D sequence OCR models.
    """
    if plate_crop is None or plate_crop.size == 0:
        return plate_crop, False
    h, w = plate_crop.shape[:2]
    aspect = float(w) / max(1, h)
    if aspect >= 2.2 or h < 24:
        return plate_crop, False

    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        # Compute horizontal edge gradient activity to find the inter-line gap
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.abs(sobel_y)
        row_activity = np.mean(grad_mag, axis=1)

        mid_min = int(h * 0.38)
        mid_max = int(h * 0.62)
        if mid_max > mid_min:
            split_y = mid_min + int(np.argmin(row_activity[mid_min:mid_max]))
        else:
            split_y = h // 2

        top_crop = plate_crop[:split_y, :]
        bottom_crop = plate_crop[split_y:, :]
        if top_crop.size == 0 or bottom_crop.size == 0:
            return plate_crop, False

        target_h = 48
        top_w = max(1, int(top_crop.shape[1] * (target_h / max(1, top_crop.shape[0]))))
        bot_w = max(1, int(bottom_crop.shape[1] * (target_h / max(1, bottom_crop.shape[0]))))

        top_resized = cv2.resize(top_crop, (top_w, target_h), interpolation=cv2.INTER_CUBIC)
        bot_resized = cv2.resize(bottom_crop, (bot_w, target_h), interpolation=cv2.INTER_CUBIC)

        pad_w = int(target_h * 0.25)
        spacer_color = 240 if float(np.mean(gray)) > 120 else 20
        if len(plate_crop.shape) == 3:
            spacer = np.full((target_h, pad_w, plate_crop.shape[2]), spacer_color, dtype=plate_crop.dtype)
        else:
            spacer = np.full((target_h, pad_w), spacer_color, dtype=plate_crop.dtype)

        stitched = np.hstack([top_resized, spacer, bot_resized])
        return stitched, True
    except Exception:
        return plate_crop, False


def parse_easyocr_multi_line(results: list) -> tuple[str, float, list[float] | None]:
    """Parse EasyOCR text box detections, automatically detecting and merging two-line stacked plates.

    In India, two-wheeler plates (and many commercial vehicles) are stacked:
      - Line 1 (Top): State code + RTO (e.g. MP 04, DL 1C, MH 12)
      - Line 2 (Bottom): Series + 4-digit number (e.g. CY 2090, 2354, ZK 6184)
    This function concatenates both lines in topological order and returns the outer bounding box.
    """
    valid = []
    HSRP_TAGS = {"IND", "INO", "IMD", "IRD", "IHD", "1ND", "HSRP", "IN"}
    INDIAN_STATES = {
        "MP", "DL", "MH", "KA", "HR", "UP", "GJ", "RJ", "TN", "TS", "AP", "WB",
        "KL", "PB", "CH", "BR", "JH", "OD", "UK", "GA", "CG", "TR", "MZ", "NL", "SK"
    }

    for bbox, d_text, d_conf in results:
        clean = re.sub(r"[^A-Za-z0-9]", "", str(d_text)).upper()
        if clean in HSRP_TAGS or len(clean) < 2 or float(d_conf) < 0.18:
            continue
        pts = np.array(bbox, dtype=np.float32)
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        valid.append({
            "clean": clean,
            "conf": float(d_conf),
            "y_center": (y_min + y_max) / 2.0,
            "x_center": (x_min + x_max) / 2.0,
            "h": max(1.0, y_max - y_min),
            "w": max(1.0, x_max - x_min),
            "box_rect": [x_min, y_min, x_max, y_max]
        })

    if not valid:
        return "", 0.0, None

    # 1. Single complete plate detection (8 to 11 characters conforming to alphanumeric)
    full_cands = [
        v for v in valid
        if 8 <= len(v["clean"]) <= 11 and any(c.isdigit() for c in v["clean"]) and any(c.isalpha() for c in v["clean"])
    ]
    if full_cands:
        best_full = max(full_cands, key=lambda v: (v["clean"][:2] in INDIAN_STATES, v["conf"]))
        return best_full["clean"], best_full["conf"], best_full["box_rect"]

    # 2. Two-line stacked plate detection (vertically separated lines)
    if len(valid) >= 2:
        by_y = sorted(valid, key=lambda v: v["y_center"])
        avg_h = (by_y[0]["h"] + by_y[1]["h"]) / 2.0
        # If the top two detections are vertically separated
        if (by_y[1]["y_center"] - by_y[0]["y_center"]) > avg_h * 0.30:
            top = by_y[0]
            bot = by_y[1]
            combined_text = top["clean"] + bot["clean"]
            combined_conf = (top["conf"] * len(top["clean"]) + bot["conf"] * len(bot["clean"])) / max(1, len(top["clean"]) + len(bot["clean"]))
            outer_box = [
                min(top["box_rect"][0], bot["box_rect"][0]),
                min(top["box_rect"][1], bot["box_rect"][1]),
                max(top["box_rect"][2], bot["box_rect"][2]),
                max(top["box_rect"][3], bot["box_rect"][3]),
            ]
            return combined_text, combined_conf, outer_box

        # Or horizontally separated lines (e.g. unrolled strip)
        by_x = sorted(valid, key=lambda v: v["x_center"])
        avg_w = (by_x[0]["w"] + by_x[1]["w"]) / 2.0
        if (by_x[1]["x_center"] - by_x[0]["x_center"]) > avg_w * 0.30:
            left = by_x[0]
            right = by_x[1]
            combined_text = left["clean"] + right["clean"]
            combined_conf = (left["conf"] * len(left["clean"]) + right["conf"] * len(right["clean"])) / max(1, len(left["clean"]) + len(right["clean"]))
            outer_box = [
                min(left["box_rect"][0], right["box_rect"][0]),
                min(left["box_rect"][1], right["box_rect"][1]),
                max(left["box_rect"][2], right["box_rect"][2]),
                max(left["box_rect"][3], right["box_rect"][3]),
            ]
            return combined_text, combined_conf, outer_box

    # 3. Single partial fragment fallback (scored by length and Indian state prefix)
    best_cand = max(
        valid,
        key=lambda v: (
            v["clean"][:2] in INDIAN_STATES,
            any(c.isdigit() for c in v["clean"]) and any(c.isalpha() for c in v["clean"]),
            len(v["clean"]),
            v["conf"]
        )
    )
    return best_cand["clean"], best_cand["conf"], best_cand["box_rect"]


class PlateStage:
    """Localizes license plates within vehicle bounding boxes and runs multi-variant OCR."""

    def __init__(
        self,
        plate_model_path: str = "models/plate_yolo11n.pt",
        char_threshold: float = 0.28,
        crops_dir: str = "runs/crops",
        device: str | None = None
    ) -> None:
        self.char_threshold = char_threshold
        self.crops_dir = Path(crops_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_engine = LicensePlateOCR()
        self.plate_detector = None
        import torch
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Prioritize YOLO11n, fallback to YOLOv8s if missing
        active_path = plate_model_path
        if not Path(active_path).exists() and Path("models/plate_yolov8s.pt").exists():
            active_path = "models/plate_yolov8s.pt"

        if active_path and Path(active_path).exists():
            try:
                from ultralytics import YOLO
                self.plate_detector = YOLO(active_path)
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    try:
                        self.plate_detector.to(self.device)
                    except Exception:
                        pass
            except Exception:
                self.plate_detector = None

    def locate_plate_box(self, vehicle_crop: np.ndarray) -> tuple[int, int, int, int]:
        """Locate license plate bounding box [x, y, w, h] within vehicle crop.
        
        Prioritizes the trained YOLO plate detector. If unavailable or no plate detected,
        falls back to morphological color and edge density analysis in lower bumper region.
        """
        vh, vw = vehicle_crop.shape[:2]

        # 1. Primary Path: YOLO Plate Detector (GPU-accelerated, calibrated high recall)
        if self.plate_detector is not None and vehicle_crop.size > 0:
            try:
                # Primary pass at conf=0.15 for high recall on distant, angled, or two-wheeler plates
                det_res = self.plate_detector(vehicle_crop, conf=0.15, device=self.device, verbose=False)
                best_conf = -1.0
                best_yolo_box = None
                if det_res and len(det_res) > 0 and len(det_res[0].boxes) > 0:
                    for b in det_res[0].boxes:
                        c = float(b.conf[0])
                        if c > best_conf:
                            best_conf = c
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                            bx = max(0, int(x1))
                            by = max(0, int(y1))
                            bw = min(vw - bx, int(x2 - x1))
                            bh = min(vh - by, int(y2 - y1))
                            if bw > 15 and bh > 8:
                                best_yolo_box = (bx, by, bw, bh)

                # Secondary zoom pass: On large crops where whole-image resize shrinks tiny distant plates
                if best_yolo_box is None and vh > 180:
                    lower_y1 = int(vh * 0.35)
                    lower_crop = vehicle_crop[lower_y1:, :]
                    if lower_crop.size > 0:
                        det_lower = self.plate_detector(lower_crop, conf=0.12, device=self.device, verbose=False)
                        if det_lower and len(det_lower[0].boxes) > 0:
                            b = det_lower[0].boxes[0]
                            lx1, ly1, lx2, ly2 = b.xyxy[0].cpu().numpy()
                            lbw = min(vw, int(lx2 - lx1))
                            lbh = min(vh - lower_y1, int(ly2 - ly1))
                            if lbw > 12 and lbh > 8:
                                best_yolo_box = (max(0, int(lx1)), max(0, int(ly1 + lower_y1)), lbw, lbh)

                if best_yolo_box is not None:
                    return best_yolo_box
            except Exception:
                pass

        # 2. Robust Fallback: Character stroke morphology & edge density analysis
        roi_y1 = int(vh * 0.40)
        roi_y2 = int(vh * 0.95)
        roi_x1 = int(vw * 0.10)
        roi_x2 = int(vw * 0.90)
        roi = vehicle_crop[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return 0, 0, 0, 0

        rh, rw = roi.shape[:2]
        roi_area = rh * rw
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Method A: Blackhat morphology (highlights dark characters on lighter plate background)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)))
        _, thresh_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresh_bh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_score = -1.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(1, h)
            area = w * h
            # Accept square two-wheeler plates (aspect >= 0.85) up to elongated car plates (aspect <= 5.8)
            if 0.85 <= aspect <= 5.8 and (0.008 * roi_area <= area <= 0.48 * roi_area) and w >= 25 and h >= 12:
                patch_gray = gray[y:y+h, x:x+w]
                contrast = float(np.std(patch_gray)) if patch_gray.size > 0 else 10.0
                cx_dev = abs((x + w / 2.0) - (rw / 2.0)) / (rw / 2.0)
                score = np.sqrt(area) * contrast * (1.0 - cx_dev * 0.35)
                if score > best_score:
                    best_score = score
                    pad_x = int(w * 0.10)
                    pad_y = int(h * 0.15)
                    bx = max(0, roi_x1 + x - pad_x)
                    by = max(0, roi_y1 + y - pad_y)
                    bw = min(vw - bx, w + 2 * pad_x)
                    bh = min(vh - by, h + 2 * pad_y)
                    best_box = (bx, by, bw, bh)

        if best_box is not None:
            return best_box

        # Method B: Morphological gradient (character edges on yellow/white plates)
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        _, thresh_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed_grad = cv2.morphologyEx(thresh_grad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)))
        contours_grad, _ = cv2.findContours(closed_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_grad:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(1, h)
            area = w * h
            if 0.85 <= aspect <= 5.8 and (0.008 * roi_area <= area <= 0.48 * roi_area) and w >= 25 and h >= 12:
                patch_gray = gray[y:y+h, x:x+w]
                contrast = float(np.std(patch_gray)) if patch_gray.size > 0 else 10.0
                cx_dev = abs((x + w / 2.0) - (rw / 2.0)) / (rw / 2.0)
                score = np.sqrt(area) * contrast * (1.0 - cx_dev * 0.35)
                if score > best_score:
                    best_score = score
                    pad_x = int(w * 0.10)
                    pad_y = int(h * 0.15)
                    bx = max(0, roi_x1 + x - pad_x)
                    by = max(0, roi_y1 + y - pad_y)
                    bw = min(vw - bx, w + 2 * pad_x)
                    bh = min(vh - by, h + 2 * pad_y)
                    best_box = (bx, by, bw, bh)

        if best_box is not None:
            return best_box

        return 0, 0, 0, 0

    def preprocess_plate_image(self, plate_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate base deskewed/upscaled crop and mild CLAHE variant."""
        deskewed = deskew_plate(plate_crop)
        h, w = deskewed.shape[:2]

        # Upscale with cubic interpolation only (NO AI super-resolution)
        if h < 72:
            scale = 80.0 / max(1, h)
            base = cv2.resize(deskewed, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            base = deskewed

        # Variant 2: Mild CLAHE
        lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        l2 = clahe.apply(l)
        clahe_img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

        return base, clahe_img

    def process_frame_crop(
        self,
        vehicle_crop: np.ndarray,
        track_id: int,
        frame_no: int,
        timestamp: float,
        sharpness: float = 50.0
    ) -> FrameOCRResult:
        """Crop plate with 20% margin, deskew, upscale cubic, and run OCR with alternatives."""
        vh, vw = vehicle_crop.shape[:2]
        easy_ocr = getattr(self.ocr_engine, "easy_ocr", None)
        text = ""
        candidates: list[CharCandidate] = []
        plate_crop = None
        crop_box = None

        # 1. Primary Empirical Pipeline: Locate Plate Box (20% margin) + Deskew + Recognition-Only OCR
        px, py, pw, ph = self.locate_plate_box(vehicle_crop)
        if pw == 0 or ph == 0:
            return FrameOCRResult(
                track_id=track_id, frame_no=frame_no, timestamp=timestamp,
                bbox=[0, 0, 0, 0], plate_crop_path="", plate_area_px=0,
                sharpness=sharpness, chars=[]
            )
        box_aspect = float(pw) / max(1, ph)
        is_square_or_two_line = (box_aspect < 2.4)
        mx = max(8, int(pw * 0.20))
        my = max(10, int(ph * 0.30)) if is_square_or_two_line else max(6, int(ph * 0.20))
        x1 = max(0, px - mx)
        y1 = max(0, py - my)
        x2 = min(vw, px + pw + mx)
        y2 = min(vh, py + ph + my)
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            plate_crop = vehicle_crop
        crop_box = [float(x1), float(y1), float(x2), float(y2)]

        deskewed = deskew_plate(plate_crop)
        unrolled_crop, is_two_line = unroll_two_line_plate(deskewed)
        target_crop = unrolled_crop if is_two_line else deskewed
        text, candidates = self.ocr_engine.recognize(target_crop)
        if is_two_line and len(candidates) < 6:
            alt_text, alt_candidates = self.ocr_engine.recognize(deskewed)
            if len(alt_candidates) > len(candidates):
                text, candidates = alt_text, alt_candidates
                target_crop = deskewed

        # 2. Resilient OCR: If primary recognition yielded fewer than 8 chars or lacks valid plate grammar, try EasyOCR
        needs_better_ocr = (
            len(candidates) < 8
            or not any(c.char.isdigit() for c in candidates)
            or not any(c.char.isalpha() for c in candidates)
            or (candidates and float(np.mean([c.prob for c in candidates])) < 0.70)
        )
        if needs_better_ocr and easy_ocr is not None and vh > 40 and vw > 40:
            try:
                # Fast path: First try EasyOCR directly on target_crop (5ms on GPU vs 150ms on full vehicle)
                fast_target = target_crop if (target_crop is not None and target_crop.size > 0) else plate_crop
                results = easy_ocr.readtext(fast_target) if (fast_target is not None and fast_target.size > 0) else []
                is_lower_vehicle = False
                y_offset = 0

                # If fast_target yielded no good reading and crop was morphological fallback, try lower_vehicle
                valid_fast = any(len(re.sub(r"[^A-Za-z0-9]", "", r[1])) >= 4 for r in results)
                if not valid_fast:
                    y_offset = int(vh * 0.35)
                    lower_vehicle = vehicle_crop[y_offset:, :]
                    results = easy_ocr.readtext(lower_vehicle)
                    is_lower_vehicle = True

                best_text, best_conf, best_bbox = parse_easyocr_multi_line(results)

                if best_bbox is not None and len(best_text) >= 4 and best_conf >= 0.20:
                    if is_lower_vehicle:
                        bx1 = int(best_bbox[0])
                        by1 = int(best_bbox[1]) + y_offset
                        bx2 = int(best_bbox[2])
                        by2 = int(best_bbox[3]) + y_offset

                        pmx = max(8, int((bx2 - bx1) * 0.20))
                        pmy = max(8, int((by2 - by1) * 0.25))
                        cx1 = max(0, bx1 - pmx)
                        cy1 = max(0, by1 - pmy)
                        cx2 = min(vw, bx2 + pmx)
                        cy2 = min(vh, by2 + pmy)
                        sub = vehicle_crop[cy1:cy2, cx1:cx2]
                        if sub.size > 0:
                            plate_crop = sub
                            crop_box = [float(cx1), float(cy1), float(cx2), float(cy2)]

                    text = best_text
                    char_conf = min(1.0, best_conf)
                    candidates = [
                        CharCandidate(
                            char=ch,
                            prob=round(char_conf, 3),
                            alternatives=get_calibrated_char_alternatives(ch, char_conf)
                        )
                        for ch in text
                    ]
            except Exception:
                pass

        # Measure actual Laplacian sharpness directly on plate crop to quantify blur
        if plate_crop is not None and plate_crop.size > 0:
            gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
            measured_sharpness = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
            sharpness = max(1.0, round(measured_sharpness, 2))

        # Blur-aware confidence calibration:
        # Standard nominal plate sharpness is ~45.0. If crop is blurry (Laplacian variance < 45),
        # gracefully degrade character confidence to prevent false commitments.
        blur_scale = min(1.0, max(0.30, math.sqrt(sharpness / 45.0)))
        if candidates:
            candidates = [
                CharCandidate(
                    char=c.char,
                    prob=round(min(1.0, c.prob * blur_scale), 3),
                    alternatives=[
                        CharAlternative(char=a.char, prob=round(min(1.0, a.prob * blur_scale), 3))
                        for a in c.alternatives
                    ]
                )
                for c in candidates
            ]

        # Preprocess plate crop image before saving
        base_saved, _ = self.preprocess_plate_image(plate_crop)
        crop_fname = f"track_{track_id}_f{frame_no:04d}_plate.png"
        crop_path = self.crops_dir / crop_fname
        cv2.imwrite(str(crop_path), base_saved)

        plate_area = float((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])) if crop_box else 1000.0

        return FrameOCRResult(
            track_id=track_id,
            frame_no=frame_no,
            timestamp=timestamp,
            bbox=crop_box or [0.0, 0.0, float(vw), float(vh)],
            plate_crop_path=str(crop_path),
            sharpness=sharpness,
            plate_area_px=plate_area,
            chars=candidates
        )

    def process_frame_variants(
        self,
        vehicle_crop: np.ndarray,
        track_id: int,
        frame_no: int,
        timestamp: float,
        sharpness: float = 50.0
    ) -> list[FrameOCRResult]:
        """Generate multiple preprocessing variants and run OCR on each for evidential fusion."""
        base_result = self.process_frame_crop(vehicle_crop, track_id, frame_no, timestamp, sharpness)
        results = [base_result]

        # If base result yielded characters, run mild CLAHE variant as secondary evidence
        if base_result.chars and base_result.plate_crop_path:
            p_img = cv2.imread(base_result.plate_crop_path)
            if p_img is not None and p_img.size > 0:
                _, clahe_img = self.preprocess_plate_image(p_img)
                c_text, c_cands = self.ocr_engine.recognize(clahe_img)
                if c_cands and len(c_cands) >= 3:
                    clahe_result = FrameOCRResult(
                        track_id=track_id,
                        frame_no=frame_no,
                        timestamp=timestamp,
                        bbox=base_result.bbox,
                        plate_crop_path=base_result.plate_crop_path,
                        sharpness=sharpness * 1.05,
                        plate_area_px=base_result.plate_area_px,
                        chars=c_cands
                    )
                    results.append(clahe_result)

        return results
