"""License plate localization, margin crop, and per-frame OCR inference."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from pipeline.ocr import LicensePlateOCR
from pipeline.schemas import CharAlternative, CharCandidate, FrameOCRResult


class PlateStage:
    """Localizes license plates within vehicle bounding boxes and runs per-frame OCR."""

    def __init__(
        self,
        plate_model_path: str = "models/plate_yolov8s.pt",
        char_threshold: float = 0.28,
        crops_dir: str = "runs/crops"
    ) -> None:
        self.char_threshold = char_threshold
        self.crops_dir = Path(crops_dir)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.ocr_engine = LicensePlateOCR(backend="auto")

    def locate_plate_box(self, vehicle_crop: np.ndarray) -> tuple[int, int, int, int]:
        """Locate license plate bounding box [x, y, w, h] within vehicle crop."""
        vh, vw = vehicle_crop.shape[:2]

        # Region of interest: lower 60% of vehicle, centered
        roi_y1 = int(vh * 0.40)
        roi_y2 = int(vh * 0.90)
        roi_x1 = int(vw * 0.20)
        roi_x2 = int(vw * 0.80)
        roi = vehicle_crop[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return 0, 0, vw, vh

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Morphological gradient to highlight high-contrast plate boundaries
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_aspect_score = -1.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(w) / max(1, h)
            area = w * h
            # Indian plates typically have aspect ratio between 2.2 and 5.0
            if 2.0 <= aspect <= 5.5 and area > (roi.shape[0] * roi.shape[1] * 0.03):
                score = area * (1.0 - abs(aspect - 3.8) * 0.2)
                if score > best_aspect_score:
                    best_aspect_score = score
                    best_box = (roi_x1 + x, roi_y1 + y, w, h)

        if best_box is not None:
            return best_box

        # Fallback default plate region: centered in lower third
        pw = int(vw * 0.42)
        ph = int(vh * 0.15)
        px = (vw - pw) // 2
        py = int(vh * 0.65)
        return px, py, pw, ph

    def process_frame_crop(
        self,
        vehicle_crop: np.ndarray,
        track_id: int,
        frame_no: int,
        timestamp: float,
        sharpness: float
    ) -> FrameOCRResult:
        """Crop plate with safety margin, run OCR, and persist crop image."""
        vh, vw = vehicle_crop.shape[:2]
        px, py, pw, ph = self.locate_plate_box(vehicle_crop)

        # 10% safety margin expansion
        mx = int(pw * 0.10)
        my = int(ph * 0.10)
        x1 = max(0, px - mx)
        y1 = max(0, py - my)
        x2 = min(vw, px + pw + mx)
        y2 = min(vh, py + ph + my)

        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            plate_crop = vehicle_crop

        crop_fname = f"track_{track_id}_f{frame_no:04d}_plate.png"
        crop_path = self.crops_dir / crop_fname
        cv2.imwrite(str(crop_path), plate_crop)

        # Run OCR
        text, candidates = self.ocr_engine.recognize(plate_crop)

        plate_area = float((x2 - x1) * (y2 - y1))

        return FrameOCRResult(
            track_id=track_id,
            frame_no=frame_no,
            timestamp=timestamp,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            plate_crop_path=str(crop_path),
            sharpness=sharpness,
            plate_area_px=plate_area,
            chars=candidates
        )
