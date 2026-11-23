"""Frame quality ranking, best-N frame selection, and near-duplicate suppression."""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter / float(area1 + area2 - inter)


def compute_plate_saliency(frame_image: np.ndarray | None, bbox: list[float]) -> float:
    """Evaluate whether the license plate / registration board is visible on this side of the vehicle.
    
    Penalizes side-profile crops (e.g. doors, side windows) lacking a front/rear plate.
    Boosts frames containing high-contrast white/yellow rectangular plate contours with edge transitions.
    """
    if frame_image is None or not hasattr(frame_image, "shape") or frame_image.size == 0:
        return 1.0
    try:
        import cv2
        vh, vw = frame_image.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(vw, int(bbox[2])), min(vh, int(bbox[3]))
        crop = frame_image[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 30 or cw < 30:
            return 1.0

        # Lower 65% of the vehicle where front/rear plates are mounted
        roi = crop[int(ch * 0.35):, :]
        rh, rw = roi.shape[:2]
        if rh < 15 or rw < 15:
            return 1.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Horizontal character edge density (text has high dX / dY variance)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        edge_density = float(np.mean(np.abs(sobel_x)))

        # 2. White/Yellow plate blob detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]
        hue = hsv[:, :, 0]
        
        white_mask = (val > 150) & (sat < 70)
        yellow_mask = (val > 150) & (sat > 70) & (hue >= 15) & (hue <= 35)
        plate_color_mask = ((white_mask | yellow_mask).astype(np.uint8)) * 255

        contours, _ = cv2.findContours(plate_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_plate_blob = 0.0
        roi_area = rh * rw
        for cnt in contours:
            cx, cy, cw_b, ch_b = cv2.boundingRect(cnt)
            if ch_b < 6 or cw_b < 16:
                continue
            aspect = cw_b / float(ch_b)
            area = cw_b * ch_b
            if 1.8 <= aspect <= 5.8 and (0.008 * roi_area <= area <= 0.40 * roi_area):
                cnt_gray = gray[cy:cy+ch_b, cx:cx+cw_b]
                if cnt_gray.size > 0:
                    contrast = float(np.std(cnt_gray))
                    score = np.sqrt(area) * contrast * (1.0 - abs(aspect - 3.5) * 0.12)
                    if score > best_plate_blob:
                        best_plate_blob = score

        crop_aspect = cw / float(ch)
        side_penalty = 0.3 if (crop_aspect > 1.8 or crop_aspect < 0.5) and best_plate_blob < 10.0 else 1.0
        saliency_multiplier = side_penalty * (1.0 + (best_plate_blob * 0.05) + (edge_density * 0.08))
        return float(max(0.1, saliency_multiplier))
    except Exception:
        return 1.0


def compute_plate_sharpness(frame_image: np.ndarray | None, bbox: list[float]) -> float:
    """Compute variance of Laplacian specifically on the lower vehicle plate ROI."""
    if frame_image is None or not hasattr(frame_image, "shape") or frame_image.size == 0:
        return 1.0
    try:
        import cv2
        vh, vw = frame_image.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(vw, int(bbox[2])), min(vh, int(bbox[3]))
        crop = frame_image[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 15 or cw < 15:
            return 1.0
        roi = crop[int(ch * 0.35):, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(max(1.0, lap.var()))
    except Exception:
        return 1.0


def select_best_frames(
    observations: list[dict[str, Any]],
    best_n: int = 12,
    dedup_iou: float = 0.85
) -> list[dict[str, Any]]:
    """Select top-N frames based on composite quality (plate sharpness * area * conf * plate_saliency).
    
    Prioritizes vehicle viewpoints where the front or rear number board is directly visible,
    penalizing side angles (doors/windows). Suppresses near-identical consecutive frames.
    """
    if not observations:
        return []

    def quality_score(item: dict[str, Any]) -> float:
        bbox = item.get("bbox", [0, 0, 100, 100])
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        conf = float(item.get("conf", 0.8))
        f_img = item.get("frame_image")
        plate_sharpness = compute_plate_sharpness(f_img, bbox)
        plate_saliency = compute_plate_saliency(f_img, bbox) if f_img is not None else 1.0
        item["plate_sharpness"] = plate_sharpness
        # Emphasize large close-up vehicle area to ensure high-resolution plate crops
        return float(np.sqrt(plate_sharpness) * (area ** 0.85) * conf * plate_saliency)

    # Sort all candidate observations by quality descending
    sorted_obs = sorted(observations, key=quality_score, reverse=True)

    selected: list[dict[str, Any]] = []
    for obs in sorted_obs:
        if len(selected) >= best_n:
            break
        # Check IoU and frame distance against already selected frames
        is_duplicate = False
        for s in selected:
            # If same frame number, skip
            if obs["frame_no"] == s["frame_no"]:
                is_duplicate = True
                break
            # If consecutive frame and high bbox overlap, suppress
            if abs(obs["frame_no"] - s["frame_no"]) <= 1 and compute_iou(obs["bbox"], s["bbox"]) > dedup_iou:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(obs)

    # Fallback if filtering was too aggressive
    if len(selected) < min(3, len(observations)):
        selected = sorted_obs[:best_n]

    # Re-sort chronologically by frame_no
    selected.sort(key=lambda x: x["frame_no"])
    return selected
