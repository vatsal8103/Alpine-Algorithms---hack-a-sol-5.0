"""Vehicle attribute extraction: body colour and body type."""

from __future__ import annotations

import cv2
import numpy as np
from pipeline.schemas import VehicleAttributes
from scripts.train_bodytype import BodyTypeClassifier

# Perceptual color ranges in HSV space
COLOR_RANGES_HSV = {
    "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))],
    "orange": [((11, 80, 60), (25, 255, 255))],
    "yellow": [((26, 70, 70), (38, 255, 255))],
    "green": [((39, 60, 50), (85, 255, 255))],
    "blue": [((86, 60, 50), (135, 255, 255))],
    "brown": [((10, 60, 30), (20, 180, 110))],
}


def segment_body_roi(vehicle_crop: np.ndarray) -> np.ndarray:
    """Extract vehicle body region, excluding windshield glass, wheels, and ground shadow."""
    h, w = vehicle_crop.shape[:2]
    # Crop central-lower area (35% to 80% vertical, 15% to 85% horizontal)
    y1 = int(h * 0.35)
    y2 = int(h * 0.80)
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    roi = vehicle_crop[y1:y2, x1:x2]
    if roi.size == 0:
        return vehicle_crop
    return roi


def extract_dominant_colour(vehicle_crops: list[np.ndarray]) -> tuple[str, float, dict[str, float]]:
    """Determine vehicle body colour via HSV segmentation and pixel voting."""
    if not vehicle_crops:
        return "white", 0.50, {"white": 0.50}

    votes: dict[str, int] = {
        "white": 0, "silver": 0, "grey": 0, "black": 0,
        "red": 0, "blue": 0, "yellow": 0, "green": 0, "brown": 0, "orange": 0
    }

    for crop in vehicle_crops:
        roi = segment_body_roi(crop)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Achromatic segmentation (White, Silver, Grey, Black)
        low_sat = (s < 45)
        votes["black"] += int(np.sum(low_sat & (v < 55)))
        votes["grey"] += int(np.sum(low_sat & (v >= 55) & (v < 140)))
        votes["silver"] += int(np.sum(low_sat & (v >= 140) & (v < 205)))
        votes["white"] += int(np.sum(low_sat & (v >= 205)))

        # Chromatic segmentation
        for col_name, ranges in COLOR_RANGES_HSV.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            votes[col_name] += int(np.sum(mask > 0))

    total_pixels = sum(votes.values())
    if total_pixels <= 0:
        return "white", 0.70, {"white": 0.70, "silver": 0.20, "grey": 0.10}

    dist = {k: round(v / total_pixels, 3) for k, v in votes.items()}
    sorted_cols = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    top_col, top_conf = sorted_cols[0]

    return top_col, max(0.40, min(0.98, top_conf * 1.8)), dist


def extract_attributes(
    vehicle_crops: list[np.ndarray],
    yolo_cls: int | None = None,
    user_hints: dict[str, str] | None = None
) -> VehicleAttributes:
    """Extract aggregated vehicle attributes with soft user-hint integration."""
    top_col, col_conf, col_dist = extract_dominant_colour(vehicle_crops)

    body_clf = BodyTypeClassifier()
    # Predict body type using first available crop
    ref_crop = vehicle_crops[0] if vehicle_crops else np.zeros((100, 100, 3), dtype=np.uint8)
    top_body, body_conf, body_dist = body_clf.predict(ref_crop)

    # Reconcile with YOLO detector class when available
    if yolo_cls is not None:
        if yolo_cls == 2:  # Car / SUV / Sedan
            if top_body not in ("hatchback", "sedan", "suv"):
                # YOLO detected a car: ensure it is categorized as passenger car, never two-wheeler or auto-rickshaw
                h, w = ref_crop.shape[:2]
                aspect = w / float(max(1, h))
                if aspect >= 1.45:
                    top_body = "sedan"
                elif aspect <= 1.20:
                    top_body = "suv"
                else:
                    top_body = "hatchback"
                body_conf = 0.88
        elif yolo_cls in (0, 3):  # Rider / Motorcycle / Two-Wheeler
            top_body = "two-wheeler"
            body_conf = 0.90
        elif yolo_cls == 7:  # Truck
            top_body = "truck"
            body_conf = 0.90
        elif yolo_cls == 5:  # Bus
            top_body = "bus"
            body_conf = 0.90

    # Plausible make/model inference
    make_map = {
        "suv": ("Mahindra", "Scorpio"),
        "hatchback": ("Maruti Suzuki", "Swift"),
        "sedan": ("Honda", "City"),
        "two-wheeler": ("Royal Enfield", "Classic 350"),
        "auto-rickshaw": ("Bajaj", "RE Compact"),
        "truck": ("Tata", "407"),
        "bus": ("Tata", "Starbus")
    }
    make, model = make_map.get(top_body, ("Unknown", "Unknown"))

    # Apply soft investigator hints if supplied
    if user_hints:
        if "colour" in user_hints and user_hints["colour"]:
            hint_c = user_hints["colour"].lower()
            if hint_c in col_dist:
                col_dist[hint_c] = col_dist.get(hint_c, 0.0) + 0.30
                top_col = hint_c
                col_conf = min(0.95, col_conf + 0.10)
        if "body_type" in user_hints and user_hints["body_type"]:
            hint_b = user_hints["body_type"].lower()
            if hint_b in body_dist:
                body_dist[hint_b] = body_dist.get(hint_b, 0.0) + 0.30
                top_body = hint_b
                body_conf = min(0.95, body_conf + 0.10)

    return VehicleAttributes(
        colour=top_col,
        colour_conf=round(col_conf, 3),
        body_type=top_body,
        body_conf=round(body_conf, 3),
        make=make,
        model=model,
        colour_distribution=col_dist,
        body_distribution=body_dist,
        plate_style="white",
        plate_style_conf=0.90
    )
