import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from tracex.ocr import _segmentation_variants
from tracex.config import OCRConfig

img = cv2.imread('runs/smoke1/plates/frame_0006_plate.jpg')
for name, variant in _segmentation_variants(img):
    gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if variant.ndim == 3 else variant
    h, w = gray.shape
    target_h = 120
    scale = target_h / max(1, h)
    if scale > 1:
        gray = cv2.resize(gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    H, W = gray.shape
    print('\nvariant', name, 'size', W, H)
    for polarity in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, th = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bh < 0.34 * H or bh > 0.98 * H:
                continue
            if bw < max(3, 0.035 * H) or bw > 0.55 * H:
                continue
            if y < 0.02 * H or y + bh > 1.02 * H:
                continue
            boxes.append((x, y, bw, bh))
        print(' polarity', polarity, 'raw', len(boxes), sorted(boxes, key=lambda b: b[0]))
