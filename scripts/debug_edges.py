import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tracex.detection import fallback_vehicle_box

img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
x1, y1, x2, y2 = [int(v) for v in fallback_vehicle_box(img)]
roi = img[y1:y2, x1:x2]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
for low, high in [(30, 90), (50, 150), (80, 200)]:
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 12:
            continue
        aspect = bw / max(1, bh)
        if 1.7 <= aspect <= 7 and bw * bh < 0.4 * roi.shape[0] * roi.shape[1]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            out.append((bx, by, bw, bh, round(aspect, 2), len(approx), round(cv2.contourArea(c) / max(1, bw * bh), 2)))
    print(low, high, len(out), sorted(out, key=lambda x: x[4])[:20])
