import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tracex.detection import fallback_vehicle_box, _rectangle_candidates

img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
x1, y1, x2, y2 = [int(v) for v in fallback_vehicle_box(img)]
roi = img[y1:y2, x1:x2]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
cands = _rectangle_candidates(gray, roi.shape[1], roi.shape[0], 1.0, x1, y1)
for c in sorted(cands, key=lambda c: c.score, reverse=True)[:30]:
    print(round(c.score, 3), tuple(round(v) for v in c.box), c.source)
