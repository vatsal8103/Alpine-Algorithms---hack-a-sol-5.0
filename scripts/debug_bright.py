import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tracex.detection import fallback_vehicle_box

img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
x1, y1, x2, y2 = [int(v) for v in fallback_vehicle_box(img)]
roi = img[y1:y2, x1:x2]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
print('shape', roi.shape, 'v percentiles', np.percentile(v, [50, 75, 90, 95, 99]))
white = ((v > max(135, np.percentile(v, 75))) & (s < 80)).astype(np.uint8) * 255
yellow = ((h >= 15) & (h <= 40) & (s >= 70) & (v >= 100)).astype(np.uint8) * 255
green = ((h >= 35) & (h <= 95) & (s >= 45) & (v >= 60)).astype(np.uint8) * 255
for name, mask in [('white', white), ('yellow', yellow), ('green', green)]:
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(1, bh)
        if 1.6 <= aspect <= 8 and bw >= 50 and bh >= 15 and bw * bh < 0.5 * roi.shape[0] * roi.shape[1]:
            patch = cv2.cvtColor(roi[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2GRAY)
            if patch.size == 0: continue
            edges = cv2.Canny(patch, 50, 150)
            edge_density = float((edges > 0).mean())
            out.append((bx, by, bw, bh, round(aspect,2), round(edge_density,3)))
    print(name, len(out), out)
    cv2.imwrite(f'runs/smoke1/mask_{name}.png', m)
