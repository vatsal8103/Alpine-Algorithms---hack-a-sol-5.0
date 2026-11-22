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
for clahe in (False, True):
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if clahe else gray
    for low, high in [(30, 90), (40, 120), (50, 150), (80, 200)]:
        edges = cv2.Canny(g, low, high)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if 200 <= by <= 215 and 210 <= bx <= 240 and bw > 150:
                found.append((bx, by, bw, bh, round(cv2.contourArea(c)/(bw*bh),2)))
        print('clahe', clahe, low, high, found)
