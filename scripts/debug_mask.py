import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tracex.ocr import _segmentation_variants

img = cv2.imread('runs/smoke1/plates/frame_0006_plate.jpg')
up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('runs/smoke1/plate_up.png', up)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (int(img.shape[1] * 120 / img.shape[0]), 120), interpolation=cv2.INTER_CUBIC)
gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
cv2.imwrite('runs/smoke1/plate_mask.png', th)
proj = (th > 0).sum(axis=0)
print('width', th.shape[1])
print('projection')
print(' '.join(str(int(v)) for v in proj))
