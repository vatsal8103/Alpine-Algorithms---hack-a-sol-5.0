import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

crop = cv2.imread('runs/smoke1/test_crop0.png')
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
gray = cv2.resize(gray, (int(w * 120 / h), 120), interpolation=cv2.INTER_CUBIC)
gray = cv2.copyMakeBorder(gray, 10, 10, 12, 12, cv2.BORDER_REPLICATE)
gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
H, W = mask.shape
band = mask[45:100, :]
proj = (band > 0).sum(axis=0)
print('max', proj.max())
print(' '.join(str(int(v)) for v in proj))
