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
row_frac = (mask > 0).mean(axis=1)
col_frac = (mask > 0).mean(axis=0)
print('row frac')
print(' '.join(f'{i}:{v:.2f}' for i, v in enumerate(row_frac) if v > 0.4))
print('col frac max', col_frac.max(), 'high cols', [(i, round(v,2)) for i,v in enumerate(col_frac) if v>0.8])
clean = mask.copy()
clean[row_frac > 0.62, :] = 0
clean[:, col_frac > 0.85] = 0
cv2.imwrite('runs/smoke1/mask_clean.png', clean)
proj = (clean > 0).sum(axis=0)
print('max proj', proj.max())
# print runs
active = proj > max(1, 0.045*H)
runs=[]; start=None
for x,f in enumerate(active):
    if f and start is None: start=x
    elif not f and start is not None: runs.append((start,x)); start=None
if start is not None: runs.append((start,len(active)))
print('runs', runs)
