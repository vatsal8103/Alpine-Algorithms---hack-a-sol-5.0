import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tracex.ocr import _projection_boxes, _TemplateBank

crop = cv2.imread('runs/smoke1/test_crop0.png')
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
scale = 120 / h
gray = cv2.resize(gray, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)
gray = cv2.copyMakeBorder(gray, 10, 10, 12, 12, cv2.BORDER_REPLICATE)
gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
boxes = _projection_boxes(gray)
bank = _TemplateBank.get()
for i, (x, y, bw, bh) in enumerate(boxes):
    patch = gray[y:y+bh, x:x+bw]
    cv2.imwrite(f'runs/smoke1/char_{i}.png', cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    print(i, (x,y,bw,bh), bank.match(patch, top_k=5))
