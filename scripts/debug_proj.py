import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tracex.config import load_config
from tracex.detection import PlateDetector, fallback_vehicle_box
from tracex.ocr import _projection_boxes

cfg = load_config()
img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
det = PlateDetector(cfg.detector)
c = det.detect(img, fallback_vehicle_box(img))[0]
crop, _, _ = det.crop_and_refine(img, c.box)
cv2.imwrite('runs/smoke1/test_crop0.png', crop)
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
scale = 120 / h
gray = cv2.resize(gray, (int(w * scale), 120), interpolation=cv2.INTER_CUBIC)
gray = cv2.copyMakeBorder(gray, 10, 10, 12, 12, cv2.BORDER_REPLICATE)
gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
print('median', np.median(gray), 'shape', gray.shape)
for pol_name, polarity in [('inv', cv2.THRESH_BINARY_INV), ('bin', cv2.THRESH_BINARY)]:
    _, mask = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    cv2.imwrite(f'runs/smoke1/mask_{pol_name}.png', mask)
    print(pol_name, 'white frac', (mask > 0).mean())
print('boxes', _projection_boxes(gray))
