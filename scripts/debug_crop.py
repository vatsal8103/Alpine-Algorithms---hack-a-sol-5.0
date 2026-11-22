import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from tracex.config import load_config
from tracex.detection import PlateDetector, fallback_vehicle_box
from tracex.io_utils import crop

cfg = load_config()
img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
det = PlateDetector(cfg.detector)
box = fallback_vehicle_box(img)
cands = det.detect(img, box)
print('vehicle', box)
for i, c in enumerate(cands):
    print(i, c)
    raw = crop(img, c.box, pad=0.0)
    cv2.imwrite(f'runs/smoke1/cand_{i}_raw.png', cv2.resize(raw, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    plate, refined, q = det.crop_and_refine(img, c.box)
    print(' refined', refined, 'q', q, 'shape', plate.shape)
    cv2.imwrite(f'runs/smoke1/cand_{i}_ref.png', cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
