import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from tracex.config import load_config
from tracex.detection import PlateDetector, fallback_vehicle_box
from tracex.ocr import build_ocr, read_plate

cfg = load_config()
ocr = build_ocr(cfg.ocr)
img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
det = PlateDetector(cfg.detector)
for i, c in enumerate(det.detect(img, fallback_vehicle_box(img))[:4]):
    crop, refined, q = det.crop_and_refine(img, c.box)
    path = f'runs/smoke1/cand_{i}_crop.png'
    cv2.imwrite(path, cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    print(i, c.source, tuple(round(v) for v in c.box), 'crop', crop.shape, 'q', round(q, 3), path)
    best = read_plate(crop, ocr, cfg.ocr)
    print('  best', (best.text, round(best.confidence, 3), best.backend) if best else None)
