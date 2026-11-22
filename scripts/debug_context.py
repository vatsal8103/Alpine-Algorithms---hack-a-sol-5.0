import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
img = cv2.imread('runs/smoke1/frames/frame_0006.jpg')
crop = img[285:410, 200:500]
cv2.imwrite('runs/smoke1/plate_context.png', cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
