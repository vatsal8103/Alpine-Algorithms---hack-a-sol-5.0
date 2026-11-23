# TRACE-X Model Weights Directory

This directory stores fine-tuned and pretrained checkpoint weights for the TRACE-X forensic engine:

| Weight File | Architecture | Purpose | Size | Status | How to Obtain / Train |
|---|---|---|---|---|---|
| `cct-xs-v2-global-model.onnx` | Compact Conv-Transformer (CCT-XS) | Open-source fast plate recognition | 1.2 MB | **Cached / ONNX** (Validated 68.1% char acc, 2.2ms latency) | Auto-downloaded via `fast-plate-ocr` or `pip install fast-plate-ocr` |
| `models/ocr_finetuned.pth` | Mobile CRNN (CNN + BiGRU + CTC) | License plate sequence recognition | 1.46 MB | **Available on Disk** (Validated 85.1% full plate exact match) | Train via: `python scripts/train_ocr.py --epochs 30` |
| `yolov8s.pt` (root) | YOLOv8s | Vehicle detection & localization | 21.54 MB | **Available on Disk** | Download via `ultralytics` COCO pretrained release |
| `models/plate_yolo11n.pt` / `models/plate_yolov8s.pt` | Ultralytics YOLO11n | SOTA Indian license plate detector (standard + heavy commercial trucks) | 5.25 MB | **Active Champion** (99.3% mAP50, 1.4ms latency) | Train via: `python scripts/train_plate_yolo11.py --model yolo11n.pt --epochs 50 --save-period 5` |
| `models/checkpoints/` | Directory | Periodic checkpoint snapshots (`*epoch*.pt`, `*best.pt`, `*last.pt`) | Variable | **Active** | Auto-saved during training or resume via `--resume` / `--checkpoint` |
| `configs/ocr_calibration.json` | JSON Calibration | Temperature scaling parameter ($T=0.50$) | ~1 KB | **Available on Disk** | Calibrate via: `python scripts/calibrate_ocr.py` |

### Note on Version Control:
In compliance with project specifications, all binary model weight files (`*.pt`, `*.pth`, `*.onnx`) are strictly excluded from git tracking via `.gitignore`.
