"""Vehicle detection and ByteTrack tracking module."""

from __future__ import annotations

import time
from typing import Any
import cv2
import numpy as np
import torch


class VehicleTracker:
    """Tracks vehicles across video frames using YOLOv8 and ByteTrack (GPU Accelerated)."""

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        conf: float = 0.10,
        iou: float = 0.45,
        device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Could not load vehicle detector {self.model_name}: {exc}") from exc

    def track(
        self,
        frames: list[tuple[int, float, np.ndarray, float]]
    ) -> dict[int, list[dict[str, Any]]]:
        """Track vehicles across extracted frames.
        
        Returns:
            Dictionary mapping track_id to list of frame observations:
            {track_id: [{"frame_no": ..., "timestamp": ..., "bbox": [x1, y1, x2, y2], "conf": ...}]}
        """
        t0 = time.time()
        tracks: dict[int, list[dict[str, Any]]] = {}

        for fno, ts, img, sharp in frames:
            results = self._model.track(
                source=img,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[1, 2, 3, 5, 7],  # COCO bicycle, car, motorcycle, bus, truck
                imgsz=960 if max(img.shape[:2]) >= 1920 else 640,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    tid = int(box.id[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0]) if box.cls is not None else 2
                    tracks.setdefault(tid, []).append({
                        "frame_no": fno,
                        "timestamp": ts,
                        "bbox": [round(c, 1) for c in xyxy],
                        "conf": round(conf, 3),
                        "cls": cls_id,
                        "sharpness": sharp,
                        "frame_image": img
                    })

        elapsed_ms = (time.time() - t0) * 1000.0
        print(f"[detect_track] Tracked {len(tracks)} vehicle track(s) across {len(frames)} frames in {elapsed_ms:.1f}ms")
        return tracks
