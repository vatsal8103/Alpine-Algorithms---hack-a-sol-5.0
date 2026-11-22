"""Frame extraction, Laplacian sharpness scoring, and video decodability validation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def compute_sharpness(image: np.ndarray) -> float:
    """Compute frame sharpness using variance of Laplacian.
    
    Higher variance corresponds to sharper edges and less motion/defocus blur.
    """
    if image is None or image.size == 0:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str,
    sampling_rate: int = 5,
    max_frames: int = 60,
    min_sharpness: float = 15.0
) -> list[tuple[int, float, np.ndarray, float]]:
    """Extract frames from video with timestamps and sharpness scores.
    
    Returns:
        List of tuples: (frame_no, timestamp_sec, frame_bgr, sharpness_score)
    """
    t0 = time.time()
    p_video = Path(video_path)
    if not p_video.exists():
        print(f"[frames] Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(p_video))
    if not cap.isOpened():
        print(f"[frames] Corrupted or unsupported video stream: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(fps / sampling_rate))) if sampling_rate > 0 else 1

    extracted: list[tuple[int, float, np.ndarray, float]] = []
    frame_idx = 0

    while cap.isOpened() and len(extracted) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            sharpness = compute_sharpness(frame)
            if sharpness >= min_sharpness or len(extracted) == 0:
                extracted.append((frame_idx, timestamp, frame, round(sharpness, 2)))

        frame_idx += 1

    cap.release()
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[frames] Extracted {len(extracted)} frames from {p_video.name} in {elapsed_ms:.1f}ms (FPS: {fps:.1f})")
    return extracted
