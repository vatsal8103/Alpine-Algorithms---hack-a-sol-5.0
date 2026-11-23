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
    Uses downscaled ROI and CV_32F for 10x faster execution.
    """
    if image is None or image.size == 0:
        return 0.0
    try:
        h, w = image.shape[:2]
        if w > 640:
            scale = 640.0 / w
            small = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        else:
            small = image
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        return 0.0


def extract_frames(
    video_path: str,
    sampling_rate: int = 5,
    max_frames: int = 60,
    min_sharpness: float = 15.0
) -> list[tuple[int, float, np.ndarray, float]]:
    """Extract frames from video with timestamps and sharpness scores.
    
    Skips unneeded intermediate frames using cap.grab() for accelerated decoding.
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
    sample_indices = None
    if max_frames > 0 and total_video_frames > max_frames * frame_interval:
        sample_indices = set(np.linspace(0, total_video_frames - 1, max_frames, dtype=int))

    extracted: list[tuple[int, float, np.ndarray, float]] = []
    frame_idx = 0

    while cap.isOpened() and len(extracted) < max_frames:
        is_sample = (frame_idx in sample_indices if sample_indices is not None else frame_idx % frame_interval == 0)
        if not is_sample:
            if not cap.grab():
                break
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        sharpness = compute_sharpness(frame)
        if sharpness >= min_sharpness or len(extracted) == 0:
            extracted.append((frame_idx, timestamp, frame, round(sharpness, 2)))

        frame_idx += 1

    cap.release()
    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[frames] Extracted {len(extracted)} frames from {p_video.name} in {elapsed_ms:.1f}ms (FPS: {fps:.1f})")
    return extracted
