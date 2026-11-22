"""Video / image-sequence sampling.

A clip is never split by frame for evaluation.  This module only *samples*
frames for inference; the clip/vehicle boundary is preserved by the caller
and by ``scripts/split_dataset.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .io_utils import imread

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    source_path: str | None = None


@dataclass
class FrameSet:
    frames: list[Frame] = field(default_factory=list)
    fps: float = 0.0
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    kind: str = "video"
    total_frames: int = 0
    source_path: str = ""

    def __len__(self) -> int:
        return len(self.frames)


def _video_frames(path: Path, max_frames: int, stride: int) -> FrameSet:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total > 0:
        step = max(int(stride), int(math.ceil(total / max_frames)))
    else:
        step = max(1, int(stride))

    frames: list[Frame] = []
    idx = 0
    while len(frames) < max_frames:
        ok, image = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if width <= 0 or height <= 0:
                height, width = image.shape[:2]
            frames.append(
                Frame(
                    index=idx,
                    timestamp=idx / fps if fps else 0.0,
                    image=image,
                    source_path=str(path),
                )
            )
        idx += 1
        # If the frame count is unknown, stop after a bounded read.
        if total <= 0 and idx > max_frames * step * 2:
            break
    cap.release()

    duration = (total / fps) if total > 0 and fps > 0 else (idx / fps if fps > 0 else 0.0)
    return FrameSet(
        frames=frames,
        fps=fps,
        duration_s=float(duration),
        width=width,
        height=height,
        kind="video",
        total_frames=total or idx,
        source_path=str(path),
    )


def _directory_frames(path: Path, max_frames: int, stride: int) -> FrameSet:
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No image frames found in directory: {path}")
    if len(paths) > max_frames:
        step = max(int(stride), int(math.ceil(len(paths) / max_frames)))
        paths = paths[::step][:max_frames]
    frames: list[Frame] = []
    for i, p in enumerate(paths):
        image = imread(p)
        if image is None:
            continue
        frames.append(Frame(index=i, timestamp=float(i), image=image, source_path=str(p)))
    if not frames:
        raise ValueError(f"Could not read any frames from: {path}")
    h, w = frames[0].image.shape[:2]
    return FrameSet(
        frames=frames,
        fps=1.0,
        duration_s=float(len(frames)),
        width=w,
        height=h,
        kind="frames",
        total_frames=len(paths),
        source_path=str(path),
    )


def sample_frames(source: str | Path, max_frames: int = 36, stride: int = 1) -> FrameSet:
    """Read at most ``max_frames`` frames from a video, image or frame folder."""

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    max_frames = max(1, int(max_frames))
    stride = max(1, int(stride))

    if path.is_dir():
        return _directory_frames(path, max_frames, stride)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image = imread(path)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        return FrameSet(
            frames=[Frame(index=0, timestamp=0.0, image=image, source_path=str(path))],
            fps=1.0,
            duration_s=0.0,
            width=w,
            height=h,
            kind="image",
            total_frames=1,
            source_path=str(path),
        )
    return _video_frames(path, max_frames, stride)


def frame_quality(image: np.ndarray, box: tuple[float, float, float, float] | None = None) -> dict[str, float]:
    """Sharpness / contrast / brightness metrics used to weight evidence."""

    if box is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        image = image[y1:y2, x1:x2]
    if image.size == 0:
        return {"sharpness": 0.0, "contrast": 0.0, "brightness": 0.0, "quality": 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    sharp_norm = min(1.0, sharpness / 350.0)
    contrast_norm = min(1.0, contrast / 60.0)
    bright_norm = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
    quality = 0.5 * sharp_norm + 0.3 * contrast_norm + 0.2 * bright_norm
    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "quality": max(0.0, min(1.0, quality)),
    }
