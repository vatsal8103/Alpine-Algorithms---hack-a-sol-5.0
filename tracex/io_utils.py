"""Small image/IO helpers that work with non-ASCII Windows paths."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def imread(path: str | Path) -> np.ndarray | None:
    """Read an image, including paths containing spaces/Unicode on Windows."""

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path: str | Path, image: np.ndarray, quality: int = 90) -> bool:
    path = Path(path)
    ensure_dir(path.parent)
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def crop(image: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    if pad:
        pw = (x2 - x1) * pad
        ph = (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    x1, y1, x2, y2 = clip_box((x1, y1, x2, y2), w, h)
    return image[y1:y2, x1:x2].copy()


def resize_max(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def image_to_data_uri(path: str | Path, max_width: int = 640, quality: int = 82) -> str | None:
    image = imread(path)
    if image is None:
        return None
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def image_to_png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def relative_posix(path: str | Path, root: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def draw_label(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    text: str,
    color: tuple[int, int, int] = (0, 220, 120),
    thickness: int = 2,
) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.8, w / 1200))
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    ty = max(th + 6, y1 - 6)
    cv2.rectangle(image, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
    cv2.putText(image, text, (x1 + 4, ty - 4), font, scale, (10, 20, 15), 1, cv2.LINE_AA)


def to_base64_file(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def make_thumbnail(image: np.ndarray, max_width: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy()
    return cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def bgr_to_hex(color: tuple[int, int, int]) -> str:
    b, g, r = [max(0, min(255, int(v))) for v in color]
    return f"#{r:02x}{g:02x}{b:02x}"
