"""Graded image degradation pipeline (Levels 0-5) for license plates.

Degradation Levels:
    0: Clean plate (reference image)
    1: Mild Gaussian blur + subtle brightness shift
    2: Moderate motion blur (random angle) + slight downscale/upscale
    3: Strong motion blur + additive Gaussian sensor noise + JPEG compression (Q=50)
    4: Heavy motion blur + strong downscale/upscale + high sensor noise + glare gradient
    5: Severe motion blur + severe noise + low JPEG (Q=25) + partial character occlusion

Guarantees:
- Fully deterministic given a seed.
- Returns degraded image (np.ndarray BGR) and metadata dictionary.
"""

from __future__ import annotations

import math
import random
from typing import Any
import cv2
import numpy as np


def apply_motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Apply linear motion blur of specified length and angle."""
    if length <= 1:
        return image.copy()

    # Create motion blur kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    cx = (length - 1) / 2.0
    cy = (length - 1) / 2.0

    for step in np.linspace(-length / 2.0, length / 2.0, num=length * 2):
        x = int(round(cx + step * dx))
        y = int(round(cy + step * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] += 1.0

    k_sum = kernel.sum()
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[int(cy), int(cx)] = 1.0

    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_noise(image: np.ndarray, std: float, rng: random.Random) -> np.ndarray:
    """Add Gaussian sensor noise."""
    if std <= 0:
        return image.copy()
    np_rng = np.random.default_rng(rng.randint(0, 1000000))
    noise = np_rng.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Simulate CCTV transmission loss via JPEG compression."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 10, 100))]
    _, enc = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_downscale_upscale(image: np.ndarray, scale: float) -> np.ndarray:
    """Simulate low-resolution sensor downscale followed by digital zoom."""
    if scale >= 0.99:
        return image.copy()
    h, w = image.shape[:2]
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    down = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)
    return up


def apply_glare(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate headlight or streetlight lens flare."""
    h, w = image.shape[:2]
    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    radius = rng.randint(int(min(w, h) * 0.4), int(max(w, h) * 0.8))

    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    mask = np.clip(1.0 - np.sqrt(dist_sq) / max(1, radius), 0.0, 1.0)
    flare = (mask[:, :, np.newaxis] * 140.0).astype(np.float32)

    return np.clip(image.astype(np.float32) + flare, 0, 255).astype(np.uint8)


def apply_partial_occlusion(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate mud splatter or tow hook obscuring 1-2 characters."""
    img = image.copy()
    h, w = image.shape[:2]
    occ_w = rng.randint(int(w * 0.08), int(w * 0.18))
    occ_h = rng.randint(int(h * 0.40), int(h * 0.85))
    min_x = int(w * 0.15)
    max_x = max(min_x + 1, int(w * 0.80) - occ_w)
    x0 = rng.randint(min_x, max_x)

    min_y = int(h * 0.10)
    max_y = max(min_y + 1, int(h * 0.90) - occ_h)
    y0 = rng.randint(min_y, max_y)

    color = (rng.randint(30, 80), rng.randint(30, 80), rng.randint(30, 80)) # Mud/dirt tone
    cv2.ellipse(img, (x0 + occ_w // 2, y0 + occ_h // 2), (occ_w // 2, occ_h // 2),
                rng.randint(0, 180), 0, 360, color, -1)
    return img


def degrade_plate(
    image: np.ndarray,
    level: int = 0,
    seed: int = 42
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic degradation from Level 0 (clean) to Level 5 (severe).
    
    Returns:
        (degraded_image, metadata)
    """
    if level <= 0:
        return image.copy(), {"level": 0, "seed": seed, "operations": ["none"]}

    rng = random.Random(seed)
    img = image.copy()
    ops = []

    if level == 1:
        # Mild blur and subtle lighting shift
        ksize = 3
        img = cv2.GaussianBlur(img, (ksize, ksize), 0.8)
        ops.append("gaussian_blur_3x3")
        scale = 0.85
        img = apply_downscale_upscale(img, scale)
        ops.append(f"downscale_{scale}")

    elif level == 2:
        # Moderate motion blur + downscale
        angle = rng.uniform(0.0, 180.0)
        img = apply_motion_blur(img, length=5, angle_deg=angle)
        ops.append(f"motion_blur_5_deg{angle:.1f}")
        scale = 0.65
        img = apply_downscale_upscale(img, scale)
        ops.append(f"downscale_{scale}")
        img = apply_jpeg_compression(img, quality=75)
        ops.append("jpeg_q75")

    elif level == 3:
        # Strong motion blur + sensor noise + JPEG compression
        angle = rng.uniform(0.0, 180.0)
        img = apply_motion_blur(img, length=9, angle_deg=angle)
        ops.append(f"motion_blur_9_deg{angle:.1f}")
        img = apply_gaussian_noise(img, std=12.0, rng=rng)
        ops.append("gaussian_noise_std12")
        scale = 0.45
        img = apply_downscale_upscale(img, scale)
        ops.append(f"downscale_{scale}")
        img = apply_jpeg_compression(img, quality=50)
        ops.append("jpeg_q50")

    elif level == 4:
        # Heavy motion blur + low resolution + sensor noise + glare
        angle = rng.uniform(0.0, 180.0)
        img = apply_motion_blur(img, length=13, angle_deg=angle)
        ops.append(f"motion_blur_13_deg{angle:.1f}")
        img = apply_gaussian_noise(img, std=20.0, rng=rng)
        ops.append("gaussian_noise_std20")
        scale = 0.30
        img = apply_downscale_upscale(img, scale)
        ops.append(f"downscale_{scale}")
        img = apply_glare(img, rng)
        ops.append("headlight_glare")
        img = apply_jpeg_compression(img, quality=35)
        ops.append("jpeg_q35")

    elif level >= 5:
        # Severe blur + severe noise + low JPEG + partial character occlusion
        angle = rng.uniform(0.0, 180.0)
        img = apply_motion_blur(img, length=17, angle_deg=angle)
        ops.append(f"motion_blur_17_deg{angle:.1f}")
        img = apply_gaussian_noise(img, std=28.0, rng=rng)
        ops.append("gaussian_noise_std28")
        scale = 0.22
        img = apply_downscale_upscale(img, scale)
        ops.append(f"downscale_{scale}")
        img = apply_partial_occlusion(img, rng)
        ops.append("partial_character_occlusion")
        img = apply_jpeg_compression(img, quality=25)
        ops.append("jpeg_q25")

    meta = {
        "level": level,
        "seed": seed,
        "operations": ops,
        "original_shape": image.shape,
        "degraded_shape": img.shape
    }
    return img, meta
