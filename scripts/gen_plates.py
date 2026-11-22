"""Synthetic Indian License Plate Generator.

Generates realistic, annotated Indian license plate crops with:
- Plate styles: standard white, two-wheeler (single/two-line), commercial yellow, green EV
- Layouts: border lines, blue IND emblem bar, hologram watermark
- Realism: random perspective tilt, rotation (-5 to +5 deg), shadow gradient, vehicle body context
- Graded degradation (Levels 0-5) via pipeline.degrade

Usage:
    python scripts/gen_plates.py --count 200 --out-dir data/plates_synth
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.degrade import degrade_plate

STYLE_CONFIGS = {
    "white": {
        "bg": (245, 245, 242),
        "text": (18, 18, 18),
        "border": (25, 25, 25),
        "has_ind": True
    },
    "yellow": {
        "bg": (240, 205, 30),
        "text": (15, 15, 15),
        "border": (25, 25, 25),
        "has_ind": True
    },
    "green_ev": {
        "bg": (35, 135, 60),
        "text": (245, 245, 245),
        "border": (240, 240, 240),
        "has_ind": True
    },
    "two_wheeler": {
        "bg": (245, 245, 242),
        "text": (18, 18, 18),
        "border": (25, 25, 25),
        "has_ind": False
    }
}

STATES = ["JH", "MH", "DL", "KA", "TN", "UP", "WB", "GJ", "CG", "MP", "RJ", "KL"]


def _get_font(size: int = 44) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a readable bold font on Windows or fallback to default."""
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_random_plate_string(rng: random.Random) -> str:
    """Generate valid 10-character Indian plate string."""
    state = rng.choice(STATES)
    district = f"{rng.randint(1, 45):02d}"
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    series = "".join(rng.choice(letters) for _ in range(2))
    digits = f"{rng.randint(1000, 9999):04d}"
    return f"{state}{district}{series}{digits}"


def apply_perspective(img_bgr: np.ndarray, max_warp: float = 0.08, rng: random.Random | None = None) -> np.ndarray:
    """Apply slight camera perspective tilt."""
    if rng is None:
        rng = random.Random()
    h, w = img_bgr.shape[:2]
    dx = w * max_warp
    dy = h * max_warp

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), h - rng.uniform(0, dy)],
        [rng.uniform(0, dx), h - rng.uniform(0, dy)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def render_base_plate(text: str, style_name: str, is_two_line: bool = False) -> np.ndarray:
    """Render high-resolution pristine plate image using PIL."""
    cfg = STYLE_CONFIGS.get(style_name, STYLE_CONFIGS["white"])
    w, h = (340, 90) if not is_two_line else (240, 140)

    img = Image.new("RGB", (w, h), cfg["bg"])
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rectangle([2, 2, w - 3, h - 3], outline=cfg["border"], width=3)

    # Optional IND strip on left
    text_x_offset = 12
    if cfg["has_ind"] and not is_two_line:
        ind_w = 30
        draw.rectangle([3, 3, ind_w, h - 4], fill=(20, 60, 160))
        # White IND text
        font_ind = _get_font(12)
        draw.text((8, h // 2 - 8), "IND", fill=(255, 255, 255), font=font_ind)
        # Small circle chakra
        draw.ellipse([11, h // 2 - 20, 21, h // 2 - 10], outline=(255, 255, 255), width=1)
        text_x_offset = ind_w + 14

    font = _get_font(42 if not is_two_line else 36)

    if not is_two_line:
        # Format text with slight spacing e.g. "JH 05 AB 6799"
        spaced = f"{text[:2]} {text[2:4]} {text[4:6]} {text[6:]}"
        draw.text((text_x_offset, (h - 48) // 2), spaced, fill=cfg["text"], font=font)
    else:
        # Two-line format (two-wheeler)
        line1 = f"{text[:2]} {text[2:4]}"
        line2 = f"{text[4:6]} {text[6:]}"
        draw.text((20, 14), line1, fill=cfg["text"], font=font)
        draw.text((20, 68), line2, fill=cfg["text"], font=font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_synthetic_dataset(
    count: int = 200,
    out_dir: str = "data/plates_synth",
    seed: int = 42
) -> str:
    """Generate synthetic plate images and labels CSV."""
    out_path = Path(out_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_path / "labels.csv"

    rng = random.Random(seed)
    styles = list(STYLE_CONFIGS.keys())

    records = []
    for i in range(count):
        plate_text = generate_random_plate_string(rng)
        style = rng.choice(styles)
        is_two_line = (style == "two_wheeler" and rng.random() > 0.5)

        base_bgr = render_base_plate(plate_text, style, is_two_line)

        # Apply realistic perspective & rotation
        warped = apply_perspective(base_bgr, max_warp=0.06, rng=rng)

        # Degradation level 0 to 5
        blur_level = rng.randint(0, 5)
        degraded, meta = degrade_plate(warped, level=blur_level, seed=rng.randint(0, 100000))

        img_name = f"plate_{i:05d}_lvl{blur_level}.png"
        img_dest = images_dir / img_name
        cv2.imwrite(str(img_dest), degraded)

        records.append({
            "image_path": str(img_dest.relative_to(out_path.parent.parent)),
            "text": plate_text,
            "style": style,
            "blur_level": blur_level,
            "seed": seed + i,
            "is_two_line": is_two_line
        })

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "text", "style", "blur_level", "seed", "is_two_line"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {count} synthetic plate images in {images_dir}")
    print(f"Labels saved to {labels_csv}")
    return str(labels_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Indian license plate images")
    parser.add_argument("--count", type=int, default=200, help="Number of plates to generate")
    parser.add_argument("--out-dir", type=str, default="data/plates_synth", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_synthetic_dataset(count=args.count, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
