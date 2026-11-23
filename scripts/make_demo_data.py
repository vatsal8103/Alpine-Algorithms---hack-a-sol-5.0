"""Generate synthetic CCTV-like demo clips and labels.

The generator is deliberately independent of the OCR template bank: plates
are rendered with a bold font, then blurred, downscaled, noised and partly
occluded.  This gives a reproducible smoke test for the whole pipeline
without shipping real CCTV footage.

Run:
    python scripts/make_demo_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracex.io_utils import ensure_dir, imwrite
from tracex.registry import load_registry
from tracex.schemas import RegistryEntry
from tracex.utils import seed_everything

W, H = 720, 480
FPS = 12
N_FRAMES = 22

COLOR_BGR = {
    "white": (238, 238, 238),
    "silver": (180, 184, 188),
    "black": (34, 36, 38),
    "red": (38, 42, 200),
    "blue": (190, 95, 40),
    "grey": (115, 118, 120),
    "green": (70, 150, 70),
    "yellow": (40, 205, 235),
    "orange": (30, 135, 235),
    "brown": (45, 85, 120),
    "maroon": (40, 40, 120),
}

PLATE_BG = {
    "white": (244, 244, 240),
    "yellow": (30, 205, 235),
    "green": (90, 190, 110),
}


def _font_path() -> str:
    candidates = [
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""


def render_plate(text: str, style: str, width: int, height: int) -> np.ndarray:
    bg = PLATE_BG.get(style, PLATE_BG["white"])
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    border = max(2, width // 90)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(20, 20, 20), width=border)
    font_path = _font_path()
    font_size = max(16, int(height * 0.60))
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # Letter-spaced text, centred.
    spacing = max(1, int(font_size * 0.06))
    widths = []
    for ch in text:
        try:
            widths.append(float(draw.textlength(ch, font=font)))
        except Exception:
            widths.append(font_size * 0.55)
    total = sum(widths) + spacing * max(0, len(text) - 1)
    x = (width - total) / 2
    y = height / 2
    for ch, cw in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=(12, 12, 12), anchor="lm")
        x += cw + spacing
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _rounded_rect(image, p1, p2, color, radius=12, thickness=-1):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(image, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(image, (cx, cy), r, color, thickness)


def _draw_background(canvas: np.ndarray, variant: str, rng: random.Random) -> None:
    if variant == "night":
        sky_top, sky_bottom = (28, 22, 18), (80, 58, 42)
        road = (38, 38, 40)
    elif variant == "highway":
        sky_top, sky_bottom = (205, 170, 120), (235, 220, 185)
        road = (78, 80, 82)
    else:
        sky_top, sky_bottom = (150, 120, 95), (215, 205, 190)
        road = (72, 74, 76)
    for y in range(int(H * 0.48)):
        t = y / max(1, int(H * 0.48))
        color = tuple(int(sky_top[i] * (1 - t) + sky_bottom[i] * t) for i in range(3))
        cv2.line(canvas, (0, y), (W, y), color, 1)
    cv2.rectangle(canvas, (0, int(H * 0.48)), (W, H), road, -1)
    # Buildings / barriers.
    x = -20
    while x < W + 40:
        bw = rng.randint(45, 110)
        bh = rng.randint(50, 150)
        base = int(H * 0.48)
        color = (rng.randint(55, 95), rng.randint(55, 90), rng.randint(55, 95))
        cv2.rectangle(canvas, (x, base - bh), (x + bw, base), color, -1)
        for wy in range(base - bh + 12, base - 8, 22):
            for wx in range(x + 8, x + bw - 8, 20):
                if rng.random() < 0.35:
                    cv2.rectangle(canvas, (wx, wy), (wx + 8, wy + 10), (150, 170, 120), -1)
        x += bw + rng.randint(6, 22)
    # Road markings.
    for offset in (-160, 0, 160):
        cv2.line(canvas, (W // 2 + offset, int(H * 0.52)), (W // 2 + offset * 2, H), (220, 220, 210), 4)
    cv2.line(canvas, (0, int(H * 0.48)), (W, int(H * 0.48)), (45, 45, 45), 3)


def _draw_vehicle(canvas: np.ndarray, entry: RegistryEntry, cx: int, base_y: int, scale: float) -> tuple[int, int, int, int, tuple[int, int, int]]:
    color = COLOR_BGR.get(entry.color, (120, 120, 120))
    body = entry.body_type
    if body in ("car", "hatchback", "sedan", "suv", "van"):
        w = int((300 if body != "suv" else 335) * scale)
        h = int((150 if body != "suv" else 195) * scale)
        x1, y1 = cx - w // 2, base_y - h
        x2, y2 = cx + w // 2, base_y
        # Shadow.
        cv2.ellipse(canvas, (cx, base_y + 4), (int(w * 0.52), int(h * 0.10)), 0, 0, 360, (35, 35, 35), -1)
        # Wheels.
        for wx in (x1 + int(w * 0.16), x2 - int(w * 0.16)):
            cv2.circle(canvas, (wx, base_y - int(h * 0.08)), int(h * 0.15), (25, 25, 25), -1)
            cv2.circle(canvas, (wx, base_y - int(h * 0.08)), int(h * 0.07), (120, 120, 120), -1)
        # Body + roof.
        _rounded_rect(canvas, (x1, y1 + int(h * 0.30)), (x2, y2 - int(h * 0.06)), color, radius=int(16 * scale))
        _rounded_rect(canvas, (x1 + int(w * 0.16), y1), (x2 - int(w * 0.16), y1 + int(h * 0.42)), color, radius=int(18 * scale))
        # Rear window.
        wx1, wy1 = x1 + int(w * 0.23), y1 + int(h * 0.07)
        wx2, wy2 = x2 - int(w * 0.23), y1 + int(h * 0.31)
        cv2.rectangle(canvas, (wx1, wy1), (wx2, wy2), (55, 70, 78), -1)
        # Taillights.
        cv2.rectangle(canvas, (x1 + int(w * 0.06), y1 + int(h * 0.42)), (x1 + int(w * 0.16), y1 + int(h * 0.50)), (30, 30, 190), -1)
        cv2.rectangle(canvas, (x2 - int(w * 0.16), y1 + int(h * 0.42)), (x2 - int(w * 0.06), y1 + int(h * 0.50)), (30, 30, 190), -1)
        plate_center = (cx, y1 + int(h * 0.66))
    elif body == "two-wheeler":
        w = int(150 * scale)
        h = int(230 * scale)
        x1, y1 = cx - w // 2, base_y - h
        x2, y2 = cx + w // 2, base_y
        cv2.ellipse(canvas, (cx, base_y + 4), (int(w * 0.55), int(h * 0.05)), 0, 0, 360, (35, 35, 35), -1)
        cv2.circle(canvas, (cx, base_y - int(h * 0.08)), int(h * 0.08), (25, 25, 25), -1)
        cv2.rectangle(canvas, (cx - int(w * 0.18), y1 + int(h * 0.25)), (cx + int(w * 0.18), y2 - int(h * 0.16)), color, -1)
        cv2.circle(canvas, (cx, y1 + int(h * 0.20)), int(w * 0.30), (40, 45, 50), -1)  # rider helmet
        plate_center = (cx, y2 - int(h * 0.10))
    elif body == "auto-rickshaw":
        w = int(220 * scale)
        h = int(220 * scale)
        x1, y1 = cx - w // 2, base_y - h
        x2, y2 = cx + w // 2, base_y
        cv2.ellipse(canvas, (cx, base_y + 4), (int(w * 0.50), int(h * 0.06)), 0, 0, 360, (35, 35, 35), -1)
        _rounded_rect(canvas, (x1, y1 + int(h * 0.18)), (x2, y2 - int(h * 0.10)), (30, 200, 225), radius=int(18 * scale))
        cv2.rectangle(canvas, (x1 + int(w * 0.10), y1 + int(h * 0.28)), (x2 - int(w * 0.10), y1 + int(h * 0.55)), (55, 70, 78), -1)
        cv2.rectangle(canvas, (x1 + int(w * 0.15), y1), (x2 - int(w * 0.15), y1 + int(h * 0.18)), (30, 30, 30), -1)
        cv2.circle(canvas, (cx, base_y - int(h * 0.08)), int(h * 0.09), (25, 25, 25), -1)
        plate_center = (cx, y2 - int(h * 0.22))
    elif body in ("truck", "bus"):
        w = int((390 if body == "truck" else 460) * scale)
        h = int((260 if body == "truck" else 235) * scale)
        x1, y1 = cx - w // 2, base_y - h
        x2, y2 = cx + w // 2, base_y
        cv2.ellipse(canvas, (cx, base_y + 5), (int(w * 0.52), int(h * 0.06)), 0, 0, 360, (35, 35, 35), -1)
        _rounded_rect(canvas, (x1, y1), (x2, y2 - int(h * 0.08)), color, radius=int(10 * scale))
        if body == "truck":
            cv2.rectangle(canvas, (x1 + int(w * 0.04), y1 + int(h * 0.08)), (x2 - int(w * 0.04), y1 + int(h * 0.55)), (65, 65, 70), -1)
            cv2.rectangle(canvas, (x1 + int(w * 0.04), y1 + int(h * 0.58)), (x2 - int(w * 0.04), y2 - int(h * 0.18)), (50, 55, 60), -1)
        else:
            for wx in range(x1 + int(w * 0.06), x2 - int(w * 0.06), int(w * 0.15)):
                cv2.rectangle(canvas, (wx, y1 + int(h * 0.10)), (wx + int(w * 0.11), y1 + int(h * 0.38)), (55, 70, 78), -1)
        for wx in (x1 + int(w * 0.18), x2 - int(w * 0.18)):
            cv2.circle(canvas, (wx, base_y - int(h * 0.06)), int(h * 0.11), (25, 25, 25), -1)
        plate_center = (cx, y2 - int(h * 0.18))
    else:
        w, h = int(300 * scale), int(150 * scale)
        x1, y1 = cx - w // 2, base_y - h
        x2, y2 = cx + w // 2, base_y
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        plate_center = (cx, y2 - int(h * 0.20))
    return x1, y1, x2, y2, plate_center


def make_clip(
    entry: RegistryEntry,
    out_path: Path,
    variant: str,
    rng: random.Random,
    blur_level: str = "medium",
    occlude_ratio: float = 0.18,
    heavy_unreadable: bool = False,
) -> dict:
    ensure_dir(out_path.parent)
    scale = rng.uniform(0.90, 1.08)
    base_y = int(H * 0.86)
    plate_style = entry.plate_style
    frames = []
    for i in range(N_FRAMES):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_background(canvas, variant, rng)
        # Motion + camera shake.
        cx = int(W * 0.50 + math.sin(i / N_FRAMES * math.tau) * 55 + rng.randint(-5, 5))
        base_y_i = base_y + rng.randint(-3, 3)
        x1, y1, x2, y2, plate_center = _draw_vehicle(canvas, entry, cx, base_y_i, scale)
        plate_w = int(min(250, max(120, (x2 - x1) * 0.52)))
        plate_h = max(38, int(plate_w / 3.15))
        plate = render_plate(entry.plate, plate_style, plate_w, plate_h)
        px = int(plate_center[0] - plate_w / 2)
        py = int(plate_center[1] - plate_h / 2)
        px = max(2, min(W - plate_w - 2, px))
        py = max(2, min(H - plate_h - 2, py))
        canvas[py:py + plate_h, px:px + plate_w] = plate

        # Partial occlusion: simulate a frame where only part of the plate is visible.
        occluded = False
        if heavy_unreadable:
            cover_w = int(plate_w * 0.80)
            cv2.rectangle(canvas, (px + plate_w - cover_w, py - 2), (px + plate_w + 2, py + plate_h + 2), (18, 18, 18), -1)
            occluded = True
        elif rng.random() < occlude_ratio:
            side = rng.choice(["left", "right"])
            cover_w = int(plate_w * rng.uniform(0.28, 0.48))
            if side == "left":
                cv2.rectangle(canvas, (px - 2, py - 2), (px + cover_w, py + plate_h + 2), (22, 22, 22), -1)
            else:
                cv2.rectangle(canvas, (px + plate_w - cover_w, py - 2), (px + plate_w + 2, py + plate_h + 2), (22, 22, 22), -1)
            occluded = True

        # Camera / compression degradation.
        if blur_level == "heavy" or heavy_unreadable:
            k = rng.choice([9, 11, 13])
            canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        elif blur_level == "medium":
            if rng.random() < 0.55:
                k = rng.choice([3, 5, 7])
                canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        if rng.random() < 0.35:
            # Horizontal motion blur.
            length = rng.choice([5, 7, 9])
            kernel = np.zeros((length, length), dtype=np.float32)
            kernel[length // 2, :] = 1.0 / length
            canvas = cv2.filter2D(canvas, -1, kernel)
        if rng.random() < 0.45:
            factor = rng.uniform(0.55, 0.80)
            small = cv2.resize(canvas, (int(W * factor), int(H * factor)), interpolation=cv2.INTER_AREA)
            canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
        noise = rng.uniform(2.0, 8.0)
        canvas = np.clip(canvas.astype(np.float32) + np.random.normal(0, noise, canvas.shape), 0, 255).astype(np.uint8)
        frames.append(canvas)

    # Write video; fall back to a frame directory if the codec is unavailable.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (W, H))
    wrote_video = False
    if writer.isOpened():
        for frame in frames:
            writer.write(frame)
        writer.release()
        wrote_video = out_path.exists() and out_path.stat().st_size > 1000
    if not wrote_video:
        frame_dir = out_path.with_suffix("")
        ensure_dir(frame_dir)
        for idx, frame in enumerate(frames):
            imwrite(frame_dir / f"frame_{idx:04d}.jpg", frame, 90)
        out_path = frame_dir

    return {
        "clip": str(out_path.as_posix()),
        "vehicle_id": entry.vehicle_id,
        "plate": entry.plate,
        "make": entry.make,
        "model": entry.model,
        "color": entry.color,
        "body_type": entry.body_type,
        "plate_style": entry.plate_style,
        "category": entry.category,
        "in_registry": True,
        "expected_decision": "SHORTLIST",
        "camera": variant,
        "group_id": entry.plate,
        "split": "test",
        "occluded_frames": sum(1 for _ in frames) if heavy_unreadable else "some",
        "notes": f"blur={blur_level}, occlusion={occlude_ratio}",
    }


def make_nomatch_clip(
    plate: str,
    color: str,
    body_type: str,
    out_path: Path,
    variant: str,
    rng: random.Random,
    plate_style: str = "white",
    heavy_unreadable: bool = False,
    note: str = "",
) -> dict:
    entry = RegistryEntry(
        vehicle_id=f"unknown-{plate}",
        plate=plate,
        make="unknown",
        model="unknown",
        color=color,
        body_type=body_type,
        plate_style=plate_style,
        category="commercial" if plate_style == "yellow" else "private",
        state=plate[:2],
    )
    info = make_clip(
        entry,
        out_path,
        variant,
        rng,
        blur_level="heavy" if heavy_unreadable else "medium",
        occlude_ratio=0.0 if heavy_unreadable else 0.10,
        heavy_unreadable=heavy_unreadable,
    )
    info.update(
        {
            "vehicle_id": f"unknown-{plate}",
            "in_registry": False,
            "expected_decision": "NO_CONFIDENT_MATCH",
            "notes": note or "Plate is not present in the sample registry.",
        }
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    registry = load_registry("data/registry/vehicles.csv")
    out_root = Path(args.out)
    clips_dir = ensure_dir(out_root / "clips")

    def entry(plate: str) -> RegistryEntry:
        item = registry.get(plate) or registry.get("DL08CA1234" if plate == "DL8CAF1234" else plate)
        if item is None:
            raise SystemExit(f"Demo vehicle {plate} missing from registry")
        return item

    clips: list[dict] = []
    # Same vehicle, two cameras (tracking bonus).
    clips.append(make_clip(entry("JH05AB6799"), clips_dir / "jh05ab6799_camA.mp4", "city", rng, "heavy", 0.22))
    clips.append(make_clip(entry("JH05AB6799"), clips_dir / "jh05ab6799_camB.mp4", "highway", rng, "medium", 0.15))
    # Attribute coverage.
    clips.append(make_clip(entry("MH12DE1433"), clips_dir / "mh12de1433_camA.mp4", "city", rng, "medium", 0.18))
    clips.append(make_clip(entry("KA03MN4821"), clips_dir / "ka03mn4821_camA.mp4", "night", rng, "medium", 0.18))
    clips.append(make_clip(entry("CG04TR7788"), clips_dir / "cg04tr7788_camA.mp4", "city", rng, "medium", 0.20))
    clips.append(make_clip(entry("UP32KL9021"), clips_dir / "up32kl9021_camA.mp4", "highway", rng, "medium", 0.15))
    clips.append(make_clip(entry("DL8CAF1234"), clips_dir / "dl8caf1234_camA.mp4", "city", rng, "medium", 0.18))
    # No-match cases.
    clips.append(
        make_nomatch_clip(
            "JH05AB6798", "white", "suv", clips_dir / "jh05ab6798_nomatch.mp4",
            "city", rng, note="Near miss: one digit differs from a registry entry; must NOT be forced.",
        )
    )
    clips.append(
        make_nomatch_clip(
            "TN09ZZ4321", "black", "hatchback", clips_dir / "tn09zz4321_nomatch.mp4",
            "night", rng, note="Different plate, same attributes as several registry vehicles.",
        )
    )
    clips.append(
        make_nomatch_clip(
            "WB02XX0000", "silver", "sedan", clips_dir / "wb02xx0000_unreadable.mp4",
            "city", rng, heavy_unreadable=True,
            note="Plate is deliberately unreadable in every frame; expected NO_CONFIDENT_MATCH.",
        )
    )

    labels = {
        "generated_with": "scripts/make_demo_data.py",
        "seed": args.seed,
        "fps": FPS,
        "frames_per_clip": N_FRAMES,
        "resolution": [W, H],
        "split_policy": "grouped by vehicle_id / group_id; never by frame",
        "clips": clips,
    }
    (out_root / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Wrote {len(clips)} demo clips to {clips_dir}")
    print(f"Labels: {out_root / 'labels.json'}")


if __name__ == "__main__":
    main()
