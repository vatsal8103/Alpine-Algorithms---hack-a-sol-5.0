#!/usr/bin/env python3
"""TRACE-X VLM License Plate Test Utility.

Tests Qwen2-VL-2B Indian License Plate Engine on an image crop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pipeline.vlm_engine import Qwen2VLLicensePlateRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Qwen2-VL-2B on a License Plate Image")
    parser.add_argument("image_path", type=str, nargs="?", default="runs/cases/case-test1-0047/track_118_plate.jpg",
                        help="Path to plate crop image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device ('cuda:0' or 'cpu')")
    args = parser.parse_args()

    img_p = Path(args.image_path)
    if not img_p.exists():
        # Fallback to any plate crop in runs/cases
        crops = list(Path("runs/cases").glob("*/*plate.jpg"))
        if crops:
            img_p = crops[0]
            print(f"[*] Default image not found. Using sample: {img_p}")
        else:
            print(f"[X] Image not found: {img_p}")
            sys.exit(1)

    print("=" * 65)
    print("      TRACE-X QWEN2-VL-2B LICENSE PLATE RECOGNITION TEST")
    print("=" * 65)
    print(f"  Target Image: {img_p}")
    print(f"  Device:       {args.device}")
    print("=" * 65)

    recognizer = Qwen2VLLicensePlateRecognizer(device=args.device)
    success = recognizer.load_model()
    if not success:
        print("[X] Could not load Qwen2-VL-2B.")
        sys.exit(1)

    plate_text = recognizer.recognize(img_p)
    print("\n" + "=" * 65)
    print(f"[+] Recognized Indian Plate: '{plate_text}'")
    print("=" * 65)

    recognizer.unload_model()


if __name__ == "__main__":
    main()
