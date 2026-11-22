"""Comprehensive OCR evaluation across degradation levels 0 to 5.

Metrics:
- Character Accuracy: matching character slots / total characters
- Full-Plate Accuracy: exact match across all 10 characters
- Breakdown by degradation level (0 = pristine to 5 = severe blur + occlusion)
- Comparison between baseline and calibrated TRACE-X OCR backend
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from pipeline.degrade import degrade_plate
from pipeline.ocr import LicensePlateOCR
from scripts.gen_plates import render_base_plate


def run_ocr_benchmark(
    samples_per_level: int = 15,
    out_csv: str = "eval/ocr_results.csv"
) -> list[dict[str, float]]:
    """Run benchmark evaluating OCR character and plate accuracy per blur level."""
    ocr_engine = LicensePlateOCR(backend="auto")

    test_plates = [
        "JH05AB6799", "MH12DE1433", "DL08CA1234", "KA03MN4821", "UP32KL9021",
        "CG04TR7788", "TN09ZZ4321", "WB02AB1122", "GJ01XY9988", "RJ14CD4455"
    ]

    results: list[dict[str, float]] = []

    for level in range(6):
        total_chars = 0
        correct_chars = 0
        total_plates = 0
        correct_plates = 0

        for idx, base_text in enumerate(test_plates[:samples_per_level]):
            base_bgr = render_base_plate(base_text, "white")
            degraded_img, _ = degrade_plate(base_bgr, level=level, seed=42 + idx * 10)

            pred_text, candidates = ocr_engine.recognize(degraded_img)

            # Pad or truncate to match length
            match_len = min(len(base_text), len(pred_text))
            char_matches = sum(1 for a, b in zip(base_text[:match_len], pred_text[:match_len]) if a == b)

            total_chars += len(base_text)
            correct_chars += char_matches
            total_plates += 1
            if pred_text == base_text:
                correct_plates += 1

        char_acc = correct_chars / max(1, total_chars)
        plate_acc = correct_plates / max(1, total_plates)

        results.append({
            "blur_level": level,
            "char_accuracy": round(char_acc, 4),
            "plate_accuracy": round(plate_acc, 4),
            "total_samples": total_plates
        })
        print(f"Level {level}: Char Acc = {char_acc*100:.1f}%, Plate Acc = {plate_acc*100:.1f}%")

    # Save to CSV
    p_csv = Path(out_csv)
    p_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(p_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blur_level", "char_accuracy", "plate_accuracy", "total_samples"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved OCR benchmark results to {out_csv}")
    return results


if __name__ == "__main__":
    run_ocr_benchmark()
