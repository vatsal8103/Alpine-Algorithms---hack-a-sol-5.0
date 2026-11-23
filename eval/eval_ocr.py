"""Comprehensive OCR Comparative Evaluation: Pretrained vs Fine-Tuned TRACE-X.

Metrics:
- Character Accuracy: matching character positions / total characters
- Full-Plate Accuracy: exact match across all characters
- Graded comparison across degradation levels 0 to 5
- Generates CSV (eval/ocr_comparison_results.csv)
- Generates Accuracy vs Blur Curve chart (eval/ocr_accuracy_vs_blur.png)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline.degrade import degrade_plate
from pipeline.ocr import LicensePlateOCR
from scripts.gen_plates import render_base_plate


def run_comparative_benchmark(
    samples_per_level: int = 15,
    out_csv: str = "eval/ocr_comparison_results.csv",
    out_plot: str = "eval/ocr_accuracy_vs_blur.png"
) -> list[dict[str, any]]:
    print("\n=======================================================")
    print("Evaluating OCR: Pretrained Baseline vs Fine-Tuned Model")
    print(f"Degradation Levels: 0 to 5 | Samples/Level: {samples_per_level}")
    print("=======================================================\n")

    baseline_engine = LicensePlateOCR(backend="auto")
    finetuned_engine = LicensePlateOCR(backend="finetuned")

    test_plates = [
        "MP04ZK6184", "MP13ZU2050", "JH05AB6799", "MH12DE1433", "DL08CA1234",
        "KA03MN4821", "UP32KL9021", "CG04TR7788", "TN09ZZ4321", "WB02AB1122",
        "GJ01XY9988", "RJ14CD4455", "HR26DK8392", "AP09BZ1290", "TS07FA5566"
    ]

    records: list[dict[str, any]] = []

    for level in range(6):
        # Baseline tallies
        base_correct_chars = 0
        base_correct_plates = 0

        # Fine-tuned tallies
        ft_correct_chars = 0
        ft_correct_plates = 0

        total_chars = 0
        total_plates = 0

        for idx, base_text in enumerate(test_plates[:samples_per_level]):
            base_bgr = render_base_plate(base_text, "white")
            degraded_img, _ = degrade_plate(base_bgr, level=level, seed=100 + idx * 7)

            # 1. Baseline prediction
            pred_base, _ = baseline_engine.recognize(degraded_img)
            m_base = min(len(base_text), len(pred_base))
            base_matches = sum(1 for a, b in zip(base_text[:m_base], pred_base[:m_base]) if a == b)
            base_correct_chars += base_matches
            if pred_base == base_text:
                base_correct_plates += 1

            # 2. Fine-tuned prediction
            pred_ft, _ = finetuned_engine.recognize(degraded_img)
            m_ft = min(len(base_text), len(pred_ft))
            ft_matches = sum(1 for a, b in zip(base_text[:m_ft], pred_ft[:m_ft]) if a == b)
            ft_correct_chars += ft_matches
            if pred_ft == base_text:
                ft_correct_plates += 1

            total_chars += len(base_text)
            total_plates += 1

        base_char_acc = base_correct_chars / max(1, total_chars)
        base_plate_acc = base_correct_plates / max(1, total_plates)

        ft_char_acc = ft_correct_chars / max(1, total_chars)
        ft_plate_acc = ft_correct_plates / max(1, total_plates)

        records.append({
            "blur_level": level,
            "baseline_char_acc": round(base_char_acc, 4),
            "baseline_plate_acc": round(base_plate_acc, 4),
            "finetuned_char_acc": round(ft_char_acc, 4),
            "finetuned_plate_acc": round(ft_plate_acc, 4),
            "total_plates": total_plates
        })

        print(
            f"Level {level:d} | "
            f"Baseline: Char {base_char_acc*100:5.1f}% / Plate {base_plate_acc*100:5.1f}%  -->  "
            f"Fine-Tuned: Char {ft_char_acc*100:5.1f}% / Plate {ft_plate_acc*100:5.1f}%"
        )

    # Save to CSV
    p_csv = Path(out_csv)
    p_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(p_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "blur_level", "baseline_char_acc", "baseline_plate_acc",
            "finetuned_char_acc", "finetuned_plate_acc", "total_plates"
        ])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved benchmark comparison table to {out_csv}")

    # Plot Accuracy vs Blur curve
    p_plot = Path(out_plot)
    p_plot.parent.mkdir(parents=True, exist_ok=True)

    levels = [r["blur_level"] for r in records]
    b_char = [r["baseline_char_acc"] * 100 for r in records]
    b_plate = [r["baseline_plate_acc"] * 100 for r in records]
    ft_char = [r["finetuned_char_acc"] * 100 for r in records]
    ft_plate = [r["finetuned_plate_acc"] * 100 for r in records]

    plt.figure(figsize=(9, 5.5))
    plt.plot(levels, b_char, "o--", color="#94a3b8", label="Pretrained Char Acc", linewidth=1.8)
    plt.plot(levels, b_plate, "s--", color="#64748b", label="Pretrained Full Plate Acc", linewidth=1.8)
    plt.plot(levels, ft_char, "o-", color="#3b82f6", label="Fine-Tuned Char Acc", linewidth=2.5)
    plt.plot(levels, ft_plate, "s-", color="#10b981", label="Fine-Tuned Full Plate Acc", linewidth=2.5)

    plt.title("TRACE-X OCR Accuracy vs Degradation Level (0 = Pristine, 5 = Severe)", fontsize=13, fontweight="bold")
    plt.xlabel("Degradation / Blur Level", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(str(p_plot), dpi=200)
    plt.close()
    print(f"Saved Accuracy vs Blur curve chart to {out_plot}\n")

    return records


if __name__ == "__main__":
    run_comparative_benchmark()
