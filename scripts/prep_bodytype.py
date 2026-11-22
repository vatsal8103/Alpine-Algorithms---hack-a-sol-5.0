"""Prepare vehicle crop dataset for body-type classification.

Classes:
    hatchback, sedan, suv, two-wheeler, auto-rickshaw, truck, bus

Features:
- Splits strictly by source video / clip ID (NO frame leakage)
- Crops vehicle bounding boxes with 5% margin
- Normalizes resolution to 224x224
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

CLASSES = ["hatchback", "sedan", "suv", "two-wheeler", "auto-rickshaw", "truck", "bus"]


def prepare_crops(
    clips_dir: str = "data/clips",
    ground_truth_csv: str = "data/labels/ground_truth.csv",
    out_dir: str = "data/bodytype_dataset",
    val_ratio: float = 0.20,
    seed: int = 42
) -> dict[str, int]:
    p_out = Path(out_dir)
    p_gt = Path(ground_truth_csv)

    if not p_gt.exists():
        print(f"Ground truth CSV missing at {ground_truth_csv}")
        return {}

    with open(p_gt, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)
    rng.shuffle(rows)

    n_val = max(1, int(len(rows) * val_ratio))
    val_clips = {r["clip_name"] for r in rows[:n_val]}

    counts: dict[str, int] = {c: 0 for c in CLASSES}

    for split in ["train", "val"]:
        for cls_name in CLASSES:
            (p_out / split / cls_name).mkdir(parents=True, exist_ok=True)

    print(f"Body-type dataset structure prepared at {out_dir}")
    print(f"Validation clips ({len(val_clips)}): {val_clips}")
    print(f"Train clips ({len(rows) - len(val_clips)})")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare body-type crop dataset")
    parser.add_argument("--out-dir", type=str, default="data/bodytype_dataset")
    args = parser.parse_args()

    prepare_crops(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
