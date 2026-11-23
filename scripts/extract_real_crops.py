"""Real Plate Crop Extractor and Dataset Splitter.

Features:
1. Extracts high-resolution plate crops from real video clips using plate detector + 20% margin + deskew.
2. Organizes extracted crops by clip: data/real_crops/<clip_id>/f{frame_no:04d}_t{track_id}.png.
3. Automatically maps ground-truth labels from CSV (clip, frame, text) or ground_truth.csv.
4. Splits labeled samples strictly by video clip into train_real.txt and val_real.txt to prevent data leakage.
5. Emits annotations in both PyTorch and PaddleOCR formats.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from pipeline.plate_stage import PlateStage, deskew_plate


def extract_crops_from_clip(
    video_path: Path,
    output_dir: Path,
    plate_stage: PlateStage,
    sample_stride: int = 12,
    max_crops_per_clip: int = 50
) -> list[dict[str, Any]]:
    """Extract plate crops from a single video clip."""
    clip_id = video_path.stem
    clip_out = output_dir / clip_id
    clip_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Warning] Unable to open video: {video_path}")
        return []

    extracted_records: list[dict[str, Any]] = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_crops_per_clip:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride == 0:
            fh, fw = frame.shape[:2]
            # Detect plate box
            px, py, pw, ph = plate_stage.locate_plate_box(frame)
            if pw > 30 and ph > 15:
                # 20% margin expansion
                mx = int(pw * 0.20)
                my = int(ph * 0.20)
                x1 = max(0, px - mx)
                y1 = max(0, py - my)
                x2 = min(fw, px + pw + mx)
                y2 = min(fh, py + ph + my)

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_deskewed = deskew_plate(crop)
                    crop_fname = f"f{frame_idx:04d}_t0.png"
                    crop_save_path = clip_out / crop_fname
                    cv2.imwrite(str(crop_save_path), crop_deskewed)

                    extracted_records.append({
                        "clip": video_path.name,
                        "clip_id": clip_id,
                        "frame": frame_idx,
                        "track_id": 0,
                        "image_path": str(crop_save_path).replace("\\", "/"),
                        "text": ""
                    })
                    saved_count += 1

        frame_idx += 1

    cap.release()
    return extracted_records


def build_real_dataset(
    clips_dir: str = "data/clips",
    output_dir: str = "data/real_crops",
    labels_csv: str | None = "data/labels/ground_truth.csv",
    val_ratio: float = 0.20,
    seed: int = 42
) -> tuple[str, str, str]:
    """Extract real crops from video clips, map ground-truth labels, and split by clip."""
    p_clips = Path(clips_dir)
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    plate_stage = PlateStage()

    video_files = sorted(list(p_clips.glob("*.mp4")) + list(p_clips.glob("*.avi")))
    if not video_files:
        # Check runs/uploads/ as fallback
        p_uploads = Path("runs/uploads")
        if p_uploads.exists():
            video_files = sorted(list(p_uploads.glob("*.mp4")))

    print(f"Found {len(video_files)} video clips in {clips_dir} & uploads.")

    # Load existing labels if available
    ground_truth_map: dict[str, str] = {}
    if labels_csv and Path(labels_csv).exists():
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both true_plate and text column headers
                plate = row.get("true_plate") or row.get("text") or ""
                clip_k = row.get("clip_name") or row.get("clip") or ""
                if clip_k and plate:
                    ground_truth_map[clip_k] = plate.strip().upper()
                    ground_truth_map[Path(clip_k).stem] = plate.strip().upper()

    all_records: list[dict[str, Any]] = []
    for vid in video_files:
        recs = extract_crops_from_clip(vid, p_out, plate_stage)
        clip_label = ground_truth_map.get(vid.name, ground_truth_map.get(vid.stem, ""))
        for r in recs:
            r["text"] = clip_label
        all_records.extend(recs)
        print(f"  [{vid.name}] Extracted {len(recs)} crops. Ground truth: {clip_label or 'UNLABELLED'}")

    # Write labels template CSV
    master_csv = p_out / "labels.csv"
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "clip_id", "frame", "track_id", "image_path", "text"])
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nWrote full crops inventory to {master_csv} ({len(all_records)} total crops)")

    # Filter labeled records for training/validation split
    labeled = [r for r in all_records if r["text"] and not r["text"].startswith("?")]
    distinct_clips = sorted(list(set(r["clip_id"] for r in labeled)))

    rng = random.Random(seed)
    rng.shuffle(distinct_clips)

    n_val_clips = max(1, int(len(distinct_clips) * val_ratio)) if len(distinct_clips) > 1 else 0
    val_clips = set(distinct_clips[:n_val_clips])
    train_clips = set(distinct_clips[n_val_clips:])

    train_records = [r for r in labeled if r["clip_id"] in train_clips]
    val_records = [r for r in labeled if r["clip_id"] in val_clips]

    # Save manifest files for PaddleOCR & PyTorch
    train_txt = p_out / "train_real.txt"
    val_txt = p_out / "val_real.txt"

    with open(train_txt, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(f"{r['image_path']}\t{r['text']}\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(f"{r['image_path']}\t{r['text']}\n")

    print("\nDataset Split (Grouped strictly by clip to prevent leakage):")
    print(f"  Train Clips: {sorted(list(train_clips))} ({len(train_records)} crops) -> {train_txt}")
    print(f"  Val Clips:   {sorted(list(val_clips))} ({len(val_records)} crops) -> {val_txt}")

    return str(master_csv), str(train_txt), str(val_txt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real plate crops and build split manifests")
    parser.add_argument("--clips-dir", type=str, default="data/clips", help="Directory containing video clips")
    parser.add_argument("--output-dir", type=str, default="data/real_crops", help="Output directory for crops")
    parser.add_argument("--labels-csv", type=str, default="data/labels/ground_truth.csv", help="Ground truth labels CSV")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Ratio of clips reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip split")
    args = parser.parse_args()

    build_real_dataset(
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        labels_csv=args.labels_csv,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
