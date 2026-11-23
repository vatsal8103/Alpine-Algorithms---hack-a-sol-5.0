#!/usr/bin/env python3
"""TRACE-X Kaggle Dataset Downloader & Ingestion Utility.

Downloads and organizes real Indian license plate datasets from Kaggle:
1. witvixsystem/indian-truck-license-plates-sample (Commercial Trucks LPD+LPR)
2. santoshvishwakarma99/indian-license-plate-dataset (HSRP vs Non-HSRP Plates)

Usage:
    # 1. Download all datasets (Default):
    python download.py

    # 2. Download only the Truck License Plate dataset:
    python download.py --dataset trucks

    # 3. Download only the HSRP vs Non-HSRP dataset:
    python download.py --dataset hsrp

    # 4. Download and automatically ingest crops into data/kaggle_crops/:
    python download.py --ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Registry of supported datasets
DATASETS: dict[str, dict[str, Any]] = {
    "trucks": {
        "title": "Indian Truck License Plates Sample (LPD+LPR)",
        "slug": "witvixsystem/indian-truck-license-plates-sample",
        "hf_slug": "witvix/indian-truck-license-plates-sample",
        "default_dir": "data/raw/truck_plates",
        "description": "Heavy commercial Indian trucks with bounding boxes and alphanumeric plate labels."
    },
    "hsrp": {
        "title": "Indian License Plate Dataset (HSRP vs Non-HSRP)",
        "slug": "santoshvishwakarma99/indian-license-plate-dataset",
        "default_dir": "data/raw/hsrp_plates",
        "description": "Indian vehicles classified into High-Security (HSRP) and standard non-HSRP plates."
    }
}


def download_with_kagglehub(dataset_slug: str, target_dir: Path) -> Path | None:
    """Download public dataset using kagglehub (works with zero manual API key setup)."""
    try:
        import kagglehub
        print(f"[*] Fetching '{dataset_slug}' via KaggleHub...")
        cached_path = kagglehub.dataset_download(dataset_slug)
        cached_p = Path(cached_path)
        if cached_p.exists():
            print(f"[+] Download complete! Cached at: {cached_p}")
            return cached_p
    except Exception as exc:
        print(f"[!] KaggleHub download error: {exc}")
    return None


def download_with_kaggle_api(dataset_slug: str, target_dir: Path) -> Path | None:
    """Fallback download using the official Kaggle API client or CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[*] Authenticated with Kaggle API. Downloading '{dataset_slug}'...")
        api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
        print(f"[+] Kaggle API download complete: {target_dir}")
        return target_dir
    except Exception as exc:
        print(f"[!] Kaggle API client error: {exc}")

    # Try subprocess CLI
    try:
        print(f"[*] Attempting 'kaggle datasets download' command...")
        res = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_dir), "--unzip"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            print(f"[+] CLI download complete: {target_dir}")
            return target_dir
    except Exception:
        pass

    return None


def download_with_hf_mirror(hf_slug: str, target_dir: Path) -> Path | None:
    """Fallback open mirror download from Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Trying Hugging Face open mirror: '{hf_slug}'...")
        res = snapshot_download(repo_id=hf_slug, repo_type="dataset", local_dir=str(target_dir))
        return Path(res)
    except Exception as exc:
        print(f"[!] Hugging Face mirror error: {exc}")
        return None


def sync_to_destination(source_dir: Path, target_dir: Path) -> None:
    """Safely copy or link downloaded files to target directory."""
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Syncing files to target directory: {target_dir}...")
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if not dest.exists():
            if item.is_dir():
                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest))


def count_files_in_dir(directory: Path) -> dict[str, int]:
    """Scan and count images and annotation files in a directory."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    annot_exts = {".xml", ".json", ".csv", ".txt", ".yaml", ".yml"}
    images = 0
    annots = 0
    total_bytes = 0

    if directory.exists():
        for root, _, files in os.walk(directory):
            for f in files:
                ext = Path(f).suffix.lower()
                fp = Path(root) / f
                try:
                    total_bytes += fp.stat().st_size
                except Exception:
                    pass
                if ext in image_exts:
                    images += 1
                elif ext in annot_exts:
                    annots += 1

    return {
        "images": images,
        "annotations": annots,
        "size_mb": round(total_bytes / (1024 * 1024), 2)
    }


def ingest_crops_to_registry(source_dir: Path, output_crop_dir: Path = Path("data/kaggle_crops")) -> int:
    """Auto-ingest cropped plates from downloaded images into data/kaggle_crops/."""
    output_crop_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = output_crop_dir / "labels.csv"
    existing_images = set()

    if labels_csv.exists():
        import csv
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_images.add(Path(r.get("image_path", "")).name)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    found_images = [
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    print(f"[*] Found {len(found_images)} images. Scanning and registering in {labels_csv}...")
    # Add new crops to labels.csv
    import csv
    new_count = 0
    fieldnames = ["image_path", "plate_text", "plate_style", "source"]

    with open(labels_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not labels_csv.exists() or labels_csv.stat().st_size == 0:
            writer.writeheader()

        for img_p in found_images:
            if img_p.name in existing_images:
                continue

            # Infer label or class from filename or parent folder
            parent_name = img_p.parent.name.lower()
            style = "hsrp" if "hsrp" in parent_name else ("truck" if "truck" in parent_name else "standard")
            
            # Simple clean text parsing if filename contains plate text
            stem = img_p.stem
            plate_candidate = ""
            if len(stem) >= 8:
                import re
                clean = re.sub(r"[^A-Za-z0-9]", "", stem).upper()
                if 8 <= len(clean) <= 11 and any(c.isdigit() for c in clean) and any(c.isalpha() for c in clean):
                    plate_candidate = clean

            writer.writerow({
                "image_path": str(img_p),
                "plate_text": plate_candidate or "UNKNOWN",
                "plate_style": style,
                "source": img_p.parent.parent.name or "kaggle"
            })
            existing_images.add(img_p.name)
            new_count += 1

    print(f"[+] Successfully registered {new_count} new samples in {labels_csv}")
    return new_count


def process_dataset(key: str, cfg: dict[str, Any], dest_override: str | None = None, ingest: bool = False) -> bool:
    print("\n" + "=" * 70)
    print(f"  DATASET: {cfg['title']}")
    print(f"  Kaggle Slug: {cfg['slug']}")
    print(f"  Description: {cfg['description']}")
    print("=" * 70)

    target_dir = Path(dest_override or cfg["default_dir"])
    download_dir: Path | None = None

    # Step 1: Try KaggleHub
    download_dir = download_with_kagglehub(cfg["slug"], target_dir)

    # Step 2: Fallback to official Kaggle API
    if not download_dir:
        download_dir = download_with_kaggle_api(cfg["slug"], target_dir)

    # Step 3: Fallback to Hugging Face if mirror exists
    if not download_dir and cfg.get("hf_slug"):
        download_dir = download_with_hf_mirror(cfg["hf_slug"], target_dir)

    if not download_dir:
        print(f"\n[X] Could not automatically download '{cfg['slug']}'.")
        print("    To download manually via Kaggle API:")
        print("    1. Place your 'kaggle.json' token into: ~/.kaggle/kaggle.json")
        print(f"    2. Run: kaggle datasets download -d {cfg['slug']} -p {target_dir} --unzip")
        return False

    # Sync into target directory
    sync_to_destination(download_dir, target_dir)

    # Stats
    stats = count_files_in_dir(target_dir)
    print(f"\n[✓] Finished processing {cfg['title']}:")
    print(f"    Location:    {target_dir.resolve()}")
    print(f"    Images:      {stats['images']} files")
    print(f"    Annotations: {stats['annotations']} files")
    print(f"    Total Size:  {stats['size_mb']} MB")

    if ingest:
        ingest_crops_to_registry(target_dir)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE-X Kaggle Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "trucks", "hsrp"],
        default="all",
        help="Which dataset to download ('all', 'trucks', 'hsrp'). Default: 'all'"
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Custom destination directory to store downloaded files"
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Automatically register and crop downloaded plates into data/kaggle_crops/labels.csv"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("      TRACE-X REAL INDIAN ROAD DATASET DOWNLOADER (KAGGLE)")
    print("=" * 70)

    selected = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    success_count = 0

    for key in selected:
        success = process_dataset(key, DATASETS[key], dest_override=args.dest_dir, ingest=args.ingest)
        if success:
            success_count += 1

    print("\n" + "=" * 70)
    print(f"Summary: {success_count}/{len(selected)} datasets downloaded successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
