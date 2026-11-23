"""Download and prepare Indian License Plate dataset from Kaggle.

Usage:
    # 1. Direct Kaggle download (requires ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KEY env vars):
    python scripts/download_kaggle_dataset.py --dataset adityamangal98/hr-lr-images-of-the-indian-license-plates

    # 2. Open mirror download (works out-of-the-box without Kaggle API credentials):
    python scripts/download_kaggle_dataset.py --source mirror

Kaggle / Colab Terminal Commands:
    !pip install kaggle
    !kaggle datasets download -d adityamangal98/hr-lr-images-of-the-indian-license-plates -p data/kaggle --unzip
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def download_from_kaggle_api(dataset_slug: str, out_dir: Path) -> bool:
    """Download and unzip a dataset using official Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"Authenticated with Kaggle. Downloading {dataset_slug}...")
        api.dataset_download_files(dataset_slug, path=str(out_dir), unzip=True)
        print(f"Successfully downloaded {dataset_slug} to {out_dir}")
        return True
    except Exception as e:
        print(f"Kaggle API download failed: {e}")
        return False


def download_from_hf_mirror(out_dir: Path) -> bool:
    """Download real Indian plate crops mirror from HuggingFace repository."""
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading Indian plate crop mirror from HuggingFace...")
        zip_path = hf_hub_download(
            repo_id="adityamangal98/HR-LR-images-of-the-Indian-License-plates",
            filename="LR.zip",
            repo_type="dataset"
        )
        print(f"Extracting {zip_path} to {out_dir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"Mirror download failed: {e}")
        return False


def prepare_kaggle_crops(crops_dir: Path) -> int:
    """Scan extracted images and build annotations manifest (image_path \t label)."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [p for p in crops_dir.rglob("*") if p.suffix.lower() in image_exts]
    manifest_path = crops_dir / "kaggle_manifest.txt"

    valid_count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in img_files:
            # File stem often contains registration plate (e.g. MH12DE1433.jpg)
            stem = p.stem.upper().replace("_", "").replace("-", "").replace(" ", "")
            # Basic sanity check for Indian plate format
            if len(stem) >= 8 and len(stem) <= 12 and stem[:2].isalpha():
                f.write(f"{str(p).replace('\\', '/')}\t{stem}\n")
                valid_count += 1
            else:
                f.write(f"{str(p).replace('\\', '/')}\tUNLABELED\n")

    print(f"Scanned {len(img_files)} images from Kaggle dataset ({valid_count} labeled).")
    print(f"Saved manifest to {manifest_path}")
    return len(img_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Kaggle Indian plate dataset")
    parser.add_argument("--dataset", type=str, default="adityamangal98/hr-lr-images-of-the-indian-license-plates")
    parser.add_argument("--out-dir", type=str, default="data/kaggle_crops")
    parser.add_argument("--source", type=str, choices=["kaggle", "mirror", "auto"], default="auto")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    success = False
    if args.source in ("kaggle", "auto"):
        success = download_from_kaggle_api(args.dataset, out_path)

    if not success and args.source in ("mirror", "auto"):
        print("\nFallback to open dataset mirror...")
        success = download_from_hf_mirror(out_path)

    if success:
        prepare_kaggle_crops(out_path)
    else:
        print("\nTo manually attach this dataset on Kaggle/Colab, execute:")
        print(f"  kaggle datasets download -d {args.dataset} -p {args.out_dir} --unzip")


if __name__ == "__main__":
    main()
