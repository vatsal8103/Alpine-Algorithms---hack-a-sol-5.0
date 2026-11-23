"""Ingest and crop all plates from Kaggle Dataset 2 (dataclusterlabs/indian-number-plates-dataset).

Saves cropped license plate images to data/kaggle_crops/ and registers them in data/kaggle_crops/labels.csv.
"""

from __future__ import annotations

import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from pipeline.ocr import LicensePlateOCR


def ingest_datacluster_dataset() -> int:
    base_dir = Path(r"C:\Users\argho\.cache\kagglehub\datasets\dataclusterlabs\indian-number-plates-dataset\versions\3")
    if not base_dir.exists():
        print(f"Error: Dataset directory {base_dir} does not exist.")
        return 0

    out_crop_dir = Path("data/kaggle_crops")
    out_crop_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_crop_dir / "labels.csv"

    # Load existing labels
    existing_paths = set()
    existing_rows = []
    if labels_csv.exists():
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_paths.add(r["image_path"])
                existing_rows.append(r)

    print(f"Found {len(existing_rows)} existing crops in {labels_csv}")

    # Initialize OCR recognizer for unlabeled samples
    ocr = LicensePlateOCR(backend="auto")

    xml_files = sorted(base_dir.rglob("*.xml"))
    print(f"Processing {len(xml_files)} XML annotation files from DataCluster dataset...")

    added = 0
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Skipping corrupt XML {xml_path}: {e}")
            continue

        fname_el = root.find("filename")
        if fname_el is None or not fname_el.text:
            continue
        fname = fname_el.text.strip()

        # Locate image file
        img_candidates = list(base_dir.rglob(fname))
        if not img_candidates:
            img_candidates = list(base_dir.rglob(Path(fname).stem + ".*"))
        if not img_candidates:
            continue

        img_path = img_candidates[0]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for idx, obj in enumerate(root.findall("object")):
            # Check ground truth text in attributes
            text = ""
            source = "datacluster_groundtruth"
            attrs = obj.find("attributes")
            if attrs is not None:
                for a in attrs.findall("attribute"):
                    name_el = a.find("name")
                    val_el = a.find("value")
                    if name_el is not None and name_el.text == "number_plate_text" and val_el is not None and val_el.text:
                        text = val_el.text.strip()
                        break

            # Bounding box
            bnd = obj.find("bndbox")
            if bnd is None:
                continue

            try:
                xmin_f = float(bnd.find("xmin").text)
                ymin_f = float(bnd.find("ymin").text)
                xmax_f = float(bnd.find("xmax").text)
                ymax_f = float(bnd.find("ymax").text)
            except Exception:
                continue

            # Add 8% margin
            bw = xmax_f - xmin_f
            bh = ymax_f - ymin_f
            pad_x = bw * 0.08
            pad_y = bh * 0.08

            xmin = max(0, int(xmin_f - pad_x))
            ymin = max(0, int(ymin_f - pad_y))
            xmax = min(w, int(xmax_f + pad_x))
            ymax = min(h, int(ymax_f + pad_y))

            if xmax <= xmin + 10 or ymax <= ymin + 10:
                continue

            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            # Clean text
            clean_text = re.sub(r"[^A-Z0-9]", "", text.upper())
            if not clean_text or len(clean_text) < 4:
                # OCR fallback
                rec_text, _ = ocr.recognize(crop)
                clean_text = re.sub(r"[^A-Z0-9]", "", rec_text.upper())
                source = "datacluster_ocr"

            if len(clean_text) < 4:
                continue

            # Save crop
            safe_stem = Path(fname).stem.replace(" ", "_").replace("(", "").replace(")", "")
            crop_filename = f"dc_{safe_stem}_obj{idx}.png"
            rel_crop_path = f"data/kaggle_crops/{crop_filename}"
            abs_crop_path = out_crop_dir / crop_filename

            if rel_crop_path not in existing_paths:
                cv2.imwrite(str(abs_crop_path), crop)
                existing_rows.append({
                    "image_path": rel_crop_path,
                    "text": clean_text,
                    "source": source
                })
                existing_paths.add(rel_crop_path)
                added += 1

    # Write updated labels.csv
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "text", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print(f"\n[+] Added {added} new high-quality crops from DataCluster dataset.")
    print(f"[+] Total combined real Kaggle crops: {len(existing_rows)}")
    return added


if __name__ == "__main__":
    ingest_datacluster_dataset()
