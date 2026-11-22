"""TRACE-X End-to-End Video Analysis Pipeline.

Executes:
1. Frame extraction & Laplacian sharpness calculation
2. Vehicle detection & ByteTrack tracking
3. Best-N frame quality selection & duplicate suppression
4. Plate localization, safety-margin crop & per-frame OCR
5. Multi-frame evidential consensus fusion (with Indian grammar constraints)
6. Vehicle attribute extraction (HSV colour segmentation & body-type classification)
7. Registry matching, multi-criteria ranking, and forensic abstention
8. Emits CaseResult JSON and saves debug crops
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from fusion.fuser import fuse_frames
from pipeline.attributes import extract_attributes
from pipeline.detect_track import VehicleTracker
from pipeline.frames import extract_frames
from pipeline.plate_stage import PlateStage
from pipeline.schemas import CaseResult, FusedPlate, RegistryRecord, VehicleAttributes
from pipeline.select_frames import select_best_frames
from ranking.ranker import rank_candidates


def load_registry_records(registry_csv: str = "data/registry/vehicles.csv") -> list[RegistryRecord]:
    """Load sample vehicle registry into Pydantic RegistryRecord models."""
    p = Path(registry_csv)
    if not p.exists():
        print(f"[pipeline] Warning: Registry CSV not found at {p}. Run scripts/make_registry.py.")
        return []

    records = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(RegistryRecord(
                vehicle_id=row["vehicle_id"],
                registration_number=row["registration_number"],
                colour=row["colour"],
                body_type=row["body_type"],
                make=row["make"],
                model=row["model"],
                fuel=row.get("fuel"),
                year=int(row["year"]) if row.get("year") else None,
                owner_city=row.get("owner_city")
            ))
    return records


def run_pipeline(
    video_path: str,
    track_id_target: int | None = None,
    hints: dict[str, str] | None = None,
    registry_csv: str = "data/registry/vehicles.csv",
    out_dir: str = "runs/cases",
    case_id_override: str | None = None
) -> CaseResult:
    """Execute complete end-to-end video analysis pipeline on specified clip."""
    t_start = time.time()
    clip_p = Path(video_path)
    case_id = case_id_override or f"case-{clip_p.stem}-{int(time.time()) % 10000:04d}"
    case_out = Path(out_dir) / case_id
    crops_dir = case_out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TRACE-X Video Analysis Pipeline: {clip_p.name} (Case ID: {case_id})")
    print("=" * 70)

    # Stage 1: Frame Extraction
    t0 = time.time()
    extracted_frames = extract_frames(str(clip_p), sampling_rate=5, max_frames=45)
    t_frames = (time.time() - t0) * 1000.0

    if not extracted_frames:
        print("[pipeline] Error: No decodable frames found in video.")
        empty_fused = FusedPlate(text="??????????", frames_used=0, unknown_count=10)
        empty_attr = VehicleAttributes(colour="unknown", colour_conf=0.0, body_type="unknown", body_conf=0.0)
        return CaseResult(
            case_id=case_id,
            clip_name=clip_p.name,
            fused_plate=empty_fused,
            attributes=empty_attr,
            candidates=[],
            decision="NO_CONFIDENT_MATCH",
            decision_reason="Video decoding failed or video contains zero decodable frames."
        )

    # Stage 2: Vehicle Detection & Tracking
    t0 = time.time()
    tracker = VehicleTracker(conf=0.25)
    all_tracks = tracker.track(extracted_frames)
    t_track = (time.time() - t0) * 1000.0

    # Pick track to analyze
    target_tid = track_id_target or (list(all_tracks.keys())[0] if all_tracks else 1)
    track_observations = all_tracks.get(target_tid, [])
    print(f"[pipeline] Selected Vehicle Track ID: {target_tid} ({len(track_observations)} frames)")

    # Stage 3: Frame Quality Selection & Deduplication
    t0 = time.time()
    best_frames = select_best_frames(track_observations, best_n=12)
    t_select = (time.time() - t0) * 1000.0
    print(f"[pipeline] Selected Best-{len(best_frames)} frames based on sharpness and plate area.")

    # Stage 4: Plate Crop & Per-Frame OCR
    t0 = time.time()
    plate_stage = PlateStage(crops_dir=str(crops_dir))
    frame_ocr_results = []
    vehicle_crops = []

    for item in best_frames:
        fno = item["frame_no"]
        ts = item["timestamp"]
        frame_img = item["frame_image"]
        bbox = item["bbox"]
        sharpness = item["sharpness"]

        # Crop vehicle bounding box
        vh, vw = frame_img.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(vw, int(bbox[2]))
        y2 = min(vh, int(bbox[3]))
        v_crop = frame_img[y1:y2, x1:x2]
        if v_crop.size == 0:
            v_crop = frame_img
        vehicle_crops.append(v_crop)

        # Run plate localization & OCR
        f_ocr = plate_stage.process_frame_crop(
            v_crop,
            track_id=target_tid,
            frame_no=fno,
            timestamp=ts,
            sharpness=sharpness
        )
        frame_ocr_results.append(f_ocr)

    # Save visual debug keyframe (bounding box overlay & vehicle/plate crops)
    if best_frames:
        top_item = best_frames[0]
        ann_img = top_item["frame_image"].copy()
        bx1, by1, bx2, by2 = [int(c) for c in top_item["bbox"]]
        cv2.rectangle(ann_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
        label = f"Target Vehicle #{target_tid} ({top_item.get('conf', 0.92)*100:.0f}%)"
        cv2.putText(ann_img, label, (bx1, max(25, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(str(case_out / "annotated_keyframe.jpg"), ann_img)

        if vehicle_crops and vehicle_crops[0].size > 0:
            cv2.imwrite(str(case_out / "vehicle_crop.jpg"), vehicle_crops[0])

        first_plate_crop = crops_dir / f"track_{target_tid}_f{top_item['frame_no']:04d}_plate.png"
        if first_plate_crop.exists():
            import shutil
            shutil.copyfile(first_plate_crop, case_out / "plate_crop.jpg")

    t_ocr = (time.time() - t0) * 1000.0

    # Stage 5: Multi-Frame Evidential Fusion
    t0 = time.time()
    fused_plate = fuse_frames(frame_ocr_results, accept_threshold=0.50, margin_threshold=0.12)
    t_fusion = (time.time() - t0) * 1000.0
    print(f"[pipeline] Fused Plate: '{fused_plate.text}' (Known: {fused_plate.known_count}, Unknown: {fused_plate.unknown_count}, Valid: {fused_plate.format_valid})")

    # Stage 6: Attribute Extraction
    t0 = time.time()
    attributes = extract_attributes(vehicle_crops, user_hints=hints)
    t_attr = (time.time() - t0) * 1000.0
    print(f"[pipeline] Attributes: Colour='{attributes.colour}' (conf={attributes.colour_conf:.2f}), Body='{attributes.body_type}' (conf={attributes.body_conf:.2f})")

    # Stage 7: Registry Matching & Ranking
    t0 = time.time()
    registry = load_registry_records(registry_csv)
    top_5, decision, decision_reason = rank_candidates(
        fused_plate,
        attributes,
        registry,
        no_match_threshold=0.52,
        margin_threshold=0.06
    )
    t_rank = (time.time() - t0) * 1000.0

    total_time_ms = (time.time() - t_start) * 1000.0
    print("\n" + "-" * 70)
    print(f"PIPELINE TIMING BREAKDOWN (Total: {total_time_ms:.1f}ms):")
    print(f"  Frame Extraction:   {t_frames:.1f}ms")
    print(f"  Vehicle Tracking:   {t_track:.1f}ms")
    print(f"  Frame Selection:    {t_select:.1f}ms")
    print(f"  Plate Crop & OCR:   {t_ocr:.1f}ms")
    print(f"  Evidential Fusion:  {t_fusion:.1f}ms")
    print(f"  Attribute Extract:  {t_attr:.1f}ms")
    print(f"  Registry Ranking:   {t_rank:.1f}ms")
    print("-" * 70)
    print(f"DECISION: {decision}")
    print(f"REASON:   {decision_reason}")
    if top_5 and decision == "MATCH_FOUND":
        print(f"TOP CANDIDATE: {top_5[0].registration_number} (Score: {top_5[0].final_score:.3f})")
    print("=" * 70 + "\n")

    case_result = CaseResult(
        case_id=case_id,
        clip_name=clip_p.name,
        fused_plate=fused_plate,
        attributes=attributes,
        candidates=top_5,
        decision=decision,
        decision_reason=decision_reason,
        thresholds_used={"no_match_threshold": 0.52, "margin_threshold": 0.06},
        weights_used={"plate": 0.65, "colour": 0.15, "body": 0.12, "visual": 0.08}
    )

    # Persist JSON result
    result_json_path = case_out / "result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        f.write(case_result.model_dump_json(indent=2))

    return case_result


def main() -> None:
    parser = argparse.ArgumentParser(description="TRACE-X Video Pipeline CLI")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--track-id", type=int, default=None, help="Vehicle track ID to inspect")
    parser.add_argument("--colour-hint", type=str, default=None, help="Investigator colour hint")
    parser.add_argument("--body-hint", type=str, default=None, help="Investigator body type hint")
    parser.add_argument("--registry", type=str, default="data/registry/vehicles.csv", help="Registry CSV")
    args = parser.parse_args()

    hints = {}
    if args.colour_hint:
        hints["colour"] = args.colour_hint
    if args.body_hint:
        hints["body_type"] = args.body_hint

    res = run_pipeline(args.video, track_id_target=args.track_id, hints=hints, registry_csv=args.registry)


if __name__ == "__main__":
    main()
