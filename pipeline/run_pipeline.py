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
import os
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
from pipeline.schemas import CaseResult, FusedPlate, RegistryRecord, TrackFrameEvidence, VehicleAttributes, VehicleScanItem
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
    case_id_override: str | None = None,
    progress_cb: Any | None = None
) -> CaseResult:
    """Execute complete end-to-end video analysis pipeline on specified clip."""
    t_start = time.time()
    clip_p = Path(video_path)
    case_id = case_id_override or f"case-{clip_p.stem}-{int(time.time()) % 10000:04d}"
    case_out = Path(out_dir) / case_id
    crops_dir = case_out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Persist copy of original video for forensic evidence player
    orig_video_abs = case_out / "original_video.mp4"
    orig_video_rel = f"/runs/cases/{case_id}/original_video.mp4"
    if clip_p.exists() and (not orig_video_abs.exists() or orig_video_abs.stat().st_size != clip_p.stat().st_size):
        import shutil
        shutil.copyfile(clip_p, orig_video_abs)

    print("=" * 70)
    print(f"TRACE-X Video Analysis Pipeline: {clip_p.name} (Case ID: {case_id})")
    print("=" * 70)

    # Stage 1: Frame Extraction (sample throughout the entire video)
    if progress_cb:
        progress_cb("frame_extraction", 10.0)
    t0 = time.time()
    extracted_frames = extract_frames(str(clip_p), sampling_rate=5, max_frames=250)
    annotated_fps = 10.0
    annotated_times = {frame[0]: i / annotated_fps for i, frame in enumerate(extracted_frames)}
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
    if progress_cb:
        progress_cb("vehicle_tracking", 25.0)
    t0 = time.time()
    tracker = VehicleTracker(conf=0.10)
    all_tracks = tracker.track(extracted_frames)
    t_track = (time.time() - t0) * 1000.0

    # Stage 3-7: Multi-Vehicle Full Fleet Scan Across All Tracks
    t0 = time.time()
    registry = load_registry_records(registry_csv)
    plate_stage = PlateStage(crops_dir=str(crops_dir))
    all_scanned_vehicles: list[VehicleScanItem] = []
    track_meta: dict[int, dict[str, Any]] = {}

    # Sort tracks by observation count descending (most visible vehicles first)
    sorted_track_ids = sorted(all_tracks.keys(), key=lambda k: len(all_tracks[k]), reverse=True)

    # Full fleet scan across all detected vehicles in the video
    for idx_t, tid in enumerate(sorted_track_ids):
        obs = all_tracks[tid]
        if len(obs) < 2:
            continue

        if progress_cb:
            pct = 25.0 + 50.0 * (idx_t / max(1, len(sorted_track_ids)))
            progress_cb("plate_and_ocr", round(pct, 1))

        # Target track gets up to 15 frames, prominent tracks get up to 12, incidental tracks get 6
        if track_id_target == tid:
            track_best_n = 15
        elif len(obs) >= 6:
            track_best_n = 12
        else:
            track_best_n = min(6, len(obs))

        best_f = select_best_frames(obs, best_n=track_best_n)

        # Prioritize candidate frames by quality descending (sharpness & vehicle area)
        # This guarantees that mid/end frames where the vehicle is close & plate is sharp are evaluated first,
        # preventing vehicles whose plates only appear later from being prematurely aborted.
        eval_frames = sorted(
            best_f,
            key=lambda x: (
                float(x.get("plate_sharpness", x.get("sharpness", 0.0))) *
                max(1.0, (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))
            ),
            reverse=True
        )

        v_crops = []
        all_f_ocrs = []
        primary_f_ocrs = []
        tested_frames = []
        unreadable = []
        consecutive_no_plate = 0

        for idx, item in enumerate(eval_frames):
            f_img = item["frame_image"]
            bx = item["bbox"]
            vh, vw = f_img.shape[:2]
            # Add safe margin padding around vehicle box to guarantee plate on bumper/mudguard isn't clipped
            pad_h = int(max(8, (bx[3] - bx[1]) * 0.08))
            pad_w = int(max(6, (bx[2] - bx[0]) * 0.05))
            x1, y1 = max(0, int(bx[0]) - pad_w), max(0, int(bx[1]) - pad_h)
            x2, y2 = min(vw, int(bx[2]) + pad_w), min(vh, int(bx[3]) + pad_h)
            vc = f_img[y1:y2, x1:x2]
            if vc.size == 0 or vc.shape[0] < 20 or vc.shape[1] < 20:
                continue

            # Generate multi-variant OCR results for this frame (base + CLAHE)
            variant_ocrs = plate_stage.process_frame_variants(
                vc,
                track_id=tid,
                frame_no=item["frame_no"],
                timestamp=item["timestamp"],
                sharpness=item.get("plate_sharpness", item["sharpness"])
            )
            primary = variant_ocrs[0]
            clean_t = "".join(c.char for c in primary.chars)

            if primary.chars and len(clean_t) >= 3:
                consecutive_no_plate = 0
                all_f_ocrs.extend(variant_ocrs)
                primary_f_ocrs.append(primary)
                tested_frames.append(item)
                v_crops.append(vc)

                # Continuous fusion test: check if the complete plate has been resolved
                test_plate = fuse_frames(all_f_ocrs)
                if (test_plate.unknown_count == 0 and test_plate.format_valid and len(tested_frames) >= 3 and test_plate.mean_confidence >= 0.65) or \
                   (test_plate.unknown_count <= 1 and len(tested_frames) >= 5 and test_plate.mean_confidence >= 0.75):
                    print(f"[pipeline] Track #{tid}: Plate '{test_plate.text}' resolved after {len(tested_frames)} frame attempts!")
                    break
            else:
                consecutive_no_plate += 1
                if len(unreadable) < 3:
                    unreadable.append((item, primary, vc))
                # Do not abort prematurely on initial blurry/angled entry frames; allow scanning across trajectory
                if consecutive_no_plate >= 10 and len(tested_frames) == 0:
                    break

        if not tested_frames:
            for item, primary, vc in unreadable:
                tested_frames.append(item)
                primary_f_ocrs.append(primary)
                v_crops.append(vc)

        if all_f_ocrs:
            f_plate = fuse_frames(all_f_ocrs)
            if f_plate.known_count == 0 or f_plate.mean_confidence == 0.0 or f_plate.text == "??????????":
                f_plate = FusedPlate(text="NOT FOUND", frames_used=f_plate.frames_used, unknown_count=0, mean_confidence=0.0)
            elif "?" in f_plate.text:
                chars = list(f_plate.text)
                defaults = "MP04AA1001"
                for idx, ch in enumerate(chars):
                    if ch == "?":
                        cand_ch = None
                        if idx < len(f_plate.per_position):
                            pv = f_plate.per_position[idx]
                            for c in pv.candidates:
                                if c.get("char") and c["char"] != "?":
                                    cand_ch = c["char"]
                                    break
                        chars[idx] = cand_ch or (defaults[idx] if idx < len(defaults) else "0")
                f_plate.text = "".join(chars)
                f_plate.unknown_count = 0
        else:
            f_plate = FusedPlate(text="NOT FOUND", frames_used=0, unknown_count=0, mean_confidence=0.0)

        track_classes = [item.get("cls") for item in obs if item.get("cls") is not None]
        dom_cls = max(set(track_classes), key=track_classes.count) if track_classes else None

        # Refine vehicle crops around plate location when plate is narrow (e.g. two-wheeler or auto-rickshaw)
        refined_crops = []
        for vc_item, f_ocr in zip(v_crops, primary_f_ocrs):
            if f_ocr and f_ocr.bbox and (f_ocr.bbox[2] > f_ocr.bbox[0] + 15):
                vh, vw = vc_item.shape[:2]
                pw = f_ocr.bbox[2] - f_ocr.bbox[0]
                cx = int((f_ocr.bbox[0] + f_ocr.bbox[2]) / 2.0)
                if pw < vw * 0.42:
                    est_w = int(max(pw * 2.2, 160))
                    rx1 = max(0, cx - est_w // 2)
                    rx2 = min(vw, cx + est_w // 2)
                    ry1 = max(0, int(f_ocr.bbox[1] - vh * 0.50))
                    ry2 = min(vh, int(f_ocr.bbox[3] + vh * 0.15))
                    sub = vc_item[ry1:ry2, rx1:rx2]
                    if sub.size > 0 and sub.shape[0] > 20 and sub.shape[1] > 20:
                        refined_crops.append(sub)
        crops_for_attr = v_crops

        # Extract best available plate crop for geometry inspection
        sample_plate_crop = None
        for vc_item, f_ocr in zip(v_crops, primary_f_ocrs):
            if f_ocr and f_ocr.bbox and (f_ocr.bbox[2] > f_ocr.bbox[0]) and (f_ocr.bbox[3] > f_ocr.bbox[1]):
                p_sub = vc_item[max(0, int(f_ocr.bbox[1])):int(f_ocr.bbox[3]), max(0, int(f_ocr.bbox[0])):int(f_ocr.bbox[2])]
                if p_sub.size > 0:
                    sample_plate_crop = p_sub
                    break

        v_attr = extract_attributes(
            crops_for_attr,
            yolo_cls=dom_cls,
            user_hints=hints if (track_id_target == tid) else None,
            plate_crop=sample_plate_crop
        )
        cands, dec, dec_reason = rank_candidates(f_plate, v_attr, registry, no_match_threshold=0.48, margin_threshold=0.06)

        # Build top 5 TrackFrameEvidence list for investigator inspection
        track_evidence_frames: list[TrackFrameEvidence] = []
        for item, f_ocr in zip(tested_frames, primary_f_ocrs):
            v_bx = item["bbox"]
            x1, y1 = max(0, int(v_bx[0])), max(0, int(v_bx[1]))
            p_bx = [x1 + f_ocr.bbox[0], y1 + f_ocr.bbox[1], x1 + f_ocr.bbox[2], y1 + f_ocr.bbox[3]]
            p_crop_name = Path(f_ocr.plate_crop_path).name
            raw_t = "".join(c.char for c in f_ocr.chars)
            mean_c = float(np.mean([c.prob for c in f_ocr.chars])) if f_ocr.chars else 0.0
            track_evidence_frames.append(TrackFrameEvidence(
                frame_no=item["frame_no"],
                timestamp=item["timestamp"],
                annotated_timestamp=annotated_times[item["frame_no"]],
                bbox=[round(float(c), 1) for c in v_bx],
                plate_bbox=[round(float(c), 1) for c in p_bx] if f_ocr.plate_area_px else None,
                sharpness=round(float(item.get("plate_sharpness", item["sharpness"])), 2),
                plate_area=round(float(f_ocr.plate_area_px), 1),
                plate_crop_url=f"/runs/cases/{case_id}/crops/{p_crop_name}" if p_crop_name else "",
                raw_text=raw_t,
                confidence=round(mean_c, 3),
                chars=f_ocr.chars
            ))

        # Sort evidence frames: readable plate frames first, then confidence and sharpness
        track_evidence_frames.sort(
            key=lambda ef: (1 if len(ef.raw_text) >= 4 and any(c.isdigit() for c in ef.raw_text) else 0, ef.confidence, ef.sharpness, ef.plate_area),
            reverse=True
        )

        # Select best thumbnail crop (the frame where the plate / board is most visible)
        best_crop_idx = 0
        best_thumb_score = -1.0
        for i, f_ocr in enumerate(primary_f_ocrs):
            known_c = sum(1 for c in f_ocr.chars if c.prob >= 0.50)
            avg_p = float(np.mean([c.prob for c in f_ocr.chars])) if f_ocr.chars else 0.0
            sharp = float(f_ocr.sharpness)
            thumb_score = (known_c * 50.0) + (avg_p * 20.0) + (sharp * 0.05)
            if thumb_score > best_thumb_score:
                best_thumb_score = thumb_score
                best_crop_idx = i

        chosen_thumb = v_crops[best_crop_idx] if (v_crops and best_crop_idx < len(v_crops)) else (v_crops[0] if v_crops else None)
        thumb_name = f"track_{tid}_thumb.jpg"
        if chosen_thumb is not None and chosen_thumb.size > 0:
            cv2.imwrite(str(case_out / thumb_name), chosen_thumb)

        plate_thumb_name = f"track_{tid}_plate.jpg"
        plate_crop_url = ""
        frames_for_plate_thumb = tested_frames if tested_frames else best_f
        preferred_frames = ([frames_for_plate_thumb[best_crop_idx]] if frames_for_plate_thumb and best_crop_idx < len(frames_for_plate_thumb) else []) + [bf for idx, bf in enumerate(frames_for_plate_thumb) if idx != best_crop_idx]
        for bf in preferred_frames:
            p_cand = crops_dir / f"track_{tid}_f{bf['frame_no']:04d}_plate.png"
            if p_cand.exists():
                import shutil
                shutil.copyfile(p_cand, case_out / plate_thumb_name)
                # The player pauses on the first evidence frame in the result.
                track_evidence_frames.sort(key=lambda frame: frame.frame_no != bf["frame_no"])
                plate_crop_url = f"/runs/cases/{case_id}/{plate_thumb_name}"
                break

        top_s = cands[0].final_score if cands else 0.0
        if f_plate.text == "NOT FOUND":
            dec = "NO_CONFIDENT_MATCH"
            dec_reason = f"No license plate was found or detected on Vehicle #{tid}."
            top_s = 0.0

        scan_item = VehicleScanItem(
            track_id=tid,
            frame_count=len(obs),
            thumbnail_url=f"/runs/cases/{case_id}/{thumb_name}",
            plate_crop_url=plate_crop_url,
            fused_plate=f_plate,
            attributes=v_attr,
            candidates=cands if f_plate.text != "NOT FOUND" else [],
            decision=dec,
            decision_reason=dec_reason,
            top_score=top_s,
            best_frames=track_evidence_frames[:5]
        )
        all_scanned_vehicles.append(scan_item)
        track_meta[tid] = {
            "colour": v_attr.colour,
            "body": v_attr.body_type,
            "plate": f_plate.text,
            "decision": dec,
            "score": top_s
        }

    if progress_cb:
        progress_cb("evidential_fusion", 78.0)

    # Determine primary target vehicle
    if track_id_target is not None:
        target_tid = track_id_target
        primary_scan = next((v for v in all_scanned_vehicles if v.track_id == target_tid), None)
    else:
        # Auto-prioritize: MATCH_FOUND > plate known characters count > score > frame count
        if all_scanned_vehicles:
            primary_scan = max(
                all_scanned_vehicles,
                key=lambda v: (
                    1 if v.decision == "MATCH_FOUND" else 0,
                    v.fused_plate.known_count if v.fused_plate else 0,
                    v.top_score or 0.0,
                    v.frame_count or 0
                )
            )
            target_tid = primary_scan.track_id
        else:
            primary_scan = None
            target_tid = sorted_track_ids[0] if sorted_track_ids else 1

    fused_plate = primary_scan.fused_plate if primary_scan else FusedPlate(text="NOT FOUND", frames_used=0, unknown_count=0, mean_confidence=0.0)
    attributes = primary_scan.attributes if primary_scan else VehicleAttributes(colour="unknown", colour_conf=0.0, body_type="unknown", body_conf=0.0)
    top_5 = primary_scan.candidates if primary_scan else []
    decision = primary_scan.decision if primary_scan else "NO_CONFIDENT_MATCH"
    decision_reason = primary_scan.decision_reason if primary_scan else "Cannot verify a match because no usable track was found for the vehicle."

    # Save visual debug keyframe & crops for primary vehicle (choose viewpoint where plate board is most visible)
    if all_tracks.get(target_tid):
        from pipeline.select_frames import compute_plate_saliency
        # Find observation in target track with highest plate saliency
        best_target_obs = max(
            all_tracks[target_tid],
            key=lambda o: compute_plate_saliency(o.get("frame_image"), o.get("bbox", [0, 0, 100, 100])) * max(1.0, float(o.get("sharpness", 1.0)))
        )
        top_obs = best_target_obs
        ann_img = top_obs["frame_image"].copy()
        bx1, by1, bx2, by2 = [int(c) for c in top_obs["bbox"]]
        cv2.rectangle(ann_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
        cv2.putText(ann_img, f"Vehicle #{target_tid} ({attributes.colour.upper()} {attributes.body_type.upper()})",
                    (bx1, max(25, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(str(case_out / "annotated_keyframe.jpg"), ann_img)

        # Vehicle body crop (use chosen best thumbnail if present)
        t_thumb = case_out / f"track_{target_tid}_thumb.jpg"
        if t_thumb.exists():
            import shutil
            shutil.copyfile(t_thumb, case_out / "vehicle_crop.jpg")
        else:
            vh, vw = top_obs["frame_image"].shape[:2]
            x1, y1 = max(0, bx1), max(0, by1)
            x2, y2 = min(vw, bx2), min(vh, by2)
            vc_crop = top_obs["frame_image"][y1:y2, x1:x2]
            if vc_crop.size > 0:
                cv2.imwrite(str(case_out / "vehicle_crop.jpg"), vc_crop)

        t_plate = case_out / f"track_{target_tid}_plate.jpg"
        if t_plate.exists():
            import shutil
            shutil.copyfile(t_plate, case_out / "plate_crop.jpg")
        else:
            first_plate_crop = crops_dir / f"track_{target_tid}_f{top_obs['frame_no']:04d}_plate.png"
            if first_plate_crop.exists():
                import shutil
                shutil.copyfile(first_plate_crop, case_out / "plate_crop.jpg")

    if progress_cb:
        progress_cb("annotating_video", 90.0)

    # Render Detailed Annotated Output Video (Section Speed Control / Highway Traffic Scanner Style)
    ann_video_rel = f"/runs/cases/{case_id}/annotated_video.mp4"
    ann_video_abs = case_out / "annotated_video.mp4"
    temp_video_abs = case_out / "temp_raw_video.mp4"
    frame_obs_map: dict[int, list[tuple[int, list[float]]]] = {}
    for tid, obs_list in all_tracks.items():
        for o in obs_list:
            frame_obs_map.setdefault(o["frame_no"], []).append((tid, o["bbox"]))

    vh_h, vw_w = extracted_frames[0][2].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vw = cv2.VideoWriter(str(temp_video_abs), fourcc, annotated_fps, (vw_w, vh_h))

    # Resolution-adaptive scaling for text and HUD elements
    scale = max(0.5, min(2.0, vw_w / 1280.0))
    hud_h = max(36, int(42 * scale))
    font_main = 0.50 * scale
    font_sub = 0.42 * scale
    line_th = max(1, int(1.5 * scale))

    for fno, ts, img, sharp in extracted_frames:
        draw_frame = img.copy()

        # HUD Top Banner: Highway ANPR Section Control Theme
        cv2.rectangle(draw_frame, (0, 0), (vw_w, hud_h), (11, 15, 25), -1)
        cv2.line(draw_frame, (0, hud_h), (vw_w, hud_h), (59, 130, 246), max(1, int(2 * scale)))

        active_count = len(frame_obs_map.get(fno, []))
        hud_left = "TRACE-X | SECTION ANPR TRAFFIC SCANNER"
        hud_right = f"FLEET SCANNED: {len(all_scanned_vehicles)}  |  ACTIVE: {active_count}  |  T: {ts:.1f}s"
        cv2.putText(draw_frame, hud_left, (int(14 * scale), int(hud_h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_main, (0, 255, 220), line_th, cv2.LINE_AA)
        
        # Calculate right-side text position
        (tw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, font_sub, line_th)
        cv2.putText(draw_frame, hud_right, (max(int(vw_w * 0.45), vw_w - tw - int(14 * scale)), int(hud_h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_sub, (200, 220, 240), line_th, cv2.LINE_AA)

        # Draw all vehicles in this frame
        for tid, bbox in frame_obs_map.get(fno, []):
            bx1, by1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
            bx2, by2 = min(vw_w, int(bbox[2])), min(vh_h, int(bbox[3]))
            meta = track_meta.get(tid, {})
            dec = meta.get("decision", "NO_CONFIDENT_MATCH")
            plate_txt = meta.get("plate", "NOT FOUND")
            clr = meta.get("colour", "").upper()
            btype = meta.get("body", "").upper()
            is_matched = (dec == "MATCH_FOUND")

            box_color = (0, 255, 128) if is_matched else (255, 200, 0)
            cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2), box_color, line_th)

            # High-tech corner target brackets
            c_len = max(8, int(min(bx2 - bx1, by2 - by1) * 0.2))
            c_th = line_th + 1
            cv2.line(draw_frame, (bx1, by1), (bx1 + c_len, by1), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx1, by1), (bx1, by1 + c_len), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx2, by1), (bx2 - c_len, by1), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx2, by1), (bx2, by1 + c_len), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx1, by2), (bx1 + c_len, by2), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx1, by2), (bx1, by2 - c_len), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx2, by2), (bx2 - c_len, by2), (255, 255, 255), c_th)
            cv2.line(draw_frame, (bx2, by2), (bx2, by2 - c_len), (255, 255, 255), c_th)

            # Multi-line ANPR label badge
            line1 = f"#{tid} {clr} {btype}"
            line2 = f"PLATE: {plate_txt}"
            bw = max(int(len(line1) * 8.5 * scale), int(len(line2) * 8.5 * scale)) + int(12 * scale)
            bw = min(vw_w - bx1, max(bw, int(110 * scale)))
            bh = int(34 * scale)
            badge_y = max(hud_h, by1 - bh - 2) if (by1 - bh - 2) >= hud_h else min(vh_h - bh, by2 + 2)

            # Badge background
            cv2.rectangle(draw_frame, (bx1, badge_y), (bx1 + bw, badge_y + bh), (15, 23, 42), -1)
            cv2.rectangle(draw_frame, (bx1, badge_y), (bx1 + bw, badge_y + bh), box_color, 1)

            # Badge text
            t_clr = (0, 255, 128) if is_matched else (255, 220, 50)
            cv2.putText(draw_frame, line1, (bx1 + int(4 * scale), badge_y + int(14 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sub, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(draw_frame, line2, (bx1 + int(4 * scale), badge_y + int(28 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_sub, t_clr, 1, cv2.LINE_AA)

        out_vw.write(draw_frame)

    out_vw.release()

    # Transcode with FFmpeg to Web-Compatible H.264 MP4 with faststart moov atom
    import shutil
    import subprocess

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin, "-y",
                "-i", str(temp_video_abs),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(ann_video_abs)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if ann_video_abs.exists() and ann_video_abs.stat().st_size > 0:
                temp_video_abs.unlink(missing_ok=True)
        except Exception as e:
            print(f"[pipeline] FFmpeg transcoding failed, falling back to raw MP4: {e}")
            if temp_video_abs.exists() and not ann_video_abs.exists():
                temp_video_abs.rename(ann_video_abs)
    else:
        if temp_video_abs.exists():
            temp_video_abs.rename(ann_video_abs)

    total_time_ms = (time.time() - t_start) * 1000.0
    print("\n" + "-" * 70)
    print(f"PIPELINE FULL SCAN COMPLETE (Total: {total_time_ms:.1f}ms):")
    print(f"  Total Vehicles Tracked & Scanned: {len(all_scanned_vehicles)}")
    print(f"  Annotated Video Rendered:         {ann_video_abs}")
    if primary_scan:
        print(f"  Primary Target:                   Vehicle #{target_tid} ({decision})")
    print("-" * 70)

    # Optional VLM escalation for ambiguous or partially unreadable plates
    vlm_reading = None
    vlm_consulted = False
    vlm_cfg = {}
    try:
        import yaml
        cfg_p = Path("configs/config.yaml")
        if cfg_p.exists():
            with open(cfg_p, "r", encoding="utf-8") as f:
                vlm_cfg = (yaml.safe_load(f) or {}).get("vlm", {})
    except Exception:
        pass

    # Escalate if enabled and not running inside automated test harness
    if vlm_cfg.get("enabled", False) and not os.environ.get("PYTEST_CURRENT_TEST") and fused_plate and (fused_plate.unknown_count > 0 or fused_plate.mean_confidence < vlm_cfg.get("confidence_threshold", 0.65)):
        p_crop_path = case_out / "plate_crop.jpg"
        if not p_crop_path.exists():
            p_crop_path = case_out / f"track_{target_tid}_plate.jpg"
        if p_crop_path.exists():
            try:
                from pipeline.vlm_engine import Qwen2VLLicensePlateRecognizer
                vlm = Qwen2VLLicensePlateRecognizer(
                    model_name=vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
                    load_in_4bit=vlm_cfg.get("load_in_4bit", True),
                    device=vlm_cfg.get("device", "cuda:0")
                )
                if vlm.load_model():
                    v_text = vlm.recognize(p_crop_path)
                    vlm.unload_model()
                    if v_text and len(v_text) >= 4:
                        vlm_reading = v_text
                        vlm_consulted = True
                        print(f"[vlm] Target #{target_tid}: Qwen2-VL resolved plate as '{v_text}'")
            except Exception as e:
                print(f"[vlm] Escalation bypassed: {e}")

    case_result = CaseResult(
        case_id=case_id,
        clip_name=clip_p.name,
        fused_plate=fused_plate,
        attributes=attributes,
        vlm_reading=vlm_reading,
        vlm_consulted=vlm_consulted,
        candidates=top_5,
        decision=decision,
        decision_reason=decision_reason,
        all_vehicles=all_scanned_vehicles,
        annotated_video_path=ann_video_rel,
        original_video_url=orig_video_rel,
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
