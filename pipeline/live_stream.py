"""TRACE-X Live Video & RTSP Stream Real-Time Forensic Engine.

Supports:
- USB / Built-in Webcams (device index 0, 1, 2)
- Network IP Cameras / RTSP streams (e.g. rtsp://admin:pass@ip:554/stream)
- Video file simulation in continuous loop

Features:
- Real-time vehicle detection and ByteTrack tracking
- Rolling buffer of last N frame plate crops
- Continuous multi-frame evidential consensus fusion
- Live HUD display showing tracked vehicle, consensus plate, and top candidate
"""

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion.fuser import fuse_frames
from pipeline.attributes import extract_attributes
from pipeline.detect_track import VehicleTracker
from pipeline.plate_stage import PlateStage
from pipeline.run_pipeline import load_registry_records
from pipeline.schemas import FrameOCRResult
from ranking.ranker import rank_candidates


def run_live_stream(
    source: str | int = 0,
    registry_csv: str = "data/registry/vehicles.csv",
    buffer_size: int = 15,
    device: str = "auto",
    display: bool = True
) -> None:
    """Run real-time forensic plate recovery on live stream or camera."""
    # Resolve source: integer camera index or string RTSP/file URL
    if isinstance(source, str) and source.isdigit():
        source_val: Any = int(source)
    else:
        source_val = source

    print("=" * 70)
    print(f"TRACE-X Live Stream Processing: Source = {source_val}")
    print("Press 'q' in video window to exit.")
    print("=" * 70)

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[live] Error: Unable to open video source '{source_val}'")
        return

    tracker = VehicleTracker(device=device)
    plate_stage = PlateStage()
    registry = load_registry_records(registry_csv)

    # Rolling buffer of per-track recent FrameOCRResults
    track_buffers: dict[int, collections.deque[FrameOCRResult]] = collections.defaultdict(
        lambda: collections.deque(maxlen=buffer_size)
    )
    track_crops: dict[int, collections.deque[np.ndarray]] = collections.defaultdict(
        lambda: collections.deque(maxlen=buffer_size)
    )

    frame_idx = 0
    fps_frames = 0
    t_last = time.time()
    fps_display = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If video file reached end, loop
            if isinstance(source_val, str) and Path(source_val).exists():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        frame_idx += 1
        fps_frames += 1
        t_curr = time.time()
        if t_curr - t_last >= 1.0:
            fps_display = fps_frames / (t_curr - t_last)
            fps_frames = 0
            t_last = t_curr

        h, w = frame.shape[:2]

        # Vehicle tracking on frame
        frame_obs = [(frame_idx, t_curr, frame, 50.0)]
        tracks = tracker.track(frame_obs)

        top_match_text = "NO TARGET"
        fused_plate_text = "----------"
        decision_label = "MONITORING"
        decision_color = (180, 180, 180)

        for tid, obs_list in tracks.items():
            if not obs_list:
                continue
            obs = obs_list[0]
            bbox = obs["bbox"]
            x1, y1, x2, y2 = [int(c) for c in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            v_crop = frame[y1:y2, x1:x2]
            if v_crop.size == 0:
                continue

            track_crops[tid].append(v_crop)

            # OCR on current frame crop
            f_ocr = plate_stage.process_frame_crop(
                v_crop,
                track_id=tid,
                frame_no=frame_idx,
                timestamp=t_curr,
                sharpness=50.0
            )
            track_buffers[tid].append(f_ocr)

            # Evidential fusion over rolling buffer
            fused = fuse_frames(list(track_buffers[tid]))
            fused_plate_text = fused.text

            # Extract attributes & Rank against registry
            attrs = extract_attributes(list(track_crops[tid]))
            top_5, decision, reason = rank_candidates(fused, attrs, registry, no_match_threshold=0.52)

            if decision == "MATCH_FOUND" and top_5:
                decision_label = "MATCH IDENTIFIED"
                decision_color = (0, 220, 80) # Green
                top_match_text = f"{top_5[0].registration_number} ({top_5[0].vehicle_id}) {top_5[0].final_score*100:.0f}%"
            else:
                decision_label = "NO CONFIDENT MATCH"
                decision_color = (40, 40, 230) # Red
                top_match_text = "NO CONFIDENT MATCH"

            # Draw vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), decision_color, 2)
            cv2.putText(frame, f"Track #{tid} [{decision_label}]", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, decision_color, 2)
            break # Focus primary track

        # HUD Overlay Banner
        cv2.rectangle(frame, (0, 0), (w, 75), (15, 23, 42), -1)
        cv2.putText(frame, f"TRACE-X LIVE FORENSIC ENGINE | FPS: {fps_display:.1f}", (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 2)

        # Fused plate tiles string
        cv2.putText(frame, f"FUSED PLATE: {fused_plate_text}", (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Candidate match text
        cv2.putText(frame, f"MATCH: {top_match_text}", (int(w * 0.50), 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, decision_color, 2)

        if display:
            cv2.imshow("TRACE-X Live Forensic Video Stream", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[live] User terminated live stream.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/clips/jh05ab6799_camA.mp4"
    run_live_stream(source=src)
