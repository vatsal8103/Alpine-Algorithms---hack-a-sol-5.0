"""TRACE-X FastAPI Application Backend.

REST API providing:
- POST /cases: Upload video and run asynchronous background investigation
- GET /cases/{id}/status: Real-time stage progress (0-100%) and logs
- GET /cases/{id}/tracks: Discovered vehicle tracks for investigator selection
- POST /cases/{id}/select_track: Switch/re-analyze targeted track ID
- GET /cases/{id}/result: Complete CaseResult JSON
- GET /cases/{id}/frames: Evidence frames, plate crops, and character votes
- GET /cases/{id}/report: Print-ready forensic HTML / PDF report
- POST /compare: Cross-camera vehicle matching
- GET /health: Health probe
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.run_pipeline import run_pipeline
from pipeline.schemas import CaseResult
from reports.report import generate_html_report

app = FastAPI(
    title="TRACE-X Forensic Video Investigation API",
    description="Explainable partial-plate vehicle identification and traceability engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("runs/cases.sqlite")
UPLOADS_DIR = Path("runs/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db() -> None:
    """Initialize SQLite case database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            clip_name TEXT NOT NULL,
            clip_path TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress REAL NOT NULL,
            error TEXT,
            hints_json TEXT,
            result_json TEXT,
            selected_track INTEGER,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


def update_case_db(
    case_id: str,
    status: str | None = None,
    stage: str | None = None,
    progress: float | None = None,
    error: str | None = None,
    result_json: str | None = None,
    selected_track: int | None = None
) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    updates = ["updated_at = ?"]
    params: list[Any] = [time.time()]

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if stage is not None:
        updates.append("stage = ?")
        params.append(stage)
    if progress is not None:
        updates.append("progress = ?")
        params.append(progress)
    if error is not None:
        updates.append("error = ?")
        params.append(error)
    if result_json is not None:
        updates.append("result_json = ?")
        params.append(result_json)
    if selected_track is not None:
        updates.append("selected_track = ?")
        params.append(selected_track)

    params.append(case_id)
    cur.execute(f"UPDATE cases SET {', '.join(updates)} WHERE case_id = ?", params)
    conn.commit()
    conn.close()


def process_case_background(case_id: str, clip_path: str, hints: dict[str, str], track_id: int | None = None) -> None:
    """Run end-to-end pipeline in background thread with live stage-level status updates."""
    try:
        def on_progress(stage: str, progress: float) -> None:
            update_case_db(case_id, status="running", stage=stage, progress=progress)

        update_case_db(case_id, status="running", stage="frame_extraction", progress=5.0)

        # Execute pipeline with dynamic progress reporting
        result = run_pipeline(
            video_path=clip_path,
            track_id_target=track_id,
            hints=hints,
            case_id_override=case_id,
            progress_cb=on_progress
        )

        update_case_db(
            case_id,
            status="completed",
            stage="completed",
            progress=100.0,
            result_json=result.model_dump_json()
        )
    except Exception as e:
        update_case_db(case_id, status="failed", stage="failed", error=str(e))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "TRACE-X API", "version": "1.0.0"}


@app.get("/cases/latest")
def get_latest_case() -> dict[str, Any]:
    """Return the most recently completed case ID and summary for instant UI restoration."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, clip_name, status, stage, progress, created_at 
        FROM cases 
        WHERE status = 'completed' AND (clip_name != 'test_clip.mp4' OR clip_name IS NULL)
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        cur.execute("""
            SELECT case_id, clip_name, status, stage, progress, created_at 
            FROM cases 
            WHERE status = 'completed' 
            ORDER BY created_at DESC LIMIT 1
        """)
        row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No completed case found")
    return {
        "case_id": row[0],
        "clip_name": row[1],
        "status": row[2],
        "stage": row[3],
        "progress": row[4],
        "created_at": row[5]
    }


@app.get("/cases")
def list_recent_cases(limit: int = 10) -> list[dict[str, Any]]:
    """List recent cases for rapid investigation review."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, clip_name, status, stage, progress, created_at 
        FROM cases 
        WHERE (clip_name != 'test_clip.mp4' OR clip_name IS NULL)
        ORDER BY created_at DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    if not rows:
        cur.execute("""
            SELECT case_id, clip_name, status, stage, progress, created_at 
            FROM cases 
            ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
    conn.close()
    return [
        {
            "case_id": r[0],
            "clip_name": r[1],
            "status": r[2],
            "stage": r[3],
            "progress": r[4],
            "created_at": r[5]
        }
        for r in rows
    ]


@app.get("/v1/models")
def list_v1_models() -> dict[str, Any]:
    """Compatibility endpoint for IDE extensions probing local port 8000."""
    return {
        "object": "list",
        "data": [
            {"id": "tracex-forensic-engine", "object": "model", "owned_by": "trace-x"}
        ]
    }


@app.post("/cases")
async def create_case(
    background_tasks: BackgroundTasks,
    clip: UploadFile = File(...),
    colour: str | None = Form(None),
    body_type: str | None = Form(None),
    partial_plate: str | None = Form(None),
    track_id: int | None = Form(None)
) -> dict[str, Any]:
    # Validate file extension
    ext = Path(clip.filename).suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported video format '{ext}'. Must be MP4, AVI, or MOV.")

    case_id = f"case-{uuid.uuid4().hex[:8]}"
    saved_clip = UPLOADS_DIR / f"{case_id}_{clip.filename}"

    with open(saved_clip, "wb") as buffer:
        shutil.copyfileobj(clip.file, buffer)

    hints = {}
    if colour:
        hints["colour"] = colour
    if body_type:
        hints["body_type"] = body_type
    if partial_plate:
        hints["partial_plate"] = partial_plate

    # Record initial case entry in SQLite
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cases (case_id, clip_name, clip_path, status, stage, progress, hints_json, selected_track, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (case_id, clip.filename, str(saved_clip), "queued", "queued", 0.0, json.dumps(hints), track_id, time.time(), time.time()))
    conn.commit()
    conn.close()

    background_tasks.add_task(process_case_background, case_id, str(saved_clip), hints, track_id)

    return {"case_id": case_id, "status": "queued", "message": "Video queued for analysis"}


@app.get("/cases/{case_id}/status")
def get_case_status(case_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT status, stage, progress, error, clip_name, created_at FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        disk_result = Path("runs/cases") / case_id / "result.json"
        if disk_result.exists():
            return {
                "case_id": case_id,
                "status": "completed",
                "stage": "completed",
                "progress": 100.0,
                "error": None,
                "clip_name": "video",
                "created_at": disk_result.stat().st_ctime
            }
        raise HTTPException(status_code=404, detail="Case ID not found")

    return {
        "case_id": case_id,
        "status": row[0],
        "stage": row[1],
        "progress": row[2],
        "error": row[3],
        "clip_name": row[4],
        "created_at": row[5]
    }


@app.get("/cases/{case_id}/tracks")
def get_case_tracks(case_id: str) -> dict[str, Any]:
    """Return all real scanned vehicles and tracks identified in video."""
    try:
        res = get_case_result(case_id)
        tracks_data = []
        for v in res.all_vehicles:
            plate_text = v.fused_plate.text if v.fused_plate else "NOT FOUND"
            if not plate_text or plate_text == "??????????":
                plate_text = "NOT FOUND"
            is_nf = (plate_text == "NOT FOUND" or v.fused_plate.mean_confidence == 0.0)
            if is_nf:
                plate_text = "NOT FOUND"
            tracks_data.append({
                "track_id": v.track_id,
                "label": f"Vehicle #{v.track_id} ({v.attributes.colour.upper()} {v.attributes.body_type.upper()})",
                "plate": plate_text,
                "conf": round(v.top_score, 3) if not is_nf else 0.0,
                "decision": v.decision if not is_nf else "NO_CONFIDENT_MATCH",
                "frames": v.frame_count,
                "thumb": v.thumbnail_url
            })
        return {"case_id": case_id, "tracks": tracks_data}
    except Exception:
        return {"case_id": case_id, "tracks": []}


@app.get("/cases/{case_id}/frames")
def get_case_frames(case_id: str) -> dict[str, Any]:
    """Return real evidence crops, frame numbers, and character votes."""
    crops_dir = Path("runs/cases") / case_id / "crops"
    crops_list = []
    if crops_dir.exists():
        for p in sorted(crops_dir.glob("*.png")):
            crops_list.append({
                "filename": p.name,
                "url": f"/runs/cases/{case_id}/crops/{p.name}"
            })
    try:
        res = get_case_result(case_id)
        positions = [p.model_dump(by_alias=True) for p in res.fused_plate.per_position]
    except Exception:
        positions = []
    return {
        "case_id": case_id,
        "crops": crops_list,
        "positions": positions
    }


@app.get("/cases/{case_id}/tracks/{track_id}/frames")
def get_track_frames(case_id: str, track_id: int) -> dict[str, Any]:
    """Return top ranked evidence frames for a specific vehicle track."""
    try:
        res = get_case_result(case_id)
        target_v = next((v for v in res.all_vehicles if v.track_id == track_id), None)
        if target_v is not None and target_v.best_frames:
            frames = [f.model_dump() for f in target_v.best_frames]
            return {"case_id": case_id, "track_id": track_id, "frames": frames}

        # Fallback to inspecting crops on disk for this track
        crops_dir = Path("runs/cases") / case_id / "crops"
        frames = []
        if crops_dir.exists():
            import re
            for p in sorted(crops_dir.glob(f"track_{track_id}_*.png")):
                m = re.search(r"f(\d+)_plate", p.name)
                fno = int(m.group(1)) if m else 0
                frames.append({
                    "frame_no": fno,
                    "timestamp": round(fno / 60.0, 2),
                    "bbox": [0, 0, 100, 100],
                    "plate_bbox": None,
                    "sharpness": 100.0,
                    "plate_area": 2500.0,
                    "plate_crop_url": f"/runs/cases/{case_id}/crops/{p.name}",
                    "raw_text": "",
                    "confidence": 0.5,
                    "chars": []
                })
        return {"case_id": case_id, "track_id": track_id, "frames": frames}
    except Exception as e:
        return {"case_id": case_id, "track_id": track_id, "frames": [], "error": str(e)}


@app.post("/cases/{case_id}/select_track")
def select_case_track(case_id: str, track_id: int, background_tasks: BackgroundTasks) -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT clip_path, hints_json FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Case ID not found")

    clip_path, hints_json = row
    hints = json.loads(hints_json) if hints_json else {}

    update_case_db(case_id, status="queued", stage="re_queued", progress=0.0, selected_track=track_id)
    background_tasks.add_task(process_case_background, case_id, clip_path, hints, track_id)

    return {"case_id": case_id, "status": "re_queued", "selected_track": track_id}


@app.post("/cases/{case_id}/vlm_clarify")
def clarify_with_vlm(case_id: str, track_id: int | None = None) -> dict[str, Any]:
    """On-demand Qwen2-VL forensic clarification with character-level evidential fusion."""
    case_dir = Path("runs/cases") / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    target_crop = None
    if track_id is not None:
        cand = case_dir / f"track_{track_id}_plate.jpg"
        if cand.exists():
            target_crop = cand
        else:
            crops = list(case_dir.glob(f"crops/track_{track_id}_*.png"))
            if crops:
                target_crop = crops[0]
            else:
                thumb = case_dir / f"track_{track_id}_thumb.jpg"
                if thumb.exists():
                    target_crop = thumb

    if target_crop is None or not target_crop.exists():
        target_crop = case_dir / "plate_crop.jpg"

    if not target_crop.exists():
        crops = list(case_dir.glob("*_plate.jpg"))
        if crops:
            target_crop = crops[0]

    if not target_crop or not target_crop.exists():
        raise HTTPException(status_code=404, detail="No plate crop image found for this vehicle track")

    import yaml
    vlm_cfg: dict[str, Any] = {}
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            vlm_cfg = yaml.safe_load(f).get("vlm", {})
    except Exception:
        pass

    from pipeline.vlm_engine import Qwen2VLLicensePlateRecognizer
    vlm = Qwen2VLLicensePlateRecognizer(
        model_name=vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        backend=vlm_cfg.get("backend", "auto"),
        lm_studio_url=vlm_cfg.get("lm_studio_url", "http://127.0.0.1:1234/v1"),
        lm_studio_model=vlm_cfg.get("lm_studio_model", None),
        load_in_4bit=vlm_cfg.get("load_in_4bit", True),
        device=vlm_cfg.get("device", "cuda:0")
    )

    vlm_result = vlm.recognize(target_crop)
    backend_used = vlm.last_backend_used
    vlm.unload_model()

    backend_label = "LM Studio (Q4_K_M GGUF)" if backend_used == "lm_studio" else "Local PyTorch (4-bit NF4)"

    from fusion.vlm_fusion import fuse_ocr_and_vlm
    from ranking.ranker import rank_candidates
    from pipeline.run_pipeline import load_registry_records

    result_path = case_dir / "result.json"
    fusion_stats: dict[str, Any] = {}
    updated_plate_text = vlm_result

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            case_obj = CaseResult(**case_data)
            registry = load_registry_records()

            target_v = None
            if track_id is not None:
                for v in case_obj.all_vehicles:
                    if v.track_id == track_id:
                        target_v = v
                        break
            if target_v is None and case_obj.all_vehicles:
                target_v = case_obj.all_vehicles[0]

            if target_v is not None and vlm_result:
                fused_plate, fusion_stats = fuse_ocr_and_vlm(target_v.fused_plate, vlm_result)
                target_v.fused_plate = fused_plate
                target_v.vlm_reading = vlm_result
                cands, dec, dec_reason = rank_candidates(fused_plate, target_v.attributes, registry)
                target_v.candidates = cands
                target_v.decision = dec
                target_v.decision_reason = dec_reason
                target_v.top_score = cands[0].final_score if cands else 0.0
                updated_plate_text = fused_plate.text

                if case_obj.all_vehicles and target_v.track_id == case_obj.all_vehicles[0].track_id:
                    case_obj.fused_plate = fused_plate
                    case_obj.vlm_reading = vlm_result
                    case_obj.vlm_consulted = True
                    case_obj.candidates = cands
                    case_obj.decision = dec
                    case_obj.decision_reason = dec_reason

                updated_json = case_obj.model_dump_json(indent=2)
                with open(result_path, "w", encoding="utf-8") as f:
                    f.write(updated_json)
                update_case_db(case_id, result_json=updated_json)
        except Exception as exc:
            print(f"[!] Fusion update warning: {exc}")

    return {
        "case_id": case_id,
        "track_id": track_id,
        "plate_crop": str(target_crop.name),
        "vlm_model": "Qwen2-VL-2B-Instruct",
        "backend": backend_label,
        "vlm_reading": vlm_result,
        "fused_plate": updated_plate_text,
        "fusion_details": fusion_stats,
        "status": "success"
    }


@app.post("/cases/{case_id}/vlm_clarify_all")
def clarify_all_with_vlm(case_id: str) -> dict[str, Any]:
    """Batch-clarify all unreadable or low-confidence vehicles in the case using Qwen2-VL."""
    case_dir = Path("runs/cases") / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    result_path = case_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Case result.json not found")

    with open(result_path, "r", encoding="utf-8") as f:
        case_data = json.load(f)
    case_obj = CaseResult(**case_data)

    from pipeline.run_pipeline import load_registry_records
    from ranking.ranker import rank_candidates
    from fusion.vlm_fusion import fuse_ocr_and_vlm
    from pipeline.vlm_engine import Qwen2VLLicensePlateRecognizer

    import yaml
    vlm_cfg: dict[str, Any] = {}
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            vlm_cfg = yaml.safe_load(f).get("vlm", {})
    except Exception:
        pass

    vlm = Qwen2VLLicensePlateRecognizer(
        model_name=vlm_cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
        backend=vlm_cfg.get("backend", "auto"),
        lm_studio_url=vlm_cfg.get("lm_studio_url", "http://127.0.0.1:1234/v1"),
        lm_studio_model=vlm_cfg.get("lm_studio_model", None),
        load_in_4bit=vlm_cfg.get("load_in_4bit", True),
        device=vlm_cfg.get("device", "cuda:0")
    )

    registry = load_registry_records()
    clarified_count = 0
    clarified_items: list[dict[str, Any]] = []

    for v in case_obj.all_vehicles:
        has_unknowns = (v.fused_plate and v.fused_plate.unknown_count > 0)
        low_conf = (v.fused_plate and v.fused_plate.mean_confidence < 0.70)
        unreadable = (v.fused_plate and (v.fused_plate.text == "??????????" or v.fused_plate.text == "NOT FOUND"))

        if not (has_unknowns or low_conf or unreadable):
            continue

        tid = v.track_id
        crop = case_dir / f"track_{tid}_plate.jpg"
        if not crop.exists():
            cands = list(case_dir.glob(f"crops/track_{tid}_*.png"))
            if cands:
                crop = cands[0]
            else:
                thumb = case_dir / f"track_{tid}_thumb.jpg"
                if thumb.exists():
                    crop = thumb
        if not crop or not crop.exists():
            continue

        try:
            v_text = vlm.recognize(crop)
            if v_text and len(v_text) >= 4:
                f_fused, stats = fuse_ocr_and_vlm(v.fused_plate, v_text)
                v.fused_plate = f_fused
                v.vlm_reading = v_text
                cands, dec, dec_reason = rank_candidates(f_fused, v.attributes, registry)
                v.candidates = cands
                v.decision = dec
                v.decision_reason = dec_reason
                v.top_score = cands[0].final_score if cands else 0.0
                if case_obj.all_vehicles and v.track_id == case_obj.all_vehicles[0].track_id:
                    case_obj.fused_plate = f_fused
                    case_obj.vlm_reading = v_text
                    case_obj.vlm_consulted = True
                    case_obj.candidates = cands
                    case_obj.decision = dec
                    case_obj.decision_reason = dec_reason
                clarified_count += 1
                clarified_items.append({
                    "track_id": tid,
                    "old_plate": stats.get("old_text", "??????????"),
                    "vlm_reading": v_text,
                    "fused_plate": f_fused.text,
                    "new_confidence": f_fused.mean_confidence,
                    "resolved": stats.get("resolved_chars", [])
                })
        except Exception as e:
            print(f"[!] Error clarifying track {tid}: {e}")

    vlm.unload_model()

    updated_json = case_obj.model_dump_json(indent=2)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(updated_json)
    update_case_db(case_id, result_json=updated_json)

    return {
        "case_id": case_id,
        "clarified_count": clarified_count,
        "clarified_vehicles": clarified_items,
        "status": "success"
    }


@app.post("/vlm/free_vram")
def free_vram_endpoint() -> dict[str, Any]:
    """Manually purge GPU VRAM, flush PyTorch CUDA caches, and return memory status."""
    from pipeline.vlm_engine import free_gpu_vram
    return free_gpu_vram()


@app.get("/vlm/status")
def get_vlm_status() -> dict[str, Any]:
    """Return current VLM configuration, LM Studio connectivity, and GPU VRAM statistics."""
    import yaml
    from pipeline.vlm_engine import is_lm_studio_available, get_lm_studio_models
    vlm_cfg: dict[str, Any] = {}
    try:
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            vlm_cfg = yaml.safe_load(f).get("vlm", {})
    except Exception:
        pass

    lm_url = vlm_cfg.get("lm_studio_url", "http://127.0.0.1:1234/v1")
    lm_online = is_lm_studio_available(lm_url)
    lm_models = get_lm_studio_models(lm_url) if lm_online else []

    vram_info: dict[str, Any] = {"cuda_available": False, "free_gb": 0.0, "total_gb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            free_b, tot_b = torch.cuda.mem_get_info()
            vram_info = {
                "cuda_available": True,
                "free_gb": round(free_b / 1e9, 2),
                "total_gb": round(tot_b / 1e9, 2),
                "device_name": torch.cuda.get_device_name(0)
            }
    except Exception:
        pass

    return {
        "backend_configured": vlm_cfg.get("backend", "auto"),
        "lm_studio_available": lm_online,
        "lm_studio_url": lm_url,
        "lm_studio_models": lm_models,
        "active_backend": "lm_studio" if lm_online else "local_4bit",
        "gpu_vram": vram_info
    }


@app.get("/cases/{case_id}/result", response_model=CaseResult)
def get_case_result(case_id: str) -> CaseResult:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT result_json, status, error FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()

    if not row:
        disk_result = Path("runs/cases") / case_id / "result.json"
        if disk_result.exists():
            with open(disk_result, "r", encoding="utf-8") as f:
                result_json = f.read()
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO cases (case_id, clip_name, clip_path, status, stage, progress, result_json, hints_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (case_id, "video.mp4", str(disk_result), "completed", "completed", 100.0, result_json, "{}", disk_result.stat().st_ctime, time.time()))
                conn.commit()
            except Exception:
                pass
            conn.close()
            return CaseResult.model_validate_json(result_json)
        conn.close()
        raise HTTPException(status_code=404, detail="Case ID not found")

    result_json, status, error = row
    if status == "running" or status == "queued":
        conn.close()
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    if status == "failed":
        conn.close()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}")
    if not result_json:
        disk_result = Path("runs/cases") / case_id / "result.json"
        if disk_result.exists():
            with open(disk_result, "r", encoding="utf-8") as f:
                result_json = f.read()
            try:
                cur.execute("UPDATE cases SET result_json = ?, status = 'completed', stage = 'completed', progress = 100.0 WHERE case_id = ?", (result_json, case_id))
                conn.commit()
            except Exception:
                pass
            conn.close()
            return CaseResult.model_validate_json(result_json)
        conn.close()
        raise HTTPException(status_code=404, detail="No result found for case")

    conn.close()
    return CaseResult.model_validate_json(result_json)


@app.get("/cases/{case_id}/report")
def get_case_report(case_id: str) -> HTMLResponse:
    res = get_case_result(case_id)
    html_path = Path("runs/cases") / case_id / "report.html"
    generate_html_report(res, str(html_path))
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


class CompareRequest(BaseModel):
    case_id_a: str
    case_id_b: str


@app.post("/compare")
def compare_cases(req: CompareRequest) -> dict[str, Any]:
    """Cross-camera matching between two vehicle observations."""
    res_a = get_case_result(req.case_id_a)
    res_b = get_case_result(req.case_id_b)

    # Check plate compatibility: any non-'?' position must match
    plate_a = res_a.fused_plate.text
    plate_b = res_b.fused_plate.text

    matching_positions = 0
    contradictions = 0
    comp_len = min(len(plate_a), len(plate_b))

    for i in range(comp_len):
        ca, cb = plate_a[i], plate_b[i]
        if ca != "?" and cb != "?":
            if ca == cb:
                matching_positions += 1
            else:
                contradictions += 1

    color_match = (res_a.attributes.colour == res_b.attributes.colour)
    body_match = (res_a.attributes.body_type == res_b.attributes.body_type)

    is_same = (contradictions == 0 and (matching_positions >= 2 or (color_match and body_match)))
    confidence = 0.88 if is_same else 0.15

    return {
        "case_id_a": req.case_id_a,
        "case_id_b": req.case_id_b,
        "same_vehicle": is_same,
        "confidence": confidence,
        "matching_plate_chars": matching_positions,
        "plate_contradictions": contradictions,
        "colour_agreement": color_match,
        "body_agreement": body_match,
        "explanation": "High visual and partial-plate agreement across camera locations." if is_same else "Evidence contradicts: different vehicle attributes or conflicting plate characters."
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    """Serve the lightweight, modern TRACE-X web interface."""
    index_file = Path("ui/web/index.html")
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
    return HTMLResponse(content="<h1>TRACE-X Web Dashboard</h1><p>ui/web/index.html not found</p>")


@app.get("/clips")
def list_demo_clips() -> list[dict[str, Any]]:
    """List available pre-recorded demo CCTV clips."""
    clips_dir = Path("data/clips")
    if not clips_dir.exists():
        return []
    res = []
    for p in sorted(clips_dir.glob("*.mp4")):
        res.append({
            "name": p.name,
            "path": str(p).replace("\\", "/"),
            "size_kb": round(p.stat().st_size / 1024, 1)
        })
    return res


class PresetCaseRequest(BaseModel):
    clip_path: str
    colour: str | None = None
    body_type: str | None = None
    partial_plate: str | None = None
    track_id: int | None = None


@app.post("/cases/preset")
def create_preset_case(req: PresetCaseRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Queue an investigation from an existing server-side clip without re-uploading bytes."""
    clip_p = Path(req.clip_path)
    if not clip_p.exists():
        raise HTTPException(status_code=404, detail=f"Clip file not found: {req.clip_path}")

    case_id = f"case-{uuid.uuid4().hex[:8]}"
    hints = {}
    if req.colour:
        hints["colour"] = req.colour
    if req.body_type:
        hints["body_type"] = req.body_type
    if req.partial_plate:
        hints["partial_plate"] = req.partial_plate

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO cases (case_id, clip_name, clip_path, status, stage, progress, hints_json, selected_track, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (case_id, clip_p.name, str(clip_p), "queued", "queued", 0.0, json.dumps(hints), req.track_id, time.time(), time.time()))
    conn.commit()
    conn.close()

    background_tasks.add_task(process_case_background, case_id, str(clip_p), hints, req.track_id)
    return {"case_id": case_id, "status": "queued", "message": "Video queued for analysis"}


@app.get("/registry")
def get_sample_registry(limit: int = 500) -> list[dict[str, Any]]:
    """Return records from sample vehicle registry for explorer."""
    import csv
    reg_csv = Path("data/registry/vehicles.csv")
    if not reg_csv.exists():
        return []
    records = []
    with open(reg_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            records.append(dict(row))
    return records


@app.post("/compare/demo")
def compare_demo() -> dict[str, Any]:
    """Demo response for cross-camera re-identification between Cam A and Cam B."""
    return {
        "case_id_a": "camA-target",
        "case_id_b": "camB-target",
        "same_vehicle": True,
        "confidence": 0.89,
        "matching_plate_chars": 9,
        "plate_contradictions": 0,
        "colour_agreement": True,
        "body_agreement": True,
        "explanation": "High visual and partial-plate agreement across camera locations: JH05AB6799 (White Swift). Travel time (5m 26s over 2.3km) is physically plausible (25.4 km/h)."
    }


# Mount runs and data directories
runs_p = Path("runs")
runs_p.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(runs_p)), name="runs")

data_p = Path("data")
data_p.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(data_p)), name="data")

