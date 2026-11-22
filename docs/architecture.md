# TRACE-X: End-to-End System Architecture

**Hackasol 5.0 — Problem AIML-01: Vehicle Identification and Traceability from CCTV Footage**

---

## 1. Architectural Overview & Core Guarantees

TRACE-X is a forensic decision-support system designed to identify and rank candidate vehicles from degraded CCTV video footage.

### Core Engineering Principles
1. **Never Invent Characters**: Characters that cannot be deciphered with high posterior confidence and margin are marked as unknown (`?`). The system never hallucinates characters to satisfy search queries.
2. **Probability-Carrying Pipeline**: Every character in every frame carries a calibrated probability and top alternatives. Multi-frame fusion aggregates these into calibrated posteriors.
3. **Fail-Safe Abstention**: If the available evidence is ambiguous, contradictory, or below the threshold, TRACE-X explicitly returns `NO_CONFIDENT_MATCH` with an explainable plain-English rationale.
4. **Decision-Support Wording**: In all interfaces and reports, the system states: *"Highest-ranked candidate based on the evidence"*. It never claims definitive identification.

---

## 2. Text Data-Flow Diagram

```
+-------------------------------------------------------------------------------------------------+
|                                     TRACE-X PIPELINE FLOW                                       |
+-------------------------------------------------------------------------------------------------+

       [ Raw CCTV Video Clip (.mp4 / .avi) ]
                         |
                         v
     +---------------------------------------+
     | 1. Frame Extraction & Sharpness Calc  | ---> Filters out blurry frames
     |    (Laplacian Var >= min_sharpness)   |      Extracts timestamps & frame metadata
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 2. Vehicle Detection & Tracking       | ---> Pretrained YOLOv8s (COCO)
     |    (ByteTrack assigns stable track_id)|      Produces stable vehicle trajectories
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 3. License Plate Detection & Cropping | ---> Fine-tuned Indian Plate YOLOv8s
     |    (Expands bbox margin + filters)    |      Crops plate ROIs + records plate_area_px
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 4. Per-Frame CTC OCR & Calibration    | ---> PaddleOCR CTC Head + Temp Scaling
     |    (Extracts char probs + top-3 alts) |      Outputs FrameOCRResult
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 5. Multi-Frame Evidential Fusion      | ---> Frame weighting (sharpness * area * ocr_conf)
     |    (Needleman-Wunsch Edit Dist Align) |      Confusion-pair penalty reduction
     |    (Indian Plate Grammar Priors)      |      Per-position posterior & margin check
     +---------------------------------------+
                         |
           +-------------+-------------+
           |                           |
           v                           v
+----------------------+   +-----------------------+
| FusedPlate Object    |   | 6. Attribute Extract  | ---> Vehicle body ROI segmentation
| Text: "JH05AB?799"   |   |    - Colour (K-Means) |      Excludes windows, wheels & shadows
| Per-position votes   |   |    - Body Type (Clf)  |      Aggregated distribution across frames
+----------------------+   +-----------------------+
           \                           /
            \                         /
             v                       v
     +---------------------------------------+
     | 7. Sample Vehicle Registry Matching   | ---> 500+ Indian vehicle records
     |    (Pre-filters impossible lengths)   |      Evaluates plates, colour, body type
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 8. Original Multi-Criteria Ranking    | ---> plate_score (positional + confusable penalty)
     |    (Normalized weighted sum)          |      colour_score (similarity matrix)
     |    (Calibrated final score)           |      body_score (partial-credit matrix)
     +---------------------------------------+      visual_score (appearance embedding)
                         |
                         v
     +---------------------------------------+
     | 9. Abstention & No-Match Engine       | ---> Checks: min_score, margin, contradiction
     |    (Decision: MATCH / NO_MATCH)       |      Outputs plain-English decision_reason
     +---------------------------------------+
                         |
                         v
     +---------------------------------------+
     | 10. Audit Report & Investigator UI    | ---> Dark forensic dashboard (React / Streamlit)
     |     (Full frame-by-frame evidence)    |      Print-ready A4 PDF & HTML report
     +---------------------------------------+
```

---

## 3. Top 10 Failure Modes and Safe-Fail Mechanisms

| # | Failure Mode | Root Cause | Module Affected | Safe-Fail Mechanism |
|---|---|---|---|---|
| **1** | **Plate Completely Unreadable** | Extreme motion blur, heavy glare, mud, or extreme distance. | `pipeline.plate_stage` & `pipeline.ocr` | Emits empty OCR character candidates. Fusion sets all positions to `?`. Abstention engine forces `NO_CONFIDENT_MATCH`. No hallucinated guess is ever made. |
| **2** | **Vehicle Detector Misses Vehicle** | Severe occlusion, low night-time illumination, or extreme angle. | `pipeline.detect_track` | Pipeline logs `ZERO_VEHICLES_DETECTED`, marks case status gracefully as failed with reason, skips downstream OCR without throwing an unhandled exception. |
| **3** | **Track ID Fragmentation / Switch** | Vehicle temporarily occluded behind light post or another car. | `pipeline.detect_track` | System clusters spatial trajectories or surfaces multiple candidate tracks in the UI for investigator selection. Each track produces its own isolated evidence trail. |
| **4** | **Partial Plate Occlusion** | Tow hook, dirt smudge, or high-security hologram covering 1-2 digits. | `fusion.fuser` | Per-position voting notes low margin / contradictory reads at occluded slot, safely outputs `?` at that exact index (e.g. `DL01AB?234`) while preserving surrounding characters. |
| **5** | **Misleading Optical Character Confusion** | Ambiguity between `O` and `0`, `B` and `8`, `I` and `1`, `S` and `5`, `Z` and `2`. | `fusion.grammar` & `ranking.plate_matcher` | Positional grammar applies state/district/digit priors. Ranking applies a reduced penalty for confusable pairs instead of an outright rejection. |
| **6** | **Target Vehicle Absent From Registry** | Stolen vehicle, out-of-state unmapped registration, or counterfeit plate. | `ranking.ranker` | If top registry candidate score is below `no_match_threshold` (0.52), system emits `NO_CONFIDENT_MATCH: "No registry record satisfies the observed plate and attribute evidence."` |
| **7** | **Two Identical Plate Candidates in Registry** | Fake/cloned plate in registry or collision. | `ranking.ranker` | Ranks both with identical plate scores, then breaks tie cleanly using extracted body colour and body-type attributes. Explains tie in `why_match` / `why_not`. |
| **8** | **Investigator Supplies Conflicting Hint** | Investigator enters "Red SUV" but video shows "Silver Hatchback". | `pipeline.attributes` | System treats investigator hint as a soft prior only. Reports detected evidence vs hint mismatch explicitly in report without corrupting automated readings. |
| **9** | **Corrupted / Truncated Video Stream** | Broken CCTV codec, incomplete upload, or missing moov atom. | `pipeline.frames` | `cv2.VideoCapture` validation checks total frame count and decodability. If corrupt, returns immediate HTTP 422 with clear diagnostic error message. |
| **10** | **Extreme Night Lighting / Sodium Glare** | Strong monochrome streetlights distorting vehicle paint colour. | `pipeline.attributes` | Colour distribution entropy check detects low chromatic saturation; colour confidence is degraded towards 0. Ranking automatically down-weights colour score. |

---

## 4. 36-Hour Hackathon Task-Split Matrix (4 Teammates)

**Team Roster:**
- **Dev 1 (Lead / ML & CV Pipeline)**: Core video pipeline, YOLO detection, ByteTrack, plate localization, PaddleOCR fine-tuning.
- **Dev 2 (Algorithms / Fusion & Ranking)**: Multi-frame evidential fusion, grammar constraints, registry schema, ranking engine, abstention logic.
- **Dev 3 (Backend & Data Engineering)**: Synthetic registry generation, synthetic plate generator & degradation pipeline, FastAPI backend, SQLite storage.
- **Dev 4 (Frontend & Reporting / Evaluation)**: Investigator dashboard (Streamlit / React), HTML/PDF forensic report generator, evaluation suite, demo clips.

### Master Schedule (Hour 0 to Hour 36)

```
[Hours 00 - 04]: Architecture Freeze, Repository Skeleton & Synthetic Data Generation
[Hours 04 - 12]: Core Model Training & Baseline Video Pipeline
[Hours 12 - 20]: Original Fusion Algorithm & Registry Matching Engine
[Hours 20 - 28]: Backend API, Interactive UI & Forensic Report Generation
[Hours 28 - 34]: End-to-End Evaluation, Ablation Studies & Tuning
[Hours 34 - 36]: Demo Rehearsal, Pitch Deck & Code Freeze
```

### Detailed Breakdown

| Part # | Module / Task | Owner | Hours | Dependencies | Deliverable |
|---|---|---|---|---|---|
| **Part 1** | Architecture, Schemas & Master Config | Dev 1 & Dev 2 | 0 - 2 | None | `docs/architecture.md`, `pipeline/schemas.py`, `configs/config.yaml` |
| **Part 2** | Synthetic Registry & Ground Truth Format | Dev 3 | 2 - 5 | Part 1 | `scripts/make_registry.py`, `data/registry/vehicles.csv`, `tests/test_registry.py` |
| **Part 3** | Synthetic Plate Generator & OCR Pipeline | Dev 1 & Dev 3 | 4 - 10 | Part 2 | `scripts/gen_plates.py`, `pipeline/degrade.py`, `pipeline/ocr.py`, OCR calibration |
| **Part 4** | Plate Detector & Body Type Classifier | Dev 1 | 8 - 14 | Part 1 | `models/plate_yolov8s.pt`, `scripts/prep_bodytype.py`, training scripts |
| **Part 5** | Core Video Pipeline (Detect, Track, Crops) | Dev 1 | 12 - 18 | Part 4 | `pipeline/frames.py`, `pipeline/detect_track.py`, `pipeline/plate_stage.py`, `pipeline/attributes.py` |
| **Part 6** | Multi-Frame Fusion (Original Algorithm) | Dev 2 | 10 - 18 | Part 1, 3 | `fusion/weights.py`, `fusion/align.py`, `fusion/grammar.py`, `fusion/fuser.py`, unit tests |
| **Part 7** | Registry Matching & Ranking Engine | Dev 2 | 16 - 22 | Part 2, 6 | `ranking/plate_matcher.py`, `ranking/ranker.py`, abstention logic, tuning script |
| **Part 8** | Backend FastAPI Service | Dev 3 | 18 - 24 | Part 5, 7 | `api/app.py`, background task runner, REST endpoints, pytest API tests |
| **Part 9** | Investigator Forensic UI | Dev 4 | 20 - 28 | Part 8 | `ui/app.py` (dark theme dashboard, evidence heatmap, shortlist review) |
| **Part 10** | Forensic Report Generator (HTML & PDF) | Dev 4 | 24 - 30 | Part 7, 8 | `reports/report.py`, Jinja2 template, WeasyPrint/ReportLab PDF export |
| **Part 11** | Evaluation Suite & Ablation Studies | Dev 2 & Dev 4 | 28 - 32 | Part 6, 7, 10 | `eval/run_all.py`, character accuracy vs blur curves, ablation charts, metrics CSV |
| **Part 12** | Bonus Features (Make/Model, Cross-Camera) | Dev 1 & Dev 2 | 30 - 33 | Part 5, 8 | Re-ID appearance embedding, cross-camera temporal matching |
| **Part 13** | Pitch Deck, Demo Scripts & Rehearsal | All 4 Devs | 33 - 36 | All Parts | 5-minute demo video script, 6-slide presentation deck, contingency cache |
