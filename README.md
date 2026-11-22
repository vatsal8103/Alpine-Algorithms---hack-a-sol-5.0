# TRACE-X: Forensic Vehicle Identification and Traceability from CCTV

**Hackasol 5.0 — Track 2: AIML (Problem Statement AIML-01)**  
*Student Team: Alpine Algorithms*

---

## 📌 Executive Summary

TRACE-X is an explainable forensic decision-support platform designed to identify and rank candidate vehicles from degraded, low-resolution, and blurry CCTV video clips. In hit-and-run, theft, or traffic violation scenarios, license plates are rarely legible in a single snapshot. 

TRACE-X solves this through **multi-frame evidential consensus**: extracting probabilistic character predictions across multiple video frames, aligning them with weighted Needleman-Wunsch string alignment, applying Indian registration grammar constraints, and fusing them into a consensus plate. Any unreadable character is honestly preserved as `?` and **never invented**. The candidate is then scored against a sample vehicle registry using an original multi-criteria ranking algorithm (partial-plate match, body colour similarity, and body-type compatibility). When evidence is insufficient or contradictory, TRACE-X safely abstains with `NO_CONFIDENT_MATCH`.

> **Guiding Principle**: *"Highest-ranked candidate based on the evidence."* TRACE-X is an investigator decision-support tool, not definitive judicial proof.

---

## 🏗️ Repository Architecture

```text
Alpine-Algorithms---hack-a-sol-5.0/
├── configs/                  # Master configuration and similarity matrices
│   ├── config.yaml           # All tunable thresholds, weights, and model configs
│   ├── colour_similarity.json # Perceptual colour similarity partial-credit matrix
│   ├── body_type_similarity.json # Body type partial-credit matrix
│   └── schemas/              # Auto-generated JSON Schemas for all DTOs
├── data/                     # Data stores (never commit raw sensitive footage)
│   ├── registry/             # Sample vehicle registry (CSV + SQLite)
│   ├── clips/                # CCTV video clips (.mp4)
│   ├── plates_synth/         # Generated synthetic plates with graded degradation
│   └── labels/               # Ground-truth annotations split by clip
├── models/                   # Model weights and checkpoints
│   └── README.md             # Weights provenance and licensing details
├── pipeline/                 # Core video processing pipeline
│   ├── schemas.py            # Pydantic models (FrameOCRResult, FusedPlate, etc.)
│   ├── frames.py             # Frame extraction and Laplacian sharpness scoring
│   ├── detect_track.py       # YOLOv8s vehicle detection + ByteTrack tracking
│   ├── plate_stage.py        # Plate localization and per-frame CTC OCR
│   ├── attributes.py         # Colour segmentation and body-type classification
│   ├── select_frames.py      # Frame quality selection and duplicate suppression
│   └── run_pipeline.py       # End-to-end pipeline CLI
├── fusion/                   # Multi-frame evidential fusion (Original Algorithm)
│   ├── weights.py            # Sharpness, area, and confidence frame weighting
│   ├── align.py              # Weighted Needleman-Wunsch edit-distance alignment
│   ├── grammar.py            # Indian license plate grammar constraints
│   └── fuser.py              # Per-position evidential consensus fuser
├── ranking/                  # Registry matching & ranking (Original Algorithm)
│   ├── plate_matcher.py      # Positional plate scoring & confusable pair penalties
│   ├── attribute_matcher.py  # Matrix-based colour and body-type scoring
│   └── ranker.py             # Composite ranking, top-5 shortlist, and abstention
├── api/                      # FastAPI backend service
│   └── app.py                # REST endpoints (/cases, /status, /result, /report)
├── ui/                       # Investigator frontend dashboard (Streamlit / React)
│   └── app.py                # Forensic dark-theme dashboard
├── reports/                  # Forensic reporting engine
│   └── report.py             # Standalone print-ready HTML & PDF report generator
├── eval/                     # Evaluation benchmark suite
│   ├── eval_ocr.py           # Character accuracy vs blur degradation levels 0-5
│   ├── eval_fusion.py        # Single-frame vs majority-vote vs TRACE-X fusion
│   ├── eval_retrieval.py     # Top-1/Top-5 retrieval and false-accept rate
│   └── run_all.py            # Master evaluation CLI
├── scripts/                  # Utility and data generation scripts
│   ├── generate_registry.py  # Deterministic 500-vehicle Indian registry generator
│   └── make_demo_data.py     # Synthetic CCTV clip generator
├── tests/                    # Pytest test suite
│   ├── test_schemas.py       # Schema validation and JSON Schema export tests
│   └── test_config.py        # Master config and matrix validation tests
└── docs/                     # Architectural documentation
    ├── architecture.md       # Complete system architecture and data-flow diagram
    ├── failure_modes.md      # Top 10 failure modes and safe-failing mechanisms
    └── schedule.md           # 36-hour hackathon task-split matrix
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- PyTorch & Torchvision
- OpenCV (`opencv-python-headless`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-team/Alpine-Algorithms---hack-a-sol-5.0.git
cd Alpine-Algorithms---hack-a-sol-5.0

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Run via Unified `main.py`
TRACE-X features a unified entrypoint for all operations:

```bash
# 1. Launch Forensic Web Dashboard (Streamlit)
python main.py ui

# 2. Launch FastAPI Backend Server
python main.py api

# 3. Analyze a CCTV video clip
python main.py analyze data/clips/jh05ab6799_camA.mp4 --colour white --body suv

# 4. Run real-time live video processing (Webcam or RTSP)
python main.py live --source 0
# Or RTSP stream: python main.py live --source "rtsp://camera_ip:554/live"

# 5. Check GPU acceleration & PyTorch CUDA status
python main.py gpu

# 6. Generate forensic HTML/PDF reports for all test cases
python main.py reports

# 7. Run evaluation benchmarks (OCR, Fusion, Retrieval accuracy curves)
python main.py eval

# 8. Run unit test suite (43 pytest tests)
python main.py test
```

---

## 🚀 GPU Acceleration on NVIDIA RTX 3050

TRACE-X supports automatic hardware acceleration with PyTorch CUDA (FP16 half-precision and Tensor Cores).

To enable full GPU acceleration on your NVIDIA GeForce RTX 3050 Laptop GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU acceleration with:
```bash
python main.py gpu
```
*If PyTorch is installed without CUDA, TRACE-X automatically and safely falls back to CPU execution without crashing.*

---

## 🛡️ Core Guarantees & Safe Failure Modes

1. **Never Invent Characters**: If character posterior is below threshold (0.50) or margin to runner-up is below threshold (0.12), position is marked `?`.
2. **Honest Abstention**: When no registry candidate scores above `no_match_threshold` (0.52), or when high-confidence plate characters contradict candidate #1, system outputs `NO_CONFIDENT_MATCH`.
3. **Audit Trail**: Every position in `FusedPlate` maintains the complete list of frame votes, individual frame confidences, and weights.
4. **Data Isolation**: Ground-truth datasets are split strictly by video clip / vehicle identity, never by frame.
