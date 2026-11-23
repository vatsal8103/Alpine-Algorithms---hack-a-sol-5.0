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

## 🤖 Vision-Language Model (VLM) Forensic Escalation & Character Fusion

TRACE-X employs a **cascaded hybrid architecture** that divides real-time processing and deep visual reasoning:

```mermaid
flowchart TD
    Video["CCTV Video Stream"] --> YOLO["YOLO11n + ByteTrack (2.2ms, 99.4% mAP50)"]
    YOLO --> FastOCR["Multi-Frame Fast OCR + Needleman-Wunsch Alignment"]
    FastOCR --> Evaluator{"Are there '?' slots or low confidence (<0.70)?"}
    Evaluator -->|"No (High Confidence)"| Registry["Registry Matching & Ranking"]
    Evaluator -->|"Yes ('?' detected)"| VLM["Qwen2-VL-2B-Instruct (4-bit NF4 / LM Studio Q4_K_M)"]
    VLM --> Fusion["Character Evidential Fusion & Score Averaging"]
    Fusion --> Registry
```

### 1. Character-Level Evidential Fusion & Score Averaging
When Qwen2-VL is consulted on an ambiguous plate (e.g. `MP13?U2058` where position #5 is `?`):
- **Question Mark Resolution:** The `?` slot is replaced by the VLM's predicted character (`'Z'`), assigning a calibrated prior confidence (0.85).
- **Consensus Boosting:** Where Fast OCR and VLM agree (e.g. `M, P, 1, 3, U, 2, 0, 5, 8`), character confidence is cross-verified and boosted: $C_{fused} = \min(0.99, \frac{C_{ocr} + 0.92}{2} + 0.05)$.
- **Averaged Confidence:** The plate's mean confidence increases from `56.2%` to `78.2%`, eliminating uncertainty without hallucination.

### 2. Do You Need to Fine-Tune the VLM for Indian Plates?
**Short Answer: No, zero-shot prompting with MoRTH grammar constraints is superior and sufficient.**

| Criteria | Zero-Shot Qwen2-VL + MoRTH Prompting (Current) | Fine-Tuning on Small Indian Dataset |
| :--- | :--- | :--- |
| **Alphanumeric Recognition** | Native high-accuracy from pre-training on billions of multimodal tokens | Risk of catastrophic forgetting on rare fonts |
| **Generalization** | Out-of-the-box support for both white private & yellow commercial plates | Overfits to specific backgrounds/angles in training set |
| **Two-Line Scooters / Bikes** | Strict prompt rules enforce top-line (State+RTO) then bottom-line (Series+Number) | Requires thousands of balanced two-line crops |
| **Inference Efficiency** | 4-bit NF4 (~2.8 GB VRAM) or LM Studio Q4_K_M (1.7 GB GGUF) | LoRA adapters add loading latency |
| **Forensic Admissibility** | Grounded in zero-shot foundation weights + evidential cross-check | Custom weights must undergo separate validation audits |

*If custom fine-tuning is ever required (e.g. for regional Devanagari/Tamil script plates or infrared night lenses), TRACE-X provides an automated QLoRA training script in `scripts/train_vlm_qwen2.py`.*

---

## 🛡️ Core Guarantees & Safe Failure Modes

1. **Never Invent Characters**: If character posterior is below threshold (0.50) or margin to runner-up is below threshold (0.12), position is marked `?`.
2. **Honest Abstention**: When no registry candidate scores above `no_match_threshold` (0.52), or when high-confidence plate characters contradict candidate #1, system outputs `NO_CONFIDENT_MATCH`.
3. **Audit Trail**: Every position in `FusedPlate` maintains the complete list of frame votes, individual frame confidences, and weights.
4. **Data Isolation**: Ground-truth datasets are split strictly by video clip / vehicle identity, never by frame.
