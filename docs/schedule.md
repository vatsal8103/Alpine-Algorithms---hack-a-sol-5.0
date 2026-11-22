# TRACE-X: 36-Hour Hackathon Task Schedule (Hackasol 5.0)

**Team: Alpine Algorithms (4 Teammates)**  
**Target Problem**: AIML-01 (Vehicle Identification and Traceability from CCTV Footage)

---

## 1. Team Allocation Matrix

- **Teammate 1 (Dev 1 - ML & Computer Vision Lead)**: Frame extraction, YOLOv8s vehicle detector, ByteTrack tracking, license plate localization, PaddleOCR CTC fine-tuning.
- **Teammate 2 (Dev 2 - Core Algorithms Lead)**: Multi-frame evidential consensus fusion, Needleman-Wunsch string alignment, Indian plate grammar, ranking engine, safe abstention logic.
- **Teammate 3 (Dev 3 - Backend & Data Engineering)**: Synthetic registry generation, synthetic plate generation with 0-5 degradation, FastAPI asynchronous backend, SQLite store.
- **Teammate 4 (Dev 4 - Frontend & Evaluation / Presentation)**: Investigator dark-theme dashboard (Streamlit / React), HTML/PDF forensic report generator, evaluation suite, demo scripts and pitch deck.

---

## 2. 36-Hour Gantt Timeline

| Hours | Phase | Dev 1 (ML / CV) | Dev 2 (Algorithms) | Dev 3 (Backend / Data) | Dev 4 (Frontend / Eval) |
|---|---|---|---|---|---|
| **00 - 02** | **Part 1** | Architecture & schemas | Architecture & config.yaml | Directory skeleton & stubs | Unit test harnesses |
| **02 - 05** | **Part 2 & 3A** | Collect plate fonts & layouts | Registry hard negatives design | `scripts/make_registry.py` (500 recs) | Ground truth CSV schema & labeling guide |
| **05 - 08** | **Part 3B & 3C** | `pipeline/degrade.py` (levels 0-5) | Synthetic plate generation | `scripts/train_ocr.py` setup | Kaggle / Colab training harness |
| **08 - 12** | **Part 3D & 4** | Train Indian plate YOLO & PaddleOCR | Temperature calibration script | PaddleOCR inference wrapper | Evaluation scripts (`eval_ocr.py`) |
| **12 - 16** | **Part 5** | Core video pipeline (`detect_track.py`) | Align algorithm (`align.py`) | Video crop export & caching | Baseline UI layout & theme spec |
| **16 - 20** | **Part 6** | Colour & body-type extraction | Multi-frame fusion engine (`fuser.py`)| Fusion unit tests & grammar rules | Fusion visualizer & evidence viewer |
| **20 - 24** | **Part 7 & 8** | Attribute aggregation & hints | Ranking engine (`ranker.py`) | FastAPI endpoints (`api/app.py`) | REST client integration & case polling |
| **24 - 28** | **Part 9 & 10** | End-to-end pipeline CLI integration | Abstention tuning (`tune_ranking.py`)| SQLite persistence layer | Forensic report generator (`report.py`) |
| **28 - 32** | **Part 11** | Model ablation & latency profiling | Retrieval & honesty evaluation | Batch test clip runner | Automated benchmark charts & plots |
| **32 - 34** | **Part 12** | Bonus: Make/model classifier | Bonus: Cross-camera tracking | Bonus API endpoints | Cross-camera UI timeline |
| **34 - 36** | **Part 13** | Smoke test & contingency cache | Documentation review | System freeze & Docker verify | Demo rehearsal & 6-slide deck |

---

## 3. Milestones & Gates

1. **Gate 1 (Hour 04)**: Master schema freeze, `configs/config.yaml` locked, sample registry generated with hard negatives.
2. **Gate 2 (Hour 12)**: Detection and OCR models running inference on test crops.
3. **Gate 3 (Hour 20)**: Original fusion algorithm passing all unit tests (never inventing characters).
4. **Gate 4 (Hour 28)**: Complete end-to-end integration: Video Upload -> Pipeline -> Fusion -> Ranking -> Report.
5. **Gate 5 (Hour 34)**: Evaluation metrics finalized, ablation curves generated, report generator validated.
6. **Gate 6 (Hour 36)**: Live demo rehearsed with cached fallback artifacts.
