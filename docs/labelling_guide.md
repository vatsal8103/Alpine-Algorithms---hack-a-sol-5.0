# TRACE-X: Video Ground-Truth Annotation & Clip Selection Guide

This guide describes how to collect, select, label, and split test video clips for TRACE-X evaluation under strict forensic protocol.

---

## 1. Clip Selection Guidelines (5–10 Benchmark Clips)

To validate the robustness of TRACE-X across realistic CCTV surveillance conditions, the benchmark test suite must contain diverse operational scenarios:

| Clip Category | Target Attributes | Key Challenge | Example Clip |
|---|---|---|---|
| **1. Standard Daytime** | SUV / Car, White / Silver, 10–25m range | Baseline tracking & multi-frame fusion | `jh05ab6799_camA.mp4` |
| **2. Cross-Camera** | Same vehicle passing 2 distinct camera poles | Time-gap appearance & plate compatibility | `jh05ab6799_camB.mp4` |
| **3. High Motion Blur** | Hatchback passing at 50+ km/h | Frame alignment & Laplacian sharpness filtering | `mh12de1433_camA.mp4` |
| **4. Two-Wheeler** | Motorcycle / Scooter, vertical or two-line plate | Smaller plate area (px), high vibration | `ka03mn4821_camA.mp4` |
| **5. Commercial / Yellow**| Auto-rickshaw / Taxi, yellow plate background | Plate style detection & non-standard font | `cg04tr7788_camA.mp4` |
| **6. Heavy Vehicle** | Multi-axle truck / Bus, high bumper position | Bounding box aspect ratio & dirty plate | `up32kl9021_camA.mp4` |
| **7. Night / Glare** | Headlight flare & sodium-vapor illumination | Low color saturation & extreme contrast | `dl8caf1234_camA.mp4` |
| **8. Hard Negative (1-char)**| Identical model, plate differs by 1 digit | Must not falsely match anchor vehicle | `jh05ab6798_nomatch.mp4` |
| **9. Out-of-Registry** | Legitimate readable plate NOT in database | Must return `NO_CONFIDENT_MATCH` | `tn09zz4321_nomatch.mp4` |
| **10. Severely Degraded** | Completely unreadable plate (heavy defocus/rain) | Must mark `?` and cleanly abstain | `wb02xx0000_unreadable.mp4` |

---

## 2. Strict Clip-Level Data Splitting Rule

> **CRITICAL RULE**: Splitting data by frames is **STRICTLY FORBIDDEN**.
> Adjacent frames in a CCTV clip have >95% mutual correlation. Splitting by frame causes massive data leakage and misleading 99% accuracy figures.

- **Split Unit**: Always split by **Video Clip / Unique Vehicle Passage**.
- **Training Set**: Synthetic generated plate crops and disjoint clips.
- **Validation Set**: Independent synthetic degradation sweeps.
- **Evaluation / Test Set**: Held-out physical or synthetic CCTV clips (`data/clips/`) whose exact video sequences were **never** used during detector, classifier, or OCR training.

---

## 3. Ground Truth CSV Schema

The annotation table is stored at `data/labels/ground_truth.csv`:

```csv
clip_name,true_plate,colour,body_type,in_registry,camera_id,notes
```

- **`clip_name`** (string): Filename located inside `data/clips/`.
- **`true_plate`** (string): Verified ground-truth registration string (e.g. `JH05AB6799`).
- **`colour`** (string): Ground-truth body color (`white`, `silver`, `black`, `red`, `blue`, `yellow`, `green`, `brown`, `grey`).
- **`body_type`** (string): One of `hatchback`, `sedan`, `suv`, `two-wheeler`, `auto-rickshaw`, `truck`, `bus`.
- **`in_registry`** (bool): `True` if plate exists in `data/registry/vehicles.csv`, `False` if intentionally omitted for no-match testing.
- **`camera_id`** (string): Camera station identifier (e.g. `cam-A`, `cam-B`).
- **`notes`** (string): Forensic conditions (e.g., "high vibration", "headlight glare").
