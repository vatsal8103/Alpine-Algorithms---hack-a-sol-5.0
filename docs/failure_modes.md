# TRACE-X: Top 10 Failure Modes & Safe-Fail Strategy Matrix

This document details how TRACE-X avoids dangerous hallucinations, incorrect convictions, and unhandled crashes.

---

## 1. Failure Modes and Recovery Behaviors

### Failure Mode 1: Plate Completely Unreadable
- **Condition**: Extreme motion blur, heavy headlight glare, mud-covered plate, or vehicle too far from camera.
- **Affected Stages**: `pipeline.plate_stage`, `pipeline.ocr`
- **Impact**: No reliable character features can be extracted from any frame.
- **Safe-Fail Mechanism**: The OCR module outputs empty or below-threshold character candidates. The multi-frame fusion engine designates all positions as unknown (`??????????`). The ranking engine computes scores which fall well below `no_match_threshold` (0.52). The system terminates with:
  `decision: "NO_CONFIDENT_MATCH"`  
  `decision_reason: "Plate is unreadable across all sampled frames; cannot verify candidate match."`
- **Result**: No vehicle is shortlisted or falsely accused.

---

### Failure Mode 2: Vehicle Detector Misses Vehicle
- **Condition**: Severe occlusion, night footage with no ambient lighting, or non-standard vehicle orientation.
- **Affected Stages**: `pipeline.detect_track`
- **Impact**: Zero vehicle bounding boxes detected.
- **Safe-Fail Mechanism**: The detection stage checks if `len(tracks) == 0`. It returns a structured status code `NO_VEHICLES_DETECTED` with diagnostic frame metadata rather than throwing an unhandled `IndexError` or exception.
- **Result**: Graceful exit with an informative log message in the investigator UI.

---

### Failure Mode 3: Track ID Fragmentation / ID Switching
- **Condition**: Vehicle passes behind a tree or lamp post, causing ByteTrack to spawn a new track ID.
- **Affected Stages**: `pipeline.detect_track`
- **Impact**: The same physical vehicle is split into two disjoint track intervals.
- **Safe-Fail Mechanism**: The pipeline provides an investigator track selector endpoint and UI grid displaying thumbnail crops of all tracked trajectories. The investigator can independently select each track fragment for audit.
- **Result**: Each track retains an isolated, tamper-evident evidence trail without cross-track pollution.

---

### Failure Mode 4: Partial Plate Occlusion (Tow Bar / Mud Spot)
- **Condition**: A trailer hitch, tow ball, or dirt patch obscures 1 or 2 specific characters (e.g. `DL01A?3456`).
- **Affected Stages**: `fusion.fuser`, `ranking.plate_matcher`
- **Impact**: Misleading character readings at the occluded position.
- **Safe-Fail Mechanism**: Evidential fusion analyzes the posterior distribution and margin across frames. Because the occluded character yields low confidence and inconsistent votes, the engine marks that exact slot as `?`. The ranking engine treats `?` as a wildcard, allowing valid registry candidates to match without penalty while preventing an invented character from narrowing the search erroneously.
- **Result**: Honest uncertainty representation.

---

### Failure Mode 5: Misleading Optical Character Confusion (e.g., O vs 0, B vs 8)
- **Condition**: Font wear or resolution limit makes `0` look like `O`, `8` like `B`, or `5` like `S`.
- **Affected Stages**: `fusion.grammar`, `ranking.plate_matcher`
- **Impact**: Standard edit-distance string matching would apply an excessive mismatch penalty.
- **Safe-Fail Mechanism**: 
  1. Positional grammar priors enforce alphabetical characters for state/series codes and digits for district/number blocks.
  2. The ranking engine contains an explicit `CONFUSION_PAIRS` penalty discount (0.35 penalty instead of 1.00 mismatch penalty), noting the ambiguity in `why_not` / `why_match`.
- **Result**: Legitimate matches are not discarded due to single-pixel optical ambiguities.

---

### Failure Mode 6: Target Vehicle Absent From Sample Registry
- **Condition**: The vehicle is unregistered, from another state/country, or operating on a forged/cloned plate.
- **Affected Stages**: `ranking.ranker`
- **Impact**: Standard nearest-neighbor matching would force the "closest" car in the database as a match.
- **Safe-Fail Mechanism**: The ranking engine enforces a strict `no_match_threshold` (0.52). If the highest-scoring candidate in the database does not meet this threshold, the system returns `NO_CONFIDENT_MATCH`.
- **Result**: Protects innocent vehicle owners from false identification.

---

### Failure Mode 7: Multiple Identical Candidates in Registry (Plate Collisions)
- **Condition**: Cloned plates or duplicate database records.
- **Affected Stages**: `ranking.ranker`
- **Impact**: Ambiguity in ranking top candidate.
- **Safe-Fail Mechanism**: The plate scores will be tied, but colour similarity and body-type compatibility break the tie. If all attributes remain identical, the system reports both candidates with identical scores and flags a `POTENTIAL_CLONED_PLATE` warning in the rationale.
- **Result**: Complete transparency for forensic examiners.

---

### Failure Mode 8: Investigator Supplies Conflicting Prior Hint
- **Condition**: Human error or eyewitness misremembering (e.g. eyewitness says "Red Sedan", but CCTV shows "Silver SUV").
- **Affected Stages**: `pipeline.attributes`, `ranking.ranker`
- **Impact**: Hard-filtering by user hint would completely eliminate the true culprit vehicle.
- **Safe-Fail Mechanism**: Investigator hints are treated as soft Bayesian priors, never hard SQL filters. The pipeline records both the detected attribute and the investigator hint, displaying the discrepancy explicitly in the report:
  *"Warning: User hint 'Red Sedan' conflicts with detected attribute 'Silver SUV' (conf: 0.88). Shortlisting based on primary sensor evidence."*
- **Result**: Sensor evidence overrides human cognitive bias.

---

### Failure Mode 9: Corrupted or Truncated Video Stream
- **Condition**: Packet loss during transmission, incomplete file upload, or broken MP4 container index.
- **Affected Stages**: `pipeline.frames`, `api.app`
- **Impact**: OpenCV decoding crash or buffer overflow.
- **Safe-Fail Mechanism**: `pipeline.frames.extract_frames` performs video header validation, checking total frame count, FPS, and initial decodability. If corrupt, it immediately returns an empty list and logs an error, triggering HTTP 422 with diagnostic error details.
- **Result**: No server crashes; immediate user feedback.

---

### Failure Mode 10: Extreme Night-Time Streetlight Glare (Sodium Lighting)
- **Condition**: Monochromatic 589nm sodium vapor illumination turns white, silver, and yellow cars into identical orange/amber hues.
- **Affected Stages**: `pipeline.attributes`, `ranking.ranker`
- **Impact**: Colour extraction outputs high-confidence incorrect colour.
- **Safe-Fail Mechanism**: The colour extraction algorithm computes saturation entropy across the vehicle body ROI. Under monochromatic light, saturation variance collapses. When detected, colour confidence is downgraded to near-zero, and the ranking engine renormalizes component weights to rely solely on partial-plate evidence and body silhouette geometry.
- **Result**: Distorted colour does not pollute vehicle ranking.
