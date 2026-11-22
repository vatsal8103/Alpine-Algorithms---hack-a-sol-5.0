# TRACE-X: Forensic Investigator Interface Specification

**Target**: Dark Forensic Dashboard Style  
**Framework Choice**: Streamlit (with Custom CSS / Forensic Styling)  
**Rationale**: Eliminates cross-language serialization friction in a 36-hour hackathon; enables direct real-time interaction with the Python CV pipeline, ByteTrack trajectories, and multi-frame fusion engine; avoids Node.js build overhead while delivering projector-ready dark theme aesthetics.

---

## 1. Design System & Visual Palette

### Color Tokens
- **Background Deep Navy**: `#0b1120` (slate-950)
- **Card / Surface Charcoal**: `#1e293b` (slate-800)
- **Card Border**: `#334155` (slate-700)
- **Primary Accent / Brand**: `#3b82f6` (blue-500)
- **Confirmed Evidence (Match)**: `#10b981` (emerald-500)
- **Uncertain / Wildcard (`?`)**: `#f59e0b` (amber-500)
- **Abstention / Contradiction (No Match)**: `#ef4444` (red-500)
- **Text Primary**: `#f8fafc` (slate-50)
- **Text Muted / Subtitle**: `#94a3b8` (slate-400)

### Typography & Hierarchy
- **Header**: Inter / system sans-serif, 24px bold
- **Plate Tiles**: Monospace (Courier New / Roboto Mono), 32px bold, letter-spacing +2px
- **Score Breakdown**: 12px tabular numbers with CSS progress bar fills
- **Contrast Compliance**: Minimum 4.5:1 WCAG AA contrast ratio across all evidence tiles

---

## 2. Component Layout & Functional Views

```
+---------------------------------------------------------------------------------------+
|  [TRACE-X LOGO]  Forensic Vehicle Shortlisting & Evidence Audit Dashboard            |
+---------------------------------------------------------------------------------------+
| 1. EVIDENCE INTAKE                                                                    |
|    [ Upload Video File (.mp4) ]   [ Demo Presets: Clip A (SUV) | Clip B | No-Match ]  |
|    Optional Hints: Colour [ Dropdown ] | Body Type [ Dropdown ] | Partial Plate [     ]|
|    [ > EXECUTE FORENSIC ANALYSIS BUTTON ]                                             |
+---------------------------------------------------------------------------------------+
| 2. LIVE EVIDENCE PIPELINE PROGRESS                                                    |
|    [Frames: 11] -> [Track: #1] -> [Plates: 11] -> [Fusion: OK] -> [Ranked 500 Reg]   |
+---------------------------------------------------------------------------------------+
| 3. CONSENSUS EVIDENCE RECOVERY                                                        |
|    Recovered Plate: [ J ] [ H ] [ 0 ] [ 5 ] [ A ] [ B ] [ ? ] [ 7 ] [ 9 ] [ 9 ]       |
|    Status: Valid Indian Syntax | Frames Used: 11 | Mean Confidence: 89.2%             |
+---------------------------------------------------------------------------------------+
| 4. INVESTIGATION OUTCOME BANNER                                                       |
|    [ GREEN: CANDIDATE SHORTLIST FOUND ]  OR  [ RED: NO CONFIDENT MATCH ABSTENTION ]   |
|    Reason: "Highest-ranked candidate based on the evidence (Score 0.91 >= 0.52)"      |
+---------------------------------------------------------------------------------------+
| 5. TOP-5 CANDIDATE SHORTLIST                                                          |
|    Rank #1: JH05AB6799 (Mahindra Scorpio, White SUV) - 91.2% Score                    |
|             Plate [===== 94%]  Colour [==== 90%]  Body [==== 88%]                     |
|             Why: Verified match at 9/10 positions; colour matches registered          |
|    Rank #2: JH05AB8799 (Confusable B/8 negative) - 78.4% Score                        |
|             Why Not: Position 6 mismatch 'B' vs '8'                                   |
+---------------------------------------------------------------------------------------+
| 6. EVIDENCE VIEWER & AUDIT TRAIL                                                      |
|    Thumbnails: [Frame 0 Crop] [Frame 2 Crop] [Frame 4 Crop] [Frame 6 Crop]            |
|    Per-Character Voting Heatmap & Frame Vote Weights                                  |
+---------------------------------------------------------------------------------------+
| 7. EXPORT FORENSIC REPORT                                                             |
|    [ Download Official Forensic Report (HTML / PDF) ]                                 |
+---------------------------------------------------------------------------------------+
```

---

## 3. Strict Copywriting Rules
1. **Never use words**: *"Confirmed vehicle"*, *"Identified perpetrator"*, or *"Exact car"*.
2. **Always state**: *"Highest-ranked candidate based on the evidence."*
3. **Mandatory Disclaimer**: Displayed prominently at footer:
   > *"Decision-support only. Results represent candidate likelihood based on available video signals. Never use as sole proof for enforcement action."*
