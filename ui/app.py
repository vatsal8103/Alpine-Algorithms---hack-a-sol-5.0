"""TRACE-X Forensic Investigator Dashboard.

A dark forensic web interface for video evidence analysis, multi-frame plate recovery,
and explainable candidate vehicle shortlisting.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.run_pipeline import run_pipeline
from pipeline.schemas import CaseResult
from reports.report import generate_html_report

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Forensic Clean Light Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TRACE-X | Forensic Vehicle Shortlisting",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Forensic Clean Light Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .plate-tile-known {
        display: inline-block;
        width: 44px;
        height: 56px;
        line-height: 56px;
        text-align: center;
        background: #ffffff;
        color: #0f172a;
        font-family: 'Courier New', Courier, monospace;
        font-size: 28px;
        font-weight: 800;
        border: 2px solid #0f172a;
        border-radius: 6px;
        margin-right: 6px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .plate-tile-unknown {
        display: inline-block;
        width: 44px;
        height: 56px;
        line-height: 56px;
        text-align: center;
        background: #fef3c7;
        color: #b45309;
        border: 2px dashed #f59e0b;
        font-family: 'Courier New', Courier, monospace;
        font-size: 28px;
        font-weight: 800;
        border-radius: 6px;
        margin-right: 6px;
    }
    .banner-match {
        background: #ecfdf5;
        border: 1px solid #10b981;
        border-left: 6px solid #10b981;
        padding: 14px 18px;
        border-radius: 6px;
        color: #065f46;
        font-weight: 600;
        margin: 14px 0;
    }
    .banner-nomatch {
        background: #fef2f2;
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        padding: 14px 18px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: 600;
        margin: 14px 0;
    }
    .disclaimer-box {
        margin-top: 40px;
        padding: 12px;
        border-top: 1px solid #e2e8f0;
        font-size: 11px;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Sidebar: Configuration & Investigation Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("TRACE-X Core")
    st.caption("Forensic Decision-Support System")
    st.divider()

    st.subheader("System Weights & Thresholds")
    st.text("Plate Weight:      0.65")
    st.text("Colour Weight:     0.15")
    st.text("Body Weight:       0.12")
    st.text("Visual Weight:     0.08")
    st.text("No-Match Thresh:   0.52")
    st.text("Margin Thresh:     0.06")
    st.divider()

    st.subheader("Model Status")
    st.success("Vehicle Detector: YOLOv8s COCO")
    st.success("Tracker: ByteTrack Online")
    st.success("OCR: Multi-Frame Probabilistic Fuser")
    st.success("Registry: 500 Indexed Vehicles")


# -----------------------------------------------------------------------------
# 3. Main Header & Video Intake
# -----------------------------------------------------------------------------
st.title("Vehicle Evidence Analysis & Shortlisting")
st.markdown("Upload CCTV video footage to extract multi-frame plate evidence, infer attributes, and rank database candidates.")

col_upload, col_hints = st.columns([1.5, 1])

demo_clips = {
    "Select a demo clip...": None,
    "Clip 1: Daylight Mahindra Scorpio SUV (JH05AB6799)": "data/clips/jh05ab6799_camA.mp4",
    "Clip 2: Cross-Camera Pass (Camera B)": "data/clips/jh05ab6799_camB.mp4",
    "Clip 3: Silver Maruti Swift Hatchback (MH12DE1433)": "data/clips/mh12de1433_camA.mp4",
    "Clip 4: Out-of-Registry Car (Abstention Test)": "data/clips/tn09zz4321_nomatch.mp4",
    "Clip 5: Severely Degraded Plate (Unreadable)": "data/clips/wb02xx0000_unreadable.mp4",
}

with col_upload:
    uploaded_file = st.file_uploader("Upload CCTV Video Clip", type=["mp4", "avi", "mov"])
    chosen_demo = st.selectbox("Or choose a pre-loaded benchmark clip:", list(demo_clips.keys()))

with col_hints:
    st.markdown("**Optional Investigator Hints (Soft Priors)**")
    hint_colour = st.selectbox("Vehicle Colour Hint:", ["", "White", "Silver", "Black", "Grey", "Red", "Blue", "Yellow"])
    hint_body = st.selectbox("Body Type Hint:", ["", "SUV", "Hatchback", "Sedan", "Two-Wheeler", "Auto-Rickshaw", "Truck", "Bus"])

# Resolve video path
active_video_path = None
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    active_video_path = tfile.name
elif chosen_demo and demo_clips[chosen_demo] and Path(demo_clips[chosen_demo]).exists():
    active_video_path = demo_clips[chosen_demo]

if active_video_path:
    st.video(active_video_path)

run_button = st.button("RUN FORENSIC ANALYSIS", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Pipeline Execution & Output Views
# -----------------------------------------------------------------------------
if run_button and active_video_path:
    with st.spinner("Processing video: extracting frames, tracking vehicles, and fusing evidential consensus..."):
        hints = {}
        if hint_colour:
            hints["colour"] = hint_colour.lower()
        if hint_body:
            hints["body_type"] = hint_body.lower()

        case_res: CaseResult = run_pipeline(active_video_path, hints=hints)

    st.divider()

    # Section 1: Recovered Plate Display
    st.subheader("1. Consensus License Plate Recovery")
    plate_html = "<div style='margin: 16px 0;'>"
    for char in case_res.fused_plate.text:
        if char == "?":
            plate_html += f"<span class='plate-tile-unknown'>{char}</span>"
        else:
            plate_html += f"<span class='plate-tile-known'>{char}</span>"
    plate_html += "</div>"
    st.markdown(plate_html, unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Frames Used", case_res.fused_plate.frames_used)
    col_m2.metric("Known Characters", f"{case_res.fused_plate.known_count}/10")
    col_m3.metric("Plate Confidence", f"{case_res.fused_plate.mean_confidence*100:.1f}%")
    col_m4.metric("Indian Syntax Valid", "Yes" if case_res.fused_plate.format_valid else "Partial / Ambiguous")

    # Section 2: Vehicle Attributes
    st.subheader("2. Extracted Vehicle Attributes")
    ca1, ca2, ca3 = st.columns(3)
    ca1.info(f"**Observed Colour**: {case_res.attributes.colour.upper()} (conf: {case_res.attributes.colour_conf:.2f})")
    ca2.info(f"**Body Type**: {case_res.attributes.body_type.upper()} (conf: {case_res.attributes.body_conf:.2f})")
    ca3.info(f"**Estimated Make/Model**: {case_res.attributes.make} {case_res.attributes.model}")

    # Section 3: Decision Outcome Banner
    st.subheader("3. Investigation Decision")
    if case_res.decision == "MATCH_FOUND":
        st.markdown(f"""
        <div class="banner-match">
            DECISION: CANDIDATE SHORTLIST IDENTIFIED<br/>
            <span style="font-size: 13px; font-weight: normal;">{case_res.decision_reason}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="banner-nomatch">
            DECISION: NO CONFIDENT MATCH<br/>
            <span style="font-size: 13px; font-weight: normal;">{case_res.decision_reason}</span>
        </div>
        """, unsafe_allow_html=True)

    # Section 4: Ranked Candidate Shortlist
    st.subheader("4. Ranked Candidate Shortlist (Top 5)")
    if case_res.candidates:
        for idx, cand in enumerate(case_res.candidates):
            with st.expander(f"Rank #{idx+1}: {cand.registration_number} (ID: {cand.vehicle_id}) — Score: {cand.final_score:.3f}", expanded=(idx == 0)):
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    st.markdown("**Evidence Breakdown:**")
                    st.progress(cand.breakdown.plate, text=f"Partial-Plate Match: {cand.breakdown.plate*100:.1f}%")
                    st.progress(cand.breakdown.colour, text=f"Colour Compatibility: {cand.breakdown.colour*100:.1f}%")
                    st.progress(cand.breakdown.body, text=f"Body Type Compatibility: {cand.breakdown.body*100:.1f}%")
                with col_c2:
                    st.markdown("**Why it matches:**")
                    for m in cand.why_match:
                        st.markdown(f"- ✅ {m}")
                    if cand.why_not:
                        st.markdown("**Why ranked lower / conflicts:**")
                        for n in cand.why_not:
                            st.markdown(f"- ⚠️ {n}")
    else:
        st.warning("No candidates evaluated or registry empty.")

    # Section 5: Generate Report Download
    st.divider()
    report_path = f"reports/output/{case_res.case_id}_report.html"
    generate_html_report(case_res, report_path)
    if Path(report_path).exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html_bytes = f.read().encode("utf-8")
        st.download_button(
            label="DOWNLOAD OFFICIAL FORENSIC REPORT (HTML)",
            data=html_bytes,
            file_name=f"{case_res.case_id}_forensic_report.html",
            mime="text/html"
        )

# -----------------------------------------------------------------------------
# 5. Mandatory Disclaimer Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <strong>DISCLAIMER</strong>: Highest-ranked candidate based on the evidence. Decision-support only.
    Results are never a definitive identification. Never use this output as sole proof for enforcement action.
</div>
""", unsafe_allow_html=True)
