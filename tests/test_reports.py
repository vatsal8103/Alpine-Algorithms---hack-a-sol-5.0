"""Tests for forensic report generation."""

from __future__ import annotations

from pathlib import Path
import pytest
from pipeline.schemas import Candidate, CaseResult, FusedPlate, ScoreBreakdown, VehicleAttributes
from reports.report import generate_html_report


@pytest.fixture
def dummy_case():
    fused = FusedPlate(
        text="JH05AB?799",
        format_valid=True,
        frames_used=12,
        mean_confidence=0.89,
        known_count=9,
        unknown_count=1
    )
    attr = VehicleAttributes(
        colour="white",
        colour_conf=0.92,
        body_type="suv",
        body_conf=0.88,
        make="Mahindra",
        model="Scorpio"
    )
    cand = Candidate(
        vehicle_id="v-101",
        registration_number="JH05AB6799",
        final_score=0.915,
        breakdown=ScoreBreakdown(plate=0.92, colour=0.90, body=0.88),
        why_match=["Plate match 9/10", "Colour match: white"],
        why_not=[]
    )
    return CaseResult(
        case_id="case-test-report-01",
        clip_name="test_cctv_clip.mp4",
        fused_plate=fused,
        attributes=attr,
        candidates=[cand],
        decision="MATCH_FOUND",
        decision_reason="Highest-ranked candidate based on the evidence.",
        thresholds_used={"no_match_threshold": 0.52, "margin_threshold": 0.06},
        weights_used={"plate": 0.65, "colour": 0.15, "body": 0.12, "visual": 0.08}
    )


def test_report_generation_and_required_sections(tmp_path, dummy_case):
    report_file = tmp_path / "test_report.html"
    generate_html_report(dummy_case, str(report_file))

    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")

    # Check all mandatory sections
    assert "TRACE-X Forensic Vehicle Investigation Report" in content
    assert dummy_case.case_id in content
    assert dummy_case.clip_name in content
    assert "Consensus Plate Recovery Evidence" in content
    assert "Extracted Vehicle Visual Attributes" in content
    assert "Forensic Investigation Decision" in content
    assert "Candidate Shortlist (Top 5 Ranked)" in content
    assert "Audit Configuration & Thresholds Applied" in content
    assert "LEGAL & FORENSIC DISCLAIMER" in content
    assert "Decision-support only" in content
    assert "JH05AB6799" in content
