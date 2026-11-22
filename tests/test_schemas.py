"""Tests for TRACE-X Pydantic Schemas and JSON Schema generation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    Candidate,
    CaseResult,
    CharAlternative,
    CharCandidate,
    FrameOCRResult,
    FrameVote,
    FusedPlate,
    PositionVote,
    RegistryRecord,
    ScoreBreakdown,
    VehicleAttributes,
    export_json_schemas,
)


def test_frame_ocr_result_valid():
    chars = [
        CharCandidate(
            char="J",
            prob=0.92,
            alternatives=[CharAlternative(char="U", prob=0.05)]
        ),
        CharCandidate(
            char="H",
            prob=0.88,
            alternatives=[]
        )
    ]
    res = FrameOCRResult(
        track_id=1,
        frame_no=0,
        timestamp=0.0,
        bbox=[100.0, 150.0, 300.0, 200.0],
        plate_crop_path="runs/crops/f0_p0.png",
        sharpness=84.2,
        plate_area_px=10000.0,
        chars=chars
    )
    assert res.track_id == 1
    assert res.frame_no == 0
    assert len(res.chars) == 2
    assert res.chars[0].char == "J"
    assert res.chars[0].prob == 0.92


def test_frame_ocr_result_invalid_prob():
    with pytest.raises(ValidationError):
        CharCandidate(char="A", prob=1.5)  # prob must be <= 1.0


def test_frame_ocr_result_invalid_bbox():
    with pytest.raises(ValidationError):
        FrameOCRResult(
            frame_no=0,
            timestamp=0.0,
            bbox=[100.0, 200.0],  # bbox must have 4 items
            plate_crop_path="crop.png",
            sharpness=50.0,
            plate_area_px=100.0
        )


def test_fused_plate_valid():
    pos_votes = [
        PositionVote(
            index=0,
            char_or_unknown="J",
            confidence=0.95,
            votes=[FrameVote(frame_no=1, char="J", prob=0.95, weight=1.0)]
        ),
        PositionVote(
            index=1,
            char_or_unknown="?",
            confidence=0.42,
            votes=[FrameVote(frame_no=1, char="H", prob=0.42, weight=1.0)],
            explanation="Insufficient confidence"
        )
    ]
    fused = FusedPlate(
        text="J?",
        per_position=pos_votes,
        format_valid=False,
        frames_used=1,
        mean_confidence=0.95,
        known_count=1,
        unknown_count=1
    )
    assert fused.text == "J?"
    assert fused.known_count == 1
    assert fused.unknown_count == 1
    assert fused.per_position[1].char_or_unknown == "?"


def test_vehicle_attributes():
    attr = VehicleAttributes(
        colour="white",
        colour_conf=0.91,
        body_type="suv",
        body_conf=0.87,
        make="Mahindra",
        model="Scorpio",
        colour_distribution={"white": 0.91, "silver": 0.09},
        body_distribution={"suv": 0.87, "hatchback": 0.13}
    )
    assert attr.colour == "white"
    assert attr.body_type == "suv"
    assert attr.make == "Mahindra"


def test_registry_record():
    record = RegistryRecord(
        vehicle_id="v-101",
        registration_number="JH05AB6799",
        colour="white",
        body_type="suv",
        make="Mahindra",
        model="Scorpio"
    )
    assert record.registration_number == "JH05AB6799"
    assert record.colour == "white"


def test_candidate_and_case_result():
    breakdown = ScoreBreakdown(plate=0.92, colour=0.88, body=0.85, visual=None)
    candidate = Candidate(
        vehicle_id="v-101",
        registration_number="JH05AB6799",
        final_score=0.90,
        breakdown=breakdown,
        why_match=["Plate match 9/10", "Colour match: white"],
        why_not=[]
    )
    fused = FusedPlate(
        text="JH05AB?799",
        format_valid=True,
        frames_used=8,
        mean_confidence=0.89,
        known_count=9,
        unknown_count=1
    )
    attr = VehicleAttributes(
        colour="white",
        colour_conf=0.90,
        body_type="suv",
        body_conf=0.85
    )
    case = CaseResult(
        case_id="case-test-1",
        clip_name="test.mp4",
        fused_plate=fused,
        attributes=attr,
        candidates=[candidate],
        decision="MATCH_FOUND",
        decision_reason="Highest-ranked candidate based on the evidence.",
        thresholds_used={"no_match_threshold": 0.52},
        weights_used={"plate": 0.65, "colour": 0.15, "body": 0.12, "visual": 0.08}
    )
    assert case.decision == "MATCH_FOUND"
    assert len(case.candidates) == 1
    assert case.candidates[0].registration_number == "JH05AB6799"


def test_export_json_schemas(tmp_path):
    schemas = export_json_schemas(tmp_path)
    assert "FrameOCRResult" in schemas
    assert "FusedPlate" in schemas
    assert "VehicleAttributes" in schemas
    assert "RegistryRecord" in schemas
    assert "Candidate" in schemas
    assert "CaseResult" in schemas

    for name in schemas:
        file_path = tmp_path / f"{name}.schema.json"
        assert file_path.exists()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "title" in data or "type" in data
