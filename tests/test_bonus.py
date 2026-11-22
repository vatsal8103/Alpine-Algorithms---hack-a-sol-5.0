"""Tests for Bonus Features: Make/Model Recognition and Cross-Camera Tracking."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.cross_camera import match_cross_camera
from pipeline.make_model import MakeModelClassifier
from pipeline.schemas import CaseResult, FusedPlate, VehicleAttributes


def test_make_model_classifier():
    clf = MakeModelClassifier()
    dummy_crop = np.zeros((150, 200, 3), dtype=np.uint8)
    make, model, conf = clf.predict(dummy_crop, body_type="suv")
    assert make == "Mahindra"
    assert model == "Scorpio"
    assert conf >= 0.70

    # Rare / unknown body type
    make_u, model_u, conf_u = clf.predict(dummy_crop, body_type="experimental_glider")
    assert make_u == "Unknown"
    assert conf_u < 0.45


def test_cross_camera_same_vehicle():
    case_a = CaseResult(
        case_id="case-camA",
        clip_name="jh05ab6799_camA.mp4",
        fused_plate=FusedPlate(text="JH05AB?799", format_valid=True),
        attributes=VehicleAttributes(colour="white", colour_conf=0.9, body_type="suv", body_conf=0.9),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )
    case_b = CaseResult(
        case_id="case-camB",
        clip_name="jh05ab6799_camB.mp4",
        fused_plate=FusedPlate(text="JH05??6799", format_valid=True),
        attributes=VehicleAttributes(colour="white", colour_conf=0.88, body_type="suv", body_conf=0.85),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )

    match = match_cross_camera(case_a, case_b, camera_a_id="cam-A", camera_b_id="cam-B", time_gap_seconds=120.0)
    assert match.same_vehicle
    assert match.match_score >= 0.60
    assert "Same vehicle confirmed" in match.explanation


def test_cross_camera_different_vehicle_contradiction():
    case_a = CaseResult(
        case_id="case-camA",
        clip_name="jh05ab6799_camA.mp4",
        fused_plate=FusedPlate(text="JH05AB6799", format_valid=True),
        attributes=VehicleAttributes(colour="white", colour_conf=0.9, body_type="suv", body_conf=0.9),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )
    case_b = CaseResult(
        case_id="case-camB",
        clip_name="dl08ca1234_camB.mp4",
        fused_plate=FusedPlate(text="DL08CA1234", format_valid=True), # Plate completely different
        attributes=VehicleAttributes(colour="silver", colour_conf=0.88, body_type="sedan", body_conf=0.85),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )

    match = match_cross_camera(case_a, case_b, camera_a_id="cam-A", camera_b_id="cam-B", time_gap_seconds=120.0)
    assert not match.same_vehicle
    assert match.match_score < 0.30
    assert "Contradictory plate characters" in match.explanation


def test_cross_camera_implausible_travel_time():
    case_a = CaseResult(
        case_id="case-camA",
        clip_name="jh05ab6799_camA.mp4",
        fused_plate=FusedPlate(text="JH05AB6799", format_valid=True),
        attributes=VehicleAttributes(colour="white", colour_conf=0.9, body_type="suv", body_conf=0.9),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )
    case_b = CaseResult(
        case_id="case-camB",
        clip_name="jh05ab6799_camB.mp4",
        fused_plate=FusedPlate(text="JH05AB6799", format_valid=True),
        attributes=VehicleAttributes(colour="white", colour_conf=0.9, body_type="suv", body_conf=0.9),
        candidates=[],
        decision="MATCH_FOUND",
        decision_reason="Match"
    )

    # Time gap is 5 seconds between camera poles 2 km apart (impossible)
    match = match_cross_camera(case_a, case_b, time_gap_seconds=5.0, min_travel_seconds=45.0)
    assert not match.same_vehicle
    assert not match.travel_time_feasible
    assert "Implausible travel time" in match.explanation
