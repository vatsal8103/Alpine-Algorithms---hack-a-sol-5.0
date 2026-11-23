"""Tests for end-to-end video pipeline."""

from __future__ import annotations

from pathlib import Path
import pytest
from pipeline.run_pipeline import run_pipeline
from pipeline.schemas import CaseResult


def test_pipeline_on_demo_clip():
    clip_path = "data/clips/jh05ab6799_camA.mp4"
    if not Path(clip_path).exists():
        pytest.skip(f"Clip {clip_path} not found")

    result = run_pipeline(clip_path, hints={"colour": "white", "body_type": "suv"})
    assert isinstance(result, CaseResult)
    assert result.clip_name == "jh05ab6799_camA.mp4"
    assert result.decision in ("MATCH_FOUND", "NO_CONFIDENT_MATCH")
    assert len(result.thresholds_used) > 0
    assert len(result.weights_used) > 0
    assert result.attributes.colour != ""
    assert result.attributes.body_type != ""


def test_pipeline_unreadable_graceful_abstention():
    clip_path = "data/clips/wb02xx0000_unreadable.mp4"
    if not Path(clip_path).exists():
        pytest.skip(f"Clip {clip_path} not found")

    result = run_pipeline(clip_path)
    assert isinstance(result, CaseResult)
    # Severely degraded clip must abstain with NO_CONFIDENT_MATCH
    assert result.decision == "NO_CONFIDENT_MATCH"
    assert "cannot verify" in result.decision_reason.lower() or "unreadable" in result.decision_reason.lower() or "threshold" in result.decision_reason.lower()


def test_two_line_scooter_plate_unrolling():
    """Verify that square two-wheeler plates are properly unrolled into single-line strips."""
    import numpy as np
    from pipeline.plate_stage import unroll_two_line_plate

    # Simulate square two-line scooter plate (aspect ratio 1.25)
    sq_plate = np.ones((80, 100, 3), dtype=np.uint8) * 220
    unrolled, is_two_line = unroll_two_line_plate(sq_plate)
    assert is_two_line is True
    assert unrolled is not None
    uh, uw = unrolled.shape[:2]
    # Unrolled aspect ratio should be elongated horizontal (>= 3.0)
    assert (uw / uh) >= 3.0

    # Simulate standard single-line rectangular car plate (aspect ratio 4.5)
    rect_plate = np.ones((40, 180, 3), dtype=np.uint8) * 220
    unrolled_rect, is_two_line_rect = unroll_two_line_plate(rect_plate)
    assert is_two_line_rect is False
    assert unrolled_rect.shape == rect_plate.shape


def test_plate_stage_loads_yolo11_champion():
    """Verify PlateStage loads and initializes the 50-epoch YOLO11 detector."""
    from pipeline.plate_stage import PlateStage
    stage = PlateStage()
    assert stage.plate_detector is not None


def test_vlm_engine_rules_and_structure():
    """Verify Qwen2-VL forensic engine prompt rules and interface."""
    from pipeline.vlm_engine import Qwen2VLLicensePlateRecognizer, INDIAN_VLM_SYSTEM_PROMPT
    assert "DL01AB1234" in INDIAN_VLM_SYSTEM_PROMPT
    assert "TWO-LINE" in INDIAN_VLM_SYSTEM_PROMPT
    assert "HSRP" in INDIAN_VLM_SYSTEM_PROMPT
    recognizer = Qwen2VLLicensePlateRecognizer()
    assert recognizer.model_name == "Qwen/Qwen2-VL-2B-Instruct"
    assert recognizer.load_in_4bit is True


def test_two_line_plate_multi_line_easyocr_merging():
    """Verify that multi-line stacked text blocks are merged in topological order."""
    from pipeline.plate_stage import parse_easyocr_multi_line

    # Simulated EasyOCR detection of a 2-line scooter plate (top line: MP 04, bottom line: CY 2090)
    mock_results = [
        ([ [10, 10], [90, 10], [90, 40], [10, 40] ], "MP 04", 0.94),
        ([ [10, 50], [120, 50], [120, 85], [10, 85] ], "CY 2090", 0.96)
    ]
    merged_text, merged_conf, outer_box = parse_easyocr_multi_line(mock_results)
    assert merged_text == "MP04CY2090"
    assert merged_conf > 0.90
    assert outer_box == [10.0, 10.0, 120.0, 85.0]

    # Test with HSRP blue IND tag present (must be ignored)
    mock_results_with_ind = [
        ([ [2, 10], [8, 10], [8, 40], [2, 40] ], "IND", 0.88),
        ([ [10, 10], [90, 10], [90, 40], [10, 40] ], "MP 04", 0.94),
        ([ [10, 50], [120, 50], [120, 85], [10, 85] ], "2354", 0.91)
    ]
    text_ind, _, _ = parse_easyocr_multi_line(mock_results_with_ind)
    assert text_ind == "MP042354"
