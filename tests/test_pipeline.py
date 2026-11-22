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
