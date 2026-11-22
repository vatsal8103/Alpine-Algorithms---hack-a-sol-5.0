"""Tests for multi-frame evidential consensus fusion."""

from __future__ import annotations

import pytest
from fusion.fuser import fuse_frames
from pipeline.schemas import CharAlternative, CharCandidate, FrameOCRResult


def make_frame_result(
    frame_no: int,
    char_tuples: list[tuple[str, float]],
    sharpness: float = 50.0,
    area: float = 5000.0
) -> FrameOCRResult:
    chars = [
        CharCandidate(
            char=c,
            prob=p,
            alternatives=[CharAlternative(char="0" if c == "O" else ("8" if c == "B" else "X"), prob=0.10)]
        )
        for c, p in char_tuples
    ]
    return FrameOCRResult(
        track_id=1,
        frame_no=frame_no,
        timestamp=frame_no * 0.1,
        bbox=[0.0, 0.0, 100.0, 50.0],
        plate_crop_path=f"crop_{frame_no}.png",
        sharpness=sharpness,
        plate_area_px=area,
        chars=chars
    )


def test_clean_frames_exact_fusion():
    clean_plate = "JH05AB6799"
    frames = [
        make_frame_result(i, [(c, 0.95) for c in clean_plate], sharpness=80.0)
        for i in range(5)
    ]
    fused = fuse_frames(frames)
    assert fused.text == "JH05AB6799"
    assert fused.known_count == 10
    assert fused.unknown_count == 0
    assert fused.mean_confidence >= 0.85
    assert fused.format_valid


def test_heavy_noise_produces_question_marks():
    # Frame 1 and 2 give low confidence on digits 6 and 7
    noisy_chars_1 = [
        ("J", 0.90), ("H", 0.90), ("0", 0.85), ("5", 0.85),
        ("A", 0.85), ("B", 0.85), ("6", 0.35), ("7", 0.36), ("9", 0.90), ("9", 0.90)
    ]
    noisy_chars_2 = [
        ("J", 0.90), ("H", 0.90), ("0", 0.85), ("5", 0.85),
        ("A", 0.85), ("B", 0.85), ("8", 0.34), ("1", 0.33), ("9", 0.90), ("9", 0.90)
    ]
    frames = [
        make_frame_result(0, noisy_chars_1, sharpness=20.0),
        make_frame_result(1, noisy_chars_2, sharpness=22.0)
    ]
    fused = fuse_frames(frames, accept_threshold=0.50, margin_threshold=0.12)
    # Positions 6 and 7 should safely be '?'
    assert fused.text[6] == "?" or fused.text[7] == "?"
    assert fused.unknown_count >= 1


def test_confusable_pair_handling():
    # '0' vs 'O' in district code (position 2 should be digit)
    frame1 = make_frame_result(0, [
        ("J", 0.9), ("H", 0.9), ("O", 0.55), ("5", 0.9),
        ("A", 0.9), ("B", 0.9), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)
    ])
    frame2 = make_frame_result(1, [
        ("J", 0.9), ("H", 0.9), ("0", 0.60), ("5", 0.9),
        ("A", 0.9), ("B", 0.9), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)
    ])
    fused = fuse_frames([frame1, frame2])
    # Grammar prior and evidence should favor '0' over 'O'
    assert fused.text[2] == "0" or fused.text[2] == "?"


def test_contradictory_frames_yield_safe_abstention():
    # Two frames completely contradict each other at position 5
    f1 = make_frame_result(0, [("J", 0.9), ("H", 0.9), ("0", 0.9), ("5", 0.9), ("A", 0.9), ("B", 0.51), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)])
    f2 = make_frame_result(1, [("J", 0.9), ("H", 0.9), ("0", 0.9), ("5", 0.9), ("A", 0.9), ("K", 0.51), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)])
    fused = fuse_frames([f1, f2], accept_threshold=0.50, margin_threshold=0.12)
    # Position 5 must be '?' due to tie / lack of margin
    assert fused.text[5] == "?"


def test_single_frame_fusion():
    f = make_frame_result(0, [("J", 0.9), ("H", 0.9), ("0", 0.9), ("5", 0.9), ("A", 0.9), ("B", 0.9), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)])
    fused = fuse_frames([f])
    assert fused.text == "JH05AB6799"
    assert fused.frames_used == 1


def test_zero_frames_fusion():
    fused = fuse_frames([])
    assert fused.text == "??????????"
    assert fused.known_count == 0
    assert fused.frames_used == 0


def test_property_never_accept_below_threshold():
    """Property test: No character with posterior below accept_threshold should ever be in fused text."""
    threshold = 0.55
    chars = [("J", 0.40), ("H", 0.40)] # All low confidence
    f = make_frame_result(0, chars)
    fused = fuse_frames([f], accept_threshold=threshold)
    for c in fused.text:
        assert c == "?", f"Character '{c}' accepted below threshold {threshold}!"
