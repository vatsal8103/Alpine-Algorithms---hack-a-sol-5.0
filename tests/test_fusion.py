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


def test_duplicates_of_one_frame_cannot_confirm_a_plate():
    f = make_frame_result(0, [("J", 0.9), ("H", 0.9), ("0", 0.9), ("5", 0.9), ("A", 0.9), ("B", 0.9), ("6", 0.9), ("7", 0.9), ("9", 0.9), ("9", 0.9)])
    fused = fuse_frames([f, f.model_copy(deep=True)])
    assert fused.text == "??????????"
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


def test_alternative_alone_cannot_supply_a_digit():
    """Grammar cannot promote a runner-up into an observed digit."""
    # Pos 2 is district code (digit). Frame predicts 'O' with '0' as alternative
    chars = [
        CharCandidate(char="M", prob=0.90, alternatives=[]),
        CharCandidate(char="P", prob=0.90, alternatives=[]),
        CharCandidate(char="O", prob=0.85, alternatives=[CharAlternative(char="0", prob=0.35)]),
        CharCandidate(char="4", prob=0.90, alternatives=[]),
        CharCandidate(char="Z", prob=0.90, alternatives=[]),
        CharCandidate(char="K", prob=0.90, alternatives=[]),
        CharCandidate(char="6", prob=0.90, alternatives=[]),
        CharCandidate(char="1", prob=0.90, alternatives=[]),
        CharCandidate(char="8", prob=0.90, alternatives=[]),
        CharCandidate(char="4", prob=0.90, alternatives=[]),
    ]
    frame = FrameOCRResult(
        track_id=72,
        frame_no=100,
        timestamp=1.5,
        bbox=[0.0, 0.0, 100.0, 50.0],
        plate_crop_path="crop_72.png",
        sharpness=85.0,
        plate_area_px=6000.0,
        chars=chars
    )
    fused = fuse_frames([frame, frame.model_copy(update={"frame_no": 101})])
    assert fused.text == "MP?4ZK6184"



def test_o_in_digit_slot_becomes_question_mark_without_supporting_alternative():
    """Unit test: 'O' in digit slot becomes '?' when alternatives do NOT support '0' (never invent)."""
    chars = [
        CharCandidate(char="M", prob=0.90, alternatives=[]),
        CharCandidate(char="P", prob=0.90, alternatives=[]),
        CharCandidate(char="O", prob=0.85, alternatives=[CharAlternative(char="Q", prob=0.30)]), # No '0' alternative!
        CharCandidate(char="4", prob=0.90, alternatives=[]),
        CharCandidate(char="Z", prob=0.90, alternatives=[]),
        CharCandidate(char="K", prob=0.90, alternatives=[]),
        CharCandidate(char="6", prob=0.90, alternatives=[]),
        CharCandidate(char="1", prob=0.90, alternatives=[]),
        CharCandidate(char="8", prob=0.90, alternatives=[]),
        CharCandidate(char="4", prob=0.90, alternatives=[]),
    ]
    frame = FrameOCRResult(
        track_id=72,
        frame_no=100,
        timestamp=1.5,
        bbox=[0.0, 0.0, 100.0, 50.0],
        plate_crop_path="crop_72.png",
        sharpness=85.0,
        plate_area_px=6000.0,
        chars=chars
    )
    fused = fuse_frames([frame])
    assert fused.text[2] == "?", f"Expected '?' at pos 2 without supporting alternative, got '{fused.text[2]}'"


def test_best_frame_ranking_by_sharpness_and_area():
    """Unit test: Frame ranking prioritizes high plate sharpness and larger plate area."""
    import numpy as np
    from pipeline.select_frames import select_best_frames

    # High sharpness, large area observation
    img_sharp = np.zeros((100, 200, 3), dtype=np.uint8)
    img_sharp[40:80, 50:150] = 255 # High contrast rectangle
    obs_sharp = {
        "frame_no": 10,
        "timestamp": 1.0,
        "bbox": [10.0, 10.0, 150.0, 90.0],
        "conf": 0.90,
        "sharpness": 250.0,
        "frame_image": img_sharp
    }

    # Low sharpness, small area observation
    img_blurry = np.zeros((100, 200, 3), dtype=np.uint8)
    obs_blurry = {
        "frame_no": 20,
        "timestamp": 2.0,
        "bbox": [10.0, 10.0, 50.0, 40.0],
        "conf": 0.60,
        "sharpness": 15.0,
        "frame_image": img_blurry
    }

    selected = select_best_frames([obs_blurry, obs_sharp], best_n=1)
    assert len(selected) == 1
    assert selected[0]["frame_no"] == 10


def test_no_invented_characters_on_zero_evidence():
    """Unit test: Zero evidence strictly outputs '?' for all positions with zero fabricated chars."""
    fused = fuse_frames([])
    assert fused.text == "??????????"
    assert fused.known_count == 0
    assert fused.unknown_count == 10
    for pos_vote in fused.per_position:
        assert pos_vote.char_or_unknown == "?"
