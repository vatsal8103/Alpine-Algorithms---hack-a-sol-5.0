"""Tests for TRACE-X Master Configuration in configs/config.yaml."""

from __future__ import annotations

import json
from pathlib import Path
import yaml
import pytest


@pytest.fixture
def config():
    config_path = Path("configs/config.yaml")
    assert config_path.exists(), "configs/config.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def test_config_required_sections(config):
    required = [
        "project",
        "paths",
        "video",
        "detector",
        "ocr",
        "fusion",
        "confusable_pairs",
        "attributes",
        "ranking",
        "api",
        "report"
    ]
    for section in required:
        assert section in config, f"Missing required section: {section}"


def test_config_video_tunable_values(config):
    vid = config["video"]
    assert "frame_sampling_rate" in vid
    assert "best_n_frames" in vid
    assert "min_sharpness" in vid
    assert "min_plate_area_px" in vid
    assert vid["best_n_frames"] >= 1


def test_config_fusion_thresholds(config):
    fus = config["fusion"]
    assert "fusion_accept_threshold" in fus
    assert "margin_threshold" in fus
    assert 0.0 <= fus["fusion_accept_threshold"] <= 1.0
    assert 0.0 <= fus["margin_threshold"] <= 1.0


def test_config_ranking_weights_and_thresholds(config):
    rank = config["ranking"]
    assert "weights" in rank
    weights = rank["weights"]
    for w in ["plate", "colour", "body", "visual"]:
        assert w in weights
        assert weights[w] >= 0.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-4

    assert "no_match_threshold" in rank
    assert "margin_threshold" in rank
    assert 0.0 <= rank["no_match_threshold"] <= 1.0


def test_config_confusable_pairs(config):
    pairs = config["confusable_pairs"]
    assert len(pairs) >= 5
    pair_tuples = [tuple(p) for p in pairs]
    assert ("O", "0") in pair_tuples or ["O", "0"] in pairs
    assert ("B", "8") in pair_tuples or ["B", "8"] in pairs


def test_similarity_matrices_loadable(config):
    colour_path = Path(config["paths"]["colour_similarity_path"])
    body_path = Path(config["paths"]["body_type_partial_credit_path"])

    assert colour_path.exists(), f"Colour similarity table missing at {colour_path}"
    assert body_path.exists(), f"Body type similarity table missing at {body_path}"

    with open(colour_path, "r", encoding="utf-8") as f:
        colour_mat = json.load(f)
    assert "white" in colour_mat
    assert colour_mat["white"]["silver"] > 0.0

    with open(body_path, "r", encoding="utf-8") as f:
        body_mat = json.load(f)
    assert "hatchback" in body_mat
    assert body_mat["hatchback"]["sedan"] > 0.0
