"""Tests for candidate matching, ranking, hard negatives, and abstention."""

from __future__ import annotations

import pytest
from pipeline.schemas import FusedPlate, PositionVote, RegistryRecord, VehicleAttributes
from ranking.ranker import rank_candidates


@pytest.fixture
def sample_registry():
    return [
        RegistryRecord(vehicle_id="v-1", registration_number="JH05AB6799", colour="white", body_type="suv", make="Mahindra", model="Scorpio"),
        RegistryRecord(vehicle_id="v-2", registration_number="JH05AB6798", colour="white", body_type="suv", make="Mahindra", model="Scorpio"), # 1 char diff
        RegistryRecord(vehicle_id="v-3", registration_number="JH05AB8799", colour="white", body_type="suv", make="Mahindra", model="Scorpio"), # confusable B/8
        RegistryRecord(vehicle_id="v-4", registration_number="DL08CA1234", colour="white", body_type="suv", make="Mahindra", model="Scorpio"), # same colour/body, diff plate
        RegistryRecord(vehicle_id="v-5", registration_number="MH12DE1433", colour="silver", body_type="hatchback", make="Maruti Suzuki", model="Swift"),
        RegistryRecord(vehicle_id="v-6", registration_number="KA03MN4821", colour="black", body_type="two-wheeler", make="Royal Enfield", model="Classic 350"),
    ]


def test_perfect_match(sample_registry):
    fused = FusedPlate(
        text="JH05AB6799",
        format_valid=True,
        frames_used=5,
        mean_confidence=0.92,
        known_count=10,
        unknown_count=0
    )
    attr = VehicleAttributes(colour="white", colour_conf=0.90, body_type="suv", body_conf=0.90)

    top_5, decision, reason = rank_candidates(fused, attr, sample_registry)
    assert decision == "MATCH_FOUND"
    assert top_5[0].registration_number == "JH05AB6799"
    assert top_5[0].final_score >= 0.88


def test_partial_plate_with_wildcard(sample_registry):
    # 'JH05AB?799' with wildcard at position 6
    fused = FusedPlate(
        text="JH05AB?799",
        format_valid=True,
        frames_used=5,
        mean_confidence=0.85,
        known_count=9,
        unknown_count=1
    )
    attr = VehicleAttributes(colour="white", colour_conf=0.90, body_type="suv", body_conf=0.90)

    top_5, decision, reason = rank_candidates(fused, attr, sample_registry)
    assert decision == "MATCH_FOUND"
    # Should rank the two compatible plates at top
    top_plates = [c.registration_number for c in top_5[:2]]
    assert "JH05AB6799" in top_plates or "JH05AB8799" in top_plates


def test_hard_negative_out_of_registry_returns_no_match(sample_registry):
    # Plate TN09ZZ4321 is completely absent from sample_registry
    fused = FusedPlate(
        text="TN09ZZ4321",
        format_valid=True,
        frames_used=5,
        mean_confidence=0.90,
        known_count=10,
        unknown_count=0
    )
    attr = VehicleAttributes(colour="red", colour_conf=0.85, body_type="hatchback", body_conf=0.85)

    top_5, decision, reason = rank_candidates(fused, attr, sample_registry, no_match_threshold=0.52)
    assert decision == "NO_CONFIDENT_MATCH"
    assert "threshold" in reason.lower() or "contradicts" in reason.lower()


def test_unreadable_plate_abstains_despite_strong_attributes(sample_registry):
    # All '?' plate
    fused = FusedPlate(
        text="??????????",
        format_valid=False,
        frames_used=5,
        mean_confidence=0.0,
        known_count=0,
        unknown_count=10
    )
    # Strong attributes: white SUV
    attr = VehicleAttributes(colour="white", colour_conf=0.99, body_type="suv", body_conf=0.99)

    top_5, decision, reason = rank_candidates(fused, attr, sample_registry)
    # System MUST NOT claim a confident match based on colour & body alone!
    assert decision == "NO_CONFIDENT_MATCH"
    assert "unreadable" in reason.lower()


def test_high_confidence_character_contradiction(sample_registry):
    # Fused plate has DL, but candidate has JH
    votes = [
        PositionVote(index=0, char_or_unknown="D", confidence=0.95),
        PositionVote(index=1, char_or_unknown="L", confidence=0.95),
        PositionVote(index=2, char_or_unknown="0", confidence=0.90),
        PositionVote(index=3, char_or_unknown="5", confidence=0.90),
        PositionVote(index=4, char_or_unknown="A", confidence=0.90),
        PositionVote(index=5, char_or_unknown="B", confidence=0.90),
        PositionVote(index=6, char_or_unknown="6", confidence=0.90),
        PositionVote(index=7, char_or_unknown="7", confidence=0.90),
        PositionVote(index=8, char_or_unknown="9", confidence=0.90),
        PositionVote(index=9, char_or_unknown="9", confidence=0.90),
    ]
    fused = FusedPlate(
        text="DL05AB6799",
        per_position=votes,
        format_valid=True,
        frames_used=5,
        mean_confidence=0.91,
        known_count=10,
        unknown_count=0
    )
    attr = VehicleAttributes(colour="white", colour_conf=0.90, body_type="suv", body_conf=0.90)

    # v-1 in registry is JH05AB6799 (contradicts high-conf 'D' and 'L')
    top_5, decision, reason = rank_candidates(fused, attr, sample_registry)
    if top_5[0].vehicle_id == "v-1":
        assert decision == "NO_CONFIDENT_MATCH"
