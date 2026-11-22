"""Pipeline package for video decoding, vehicle tracking, plate extraction and attribute inference."""

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
)

__all__ = [
    "CharAlternative",
    "CharCandidate",
    "FrameOCRResult",
    "FrameVote",
    "PositionVote",
    "FusedPlate",
    "VehicleAttributes",
    "RegistryRecord",
    "ScoreBreakdown",
    "Candidate",
    "CaseResult",
]
