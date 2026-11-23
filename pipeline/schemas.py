"""TRACE-X Pydantic Schemas and JSON Schema generator.

All core data transfer objects (DTOs) for TRACE-X are specified here.
Ensures strict type-safety, validation, serialization, and reproducible auditing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# 1. OCR Result Schemas (Per Frame)
# ============================================================================

class CharAlternative(BaseModel):
    """An alternative character prediction from the CTC or recognition model."""
    model_config = ConfigDict(extra="ignore")

    char: str = Field(..., description="Single alphanumeric character or symbol")
    prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability")


class CharCandidate(BaseModel):
    """A character candidate at a specific position in an OCR reading."""
    model_config = ConfigDict(extra="ignore")

    char: str = Field(..., description="Top predicted character")
    prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence of top character")
    alternatives: list[CharAlternative] = Field(
        default_factory=list,
        description="Top-N alternative candidates at this position"
    )


class FrameOCRResult(BaseModel):
    """OCR inference output for a single video frame and tracked vehicle."""
    model_config = ConfigDict(extra="ignore")

    track_id: int | None = Field(default=None, description="ByteTrack vehicle track ID")
    frame_no: int = Field(..., ge=0, description="Zero-indexed frame number in video")
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds from video start")
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in pixels of vehicle or plate"
    )
    plate_crop_path: str = Field(..., description="Path to cropped plate image artifact")
    sharpness: float = Field(..., ge=0.0, description="Variance of Laplacian sharpness score")
    plate_area_px: float = Field(..., ge=0.0, description="Area of plate crop in pixels")
    chars: list[CharCandidate] = Field(
        default_factory=list,
        description="Sequence of character candidates recognized in this frame"
    )


# ============================================================================
# 2. Multi-Frame Fusion Schemas
# ============================================================================

class FrameVote(BaseModel):
    """Vote contribution from a specific frame for a character position."""
    model_config = ConfigDict(extra="ignore")

    frame_no: int = Field(..., description="Frame index that cast this vote")
    char: str = Field(..., description="Character voted for")
    prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated character probability in frame")
    weight: float = Field(..., ge=0.0, description="Normalized weight assigned to this frame")


class PositionVote(BaseModel):
    """Aggregated evidence for a single character slot across all frames."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    index: int = Field(..., ge=0, description="0-indexed position along plate string")
    char_or_unknown: str = Field(
        ...,
        alias="char_or_?",
        serialization_alias="char_or_?",
        min_length=1,
        max_length=1,
        description="Consensus character or '?' if confidence/margin insufficient"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Posterior probability for accepted character, or margin confidence"
    )
    votes: list[FrameVote] = Field(
        default_factory=list,
        description="Breakdown of individual frame votes contributing to this position"
    )
    explanation: str | None = Field(
        default=None,
        description="Human-readable reason if marked '?' (e.g. margin < threshold)"
    )
    candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top candidate characters and posterior probabilities for UI inspection/tooltips"
    )


class TrackFrameEvidence(BaseModel):
    """Forensic evidence for a single video frame of a vehicle track."""
    model_config = ConfigDict(extra="ignore")

    frame_no: int = Field(..., ge=0, description="Video frame sequence number")
    annotated_timestamp: float | None = Field(default=None, ge=0.0)
    timestamp: float = Field(..., ge=0.0, description="Timestamp in seconds")
    bbox: list[float] = Field(..., description="Vehicle bounding box [x1, y1, x2, y2]")
    plate_bbox: list[float] | None = Field(default=None, description="Plate bounding box [px1, py1, px2, py2] in frame coordinates")
    sharpness: float = Field(default=0.0, description="Variance of Laplacian sharpness score")
    plate_area: float = Field(default=0.0, description="Plate crop area in pixels")
    plate_crop_url: str = Field(default="", description="Web URL or path to plate crop image")
    raw_text: str = Field(default="", description="Raw OCR prediction text")
    confidence: float = Field(default=0.0, description="Overall OCR confidence for this frame")
    chars: list[CharCandidate] = Field(default_factory=list, description="Per-character candidates and alternatives")


class FusedPlate(BaseModel):
    """Final consensus plate produced by multi-frame fusion."""
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., description="Fused plate text with '?' for uncertain characters")
    per_position: list[PositionVote] = Field(
        default_factory=list,
        description="Evidence and vote breakdown per position"
    )
    format_valid: bool = Field(
        default=False,
        description="Whether plate text conforms to Indian license plate grammar"
    )
    frames_used: int = Field(
        default=0,
        ge=0,
        description="Count of high-quality frames accepted into fusion"
    )
    mean_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean confidence across known positions"
    )
    known_count: int = Field(default=0, ge=0, description="Number of resolved characters")
    unknown_count: int = Field(default=0, ge=0, description="Number of '?' positions")


# ============================================================================
# 3. Vehicle Attributes Schemas
# ============================================================================

class VehicleAttributes(BaseModel):
    """Extracted visual attributes for a detected vehicle."""
    model_config = ConfigDict(extra="ignore")

    colour: str = Field(..., description="Dominant detected vehicle colour name")
    colour_conf: float = Field(..., ge=0.0, le=1.0, description="Confidence in detected colour")
    body_type: str = Field(
        ...,
        description="Classified body type (hatchback, sedan, SUV, two-wheeler, auto-rickshaw, truck, bus)"
    )
    body_conf: float = Field(..., ge=0.0, le=1.0, description="Confidence in classified body type")
    make: str | None = Field(default=None, description="Predicted manufacturer/make if identified")
    model: str | None = Field(default=None, description="Predicted vehicle model if identified")
    colour_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate colours"
    )
    body_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across candidate body types"
    )
    plate_style: str = Field(
        default="white",
        description="Detected plate background style (white, yellow, green, black)"
    )
    plate_style_conf: float = Field(default=0.0, ge=0.0, le=1.0)


# ============================================================================
# 4. Registry Record Schemas
# ============================================================================

class RegistryRecord(BaseModel):
    """A vehicle entry in the regional registration database."""
    model_config = ConfigDict(extra="ignore")

    vehicle_id: str = Field(..., description="Unique registry database primary key")
    registration_number: str = Field(..., description="Standard registration plate string (e.g. JH05AB6799)")
    colour: str = Field(..., description="Registered body colour")
    body_type: str = Field(..., description="Registered vehicle body type")
    make: str = Field(..., description="Registered vehicle make")
    model: str = Field(..., description="Registered vehicle model")
    fuel: str | None = Field(default=None, description="Fuel type (petrol, diesel, cng, electric)")
    year: int | None = Field(default=None, ge=1970, le=2030, description="Year of registration")
    owner_city: str | None = Field(default=None, description="City / RTO jurisdiction")


# ============================================================================
# 5. Matching and Candidate Ranking Schemas
# ============================================================================

class ScoreBreakdown(BaseModel):
    """Component-level similarity scores for an evaluated candidate."""
    model_config = ConfigDict(extra="ignore")

    plate: float = Field(..., ge=0.0, le=1.0, description="Positional partial-plate similarity")
    colour: float = Field(..., ge=0.0, le=1.0, description="Colour compatibility score")
    body: float = Field(..., ge=0.0, le=1.0, description="Body-type compatibility score")
    visual: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Appearance / embedding similarity score"
    )


class Candidate(BaseModel):
    """A shortlisted candidate vehicle from the registry with ranking rationale."""
    model_config = ConfigDict(extra="ignore")

    vehicle_id: str = Field(..., description="Registry record identifier")
    registration_number: str = Field(..., description="Full registration number")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Weighted composite similarity score")
    breakdown: ScoreBreakdown = Field(..., description="Component score breakdown")
    why_match: list[str] = Field(
        default_factory=list,
        description="Evidence supporting this candidate match"
    )
    why_not: list[str] = Field(
        default_factory=list,
        description="Evidence against this candidate or explaining why it is ranked lower"
    )


# ============================================================================
# 6. Case Result Schemas
# ============================================================================

class VehicleScanItem(BaseModel):
    """Forensic scan and recovery details for a single vehicle tracked in the video."""
    model_config = ConfigDict(extra="ignore")

    track_id: int = Field(..., description="Unique tracker ID for this vehicle")
    frame_count: int = Field(..., description="Number of observed video frames")
    thumbnail_url: str = Field(default="", description="Path to cropped vehicle thumbnail")
    plate_crop_url: str = Field(default="", description="Path to cropped plate thumbnail")
    fused_plate: FusedPlate = Field(..., description="Consensus plate for this vehicle")
    attributes: VehicleAttributes = Field(..., description="Colour, body type, make/model")
    candidates: list[Candidate] = Field(default_factory=list, description="Top-5 shortlist")
    decision: Literal["MATCH_FOUND", "NO_CONFIDENT_MATCH"] = Field(..., description="Decision")
    decision_reason: str = Field(..., description="Plain-English explanation")
    top_score: float = Field(default=0.0, description="Highest candidate score")
    best_frames: list[TrackFrameEvidence] = Field(
        default_factory=list,
        description="Top 5 forensic frames ranked by plate sharpness, area, and OCR confidence"
    )


class CaseResult(BaseModel):
    """Complete end-to-end investigation result for a CCTV video clip."""
    model_config = ConfigDict(extra="ignore")

    case_id: str = Field(..., description="Unique case identifier (UUID or hash)")
    clip_name: str = Field(..., description="Source video file basename")
    fused_plate: FusedPlate = Field(..., description="Multi-frame fused plate result for target vehicle")
    attributes: VehicleAttributes = Field(..., description="Extracted vehicle visual attributes")
    candidates: list[Candidate] = Field(
        default_factory=list,
        max_length=5,
        description="Top-5 ranked candidates or empty if no confident match"
    )
    decision: Literal["MATCH_FOUND", "NO_CONFIDENT_MATCH"] = Field(
        ...,
        description="Final decision: either shortlisted candidates found or abstained"
    )
    decision_reason: str = Field(
        ...,
        description="Plain-English explanation of the decision rationale"
    )
    all_vehicles: list[VehicleScanItem] = Field(
        default_factory=list,
        description="Comprehensive scan of all vehicles identified across video"
    )
    annotated_video_path: str | None = Field(
        default=None,
        description="Relative path to rendered video with bounding boxes and plate overlays"
    )
    original_video_url: str | None = Field(
        default=None,
        description="Relative path or URL to original CCTV video file"
    )
    thresholds_used: dict[str, float] = Field(
        default_factory=dict,
        description="Threshold parameters applied during this analysis"
    )
    weights_used: dict[str, float] = Field(
        default_factory=dict,
        description="Ranking weights applied during this analysis"
    )
    created_at: str = Field(
        default_factory=utc_now_iso,
        description="UTC ISO timestamp of analysis completion"
    )


# ============================================================================
# JSON Schema Export Helper
# ============================================================================

def export_json_schemas(output_dir: str | Path = "configs/schemas") -> dict[str, dict[str, Any]]:
    """Export standard JSON Schemas for all primary TRACE-X Pydantic models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models: dict[str, type[BaseModel]] = {
        "FrameOCRResult": FrameOCRResult,
        "FusedPlate": FusedPlate,
        "VehicleAttributes": VehicleAttributes,
        "RegistryRecord": RegistryRecord,
        "Candidate": Candidate,
        "CaseResult": CaseResult,
    }

    schemas: dict[str, dict[str, Any]] = {}
    for name, model_cls in models.items():
        schema = model_cls.model_json_schema()
        schemas[name] = schema
        schema_file = out_path / f"{name}.schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    return schemas


if __name__ == "__main__":
    schemas = export_json_schemas()
    print(f"Exported {len(schemas)} JSON Schemas to configs/schemas/")
