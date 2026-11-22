"""Pydantic schemas shared by the pipeline, API, store and report renderer.

Every value that ends up in a report is represented here.  In particular,
``FusedPlate`` stores the confidence and supporting-frame count for every
character position, which makes the "never invent characters" guarantee
auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


class CharAlternative(BaseModel):
    char: str
    weight: float
    confidence: float
    support: int


class PlateObservation(BaseModel):
    """One OCR reading of one plate crop in one frame."""

    model_config = ConfigDict(extra="ignore")

    frame_index: int
    timestamp: float
    text: str
    confidence: float
    char_confidences: list[float] = Field(default_factory=list)
    quality: float = 0.5
    source: str = "ocr"  # ocr | investigator
    backend: str = "unknown"


class FrameEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame_index: int
    timestamp: float
    frame_path: str | None = None
    annotated_path: str | None = None
    plate_crop_path: str | None = None
    vehicle_bbox: list[float] | None = None
    plate_bbox: list[float] | None = None
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    char_confidences: list[float] = Field(default_factory=list)
    quality: float = 0.0
    used: bool = False
    reason: str = ""
    aligned_text: str = ""
    aligned_confidences: list[float] = Field(default_factory=list)
    detector_label: str = "unknown"
    detector_confidence: float = 0.0
    track_id: int | None = None
    plate_style: str = "unknown"


class FusedPlate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    pattern: str = ""
    char_confidences: list[float] = Field(default_factory=list)
    support: list[int] = Field(default_factory=list)
    alternatives: list[list[CharAlternative]] = Field(default_factory=list)
    known_chars: int = 0
    unknown_positions: list[int] = Field(default_factory=list)
    mean_confidence: float = 0.0
    n_frames_observed: int = 0
    n_frames_used: int = 0
    class_corrections: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def known_ratio(self) -> float:
        return self.known_chars / len(self.text) if self.text else 0.0


class VehicleAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    color: str = "unknown"
    color_confidence: float = 0.0
    color_distribution: dict[str, float] = Field(default_factory=dict)
    body_type: str = "unknown"
    body_type_confidence: float = 0.0
    plate_style: str = "unknown"
    plate_style_confidence: float = 0.0
    make: str = "unknown"
    model: str = "unknown"
    make_model_confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    color_histogram: list[float] = Field(default_factory=list)


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    vehicle_id: str
    plate: str
    make: str = "unknown"
    model: str = "unknown"
    color: str = "unknown"
    body_type: str = "unknown"
    plate_style: str = "white"
    category: str = "private"  # private | commercial | ev
    state: str = ""
    fuel: str = "unknown"
    year: int = 0
    owner_city: str = ""


class MatchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    vehicle: RegistryEntry
    score: float
    confidence: float
    plate_score: float
    color_score: float | None = None
    body_score: float | None = None
    plate_style_score: float | None = None
    make_model_score: float | None = None
    known_chars: int = 0
    unknown_chars: int = 0
    exact_plate: bool = False
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None


class MatchOutcome(BaseModel):
    decision: Literal["SHORTLIST", "NO_CONFIDENT_MATCH"]
    decision_reason: str
    matches: list[MatchResult] = Field(default_factory=list)
    near_misses: list[MatchResult] = Field(default_factory=list)
    top_confidence: float = 0.0
    runner_up_confidence: float = 0.0
    margin: float = 0.0
    thresholds: dict[str, float] = Field(default_factory=dict)


class TrackingResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    same_vehicle: bool
    confidence: float
    decision: str
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    plate_similarity: float = 0.0
    color_similarity: float | None = None
    body_similarity: float | None = None
    histogram_similarity: float | None = None
    known_both: int = 0
    job_id_a: str = ""
    job_id_b: str = ""


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    job_id: str
    status: Literal["queued", "running", "completed", "failed"] = "completed"
    created_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None
    source_name: str = ""
    source_path: str = ""
    source_kind: str = "video"
    duration_s: float = 0.0
    fps: float = 0.0
    n_frames_sampled: int = 0
    fused_plate: FusedPlate = Field(default_factory=FusedPlate)
    attributes: VehicleAttributes = Field(default_factory=VehicleAttributes)
    frames: list[FrameEvidence] = Field(default_factory=list)
    matches: list[MatchResult] = Field(default_factory=list)
    near_misses: list[MatchResult] = Field(default_factory=list)
    decision: Literal["SHORTLIST", "NO_CONFIDENT_MATCH"] = "NO_CONFIDENT_MATCH"
    decision_reason: str = ""
    top_confidence: float = 0.0
    report_html: str | None = None
    report_pdf: str | None = None
    result_json: str | None = None
    warnings: list[str] = Field(default_factory=list)
    config_summary: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = ""
    pipeline_version: str = "0.1.0"
    tracking: TrackingResult | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    source_name: str = ""
    created_at: str | None = None
    updated_at: str | None = None
    result: AnalysisResult | None = None
