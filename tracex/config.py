"""Typed configuration loaded from YAML.

Everything tunable - thresholds, weights, paths, tracker choice, report
limits - lives in ``configs/config.yaml``.  Code should not contain magic
numbers that affect a decision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class ProjectConfig:
    name: str = "TRACE-X"
    version: str = "0.1.0"
    seed: int = 42
    disclaimer: str = (
        "Decision-support only. Results are the highest-ranked candidates "
        "based on the available evidence, not a definitive identification."
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProjectConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class PathsConfig:
    registry: str = "data/registry/vehicles.csv"
    runs: str = "runs"
    database: str = "runs/tracex.sqlite3"
    reference_gallery: str = "data/reference"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PathsConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class VideoConfig:
    max_frames: int = 36
    stride: int = 1
    jpeg_quality: int = 88

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VideoConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class DetectorConfig:
    enabled: bool = True
    vehicle_model: str = "yolov8s.pt"
    plate_model: str | None = None
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "cpu"
    allow_download: bool = True
    tracker: str = "bytetrack.yaml"
    use_tracking: bool = True
    max_vehicles: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DetectorConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class OCRConfig:
    backend: str = "auto"  # auto | paddle | template
    lang: str = "en"
    min_line_conf: float = 0.15
    min_char_conf: float = 0.22
    min_plate_likeness: float = 0.35
    upscale: int = 3
    use_angle_cls: bool = True
    max_plate_candidates: int = 2
    max_variants: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OCRConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class FusionConfig:
    """Controls the multi-frame character voting."""

    min_observations: int = 1
    min_chars_per_reading: int = 4
    min_plate_len: int = 7
    max_plate_len: int = 11
    min_support_weight: float = 0.12
    min_vote_share: float = 0.30
    min_char_confidence: float = 0.28
    agreement_bonus: float = 0.15
    class_correction: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FusionConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class AttributeConfig:
    min_color_conf: float = 0.30
    min_body_conf: float = 0.35
    make_model_min_conf: float = 0.45
    plate_style_min_conf: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AttributeConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class MatchConfig:
    """Weights and abstention thresholds for registry ranking."""

    top_k: int = 5
    weight_plate: float = 0.66
    weight_color: float = 0.15
    weight_body: float = 0.12
    weight_plate_style: float = 0.05
    weight_make_model: float = 0.07
    min_plate_style_conf: float = 0.35
    min_confidence: float = 0.45
    min_score: float = 0.52
    min_known_chars: int = 3
    min_margin: float = 0.05
    max_high_conf_conflicts: int = 2
    high_conf_threshold: float = 0.55
    confusion_similarity: float = 0.65

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MatchConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class ReportConfig:
    max_frames: int = 12
    pdf_engine: str = "reportlab"  # reportlab | print-html-only
    include_near_misses: bool = True
    thumbnail_width: int = 520
    plate_crop_width: int = 260

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReportConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 200
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "APIConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class TrackingConfig:
    enabled: bool = True
    same_vehicle_threshold: float = 0.60
    min_plate_known: int = 4
    weight_plate: float = 0.68
    weight_color: float = 0.12
    weight_body: float = 0.10
    weight_appearance: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrackingConfig":
        return cls(**_known_fields(cls, data or {}))


@dataclass
class PipelineConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    attributes: AttributeConfig = field(default_factory=AttributeConfig)
    matching: MatchConfig = field(default_factory=MatchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    api: APIConfig = field(default_factory=APIConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # ---- Backwards-compatible convenience properties ----
    @property
    def max_frames(self) -> int:
        return self.video.max_frames

    @property
    def frame_stride(self) -> int:
        return self.video.stride

    @property
    def jpeg_quality(self) -> int:
        return self.video.jpeg_quality

    @property
    def run_root(self) -> str:
        return self.paths.runs

    @property
    def registry_path(self) -> str:
        return self.paths.registry

    @property
    def max_report_frames(self) -> int:
        return self.report.max_frames

    @property
    def save_frames(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PipelineConfig":
        data = data or {}
        # Legacy flat keys are accepted so old JSON configs still load.
        video_data = dict(data.get("video") or {})
        for key in ("max_frames", "frame_stride", "stride", "jpeg_quality"):
            if key in data:
                video_data[key] = data[key]
        if "frame_stride" in video_data and "stride" not in video_data:
            video_data["stride"] = video_data.pop("frame_stride")
        paths_data = dict(data.get("paths") or {})
        for key in ("run_root", "runs", "registry_path", "registry", "database"):
            if key in data:
                paths_data[key] = data[key]
        if "run_root" in paths_data and "runs" not in paths_data:
            paths_data["runs"] = paths_data.pop("run_root")
        if "registry_path" in paths_data and "registry" not in paths_data:
            paths_data["registry"] = paths_data.pop("registry_path")

        report_data = dict(data.get("report") or {})
        if "max_report_frames" in data and "max_frames" not in report_data:
            report_data["max_frames"] = data["max_report_frames"]

        return cls(
            project=ProjectConfig.from_dict(data.get("project")),
            paths=PathsConfig.from_dict(paths_data),
            video=VideoConfig.from_dict(video_data),
            detector=DetectorConfig.from_dict(data.get("detector")),
            ocr=OCRConfig.from_dict(data.get("ocr")),
            fusion=FusionConfig.from_dict(data.get("fusion")),
            attributes=AttributeConfig.from_dict(data.get("attributes")),
            matching=MatchConfig.from_dict(data.get("matching")),
            report=ReportConfig.from_dict(report_data),
            api=APIConfig.from_dict(data.get("api")),
            tracking=TrackingConfig.from_dict(data.get("tracking")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load YAML (preferred) or JSON configuration."""

    if path is None:
        for candidate in (Path("configs/config.yaml"), Path("configs/default.yaml"), Path("configs/default.json")):
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return PipelineConfig()
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    return PipelineConfig.from_dict(raw)
