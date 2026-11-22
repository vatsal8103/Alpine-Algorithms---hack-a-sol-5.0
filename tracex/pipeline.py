"""End-to-end TRACE-X pipeline.

Stages:
    sample frames -> detect/track vehicles -> localise plates -> OCR
    -> multi-frame fusion -> attributes -> registry ranking -> reports

Every stage is fail-safe.  If detection or OCR is unavailable, the pipeline
continues with ``unknown`` attributes and a clear warning; it never invents a
character or a vehicle attribute.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .attributes import (
    AttributeExtractor,
    BodyResult,
    ColorResult,
    FrameAttributes,
    FusedAttributes,
    PlateStyleResult,
    canonical_color,
)
from .config import PipelineConfig, load_config
from .detection import Detection, PlateDetector, VehicleDetector, fallback_vehicle_box
from .fusion import fuse_plate, observation_from_hint, observation_from_ocr
from .io_utils import draw_label, ensure_dir, imwrite, relative_posix
from .matching import match_registry
from .ocr import AutoOCR, OCRLine, build_ocr, read_plate
from .plate import align_reading, clean_text, plate_likeness
from .registry import Registry, load_registry
from .schemas import (
    AnalysisResult,
    FrameEvidence,
    PlateObservation,
    VehicleAttributes,
    utc_now_iso,
)
from .store import JobStore
from .tracking import SimpleIoUTracker, compare_results
from .utils import new_job_id, seed_everything
from .video import Frame, FrameSet, frame_quality, sample_frames

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class Hints:
    plate: str = ""
    color: str = ""
    body_type: str = ""
    plate_style: str = ""
    make: str = ""
    model: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Hints":
        data = data or {}
        return cls(
            plate=str(data.get("plate") or data.get("plate_hint") or "").strip(),
            color=str(data.get("color") or data.get("color_hint") or "").strip(),
            body_type=str(data.get("body_type") or data.get("body_type_hint") or "").strip(),
            plate_style=str(data.get("plate_style") or data.get("plate_style_hint") or "").strip(),
            make=str(data.get("make") or data.get("make_hint") or "").strip(),
            model=str(data.get("model") or data.get("model_hint") or "").strip(),
        )

    def any(self) -> bool:
        return any([self.plate, self.color, self.body_type, self.plate_style, self.make, self.model])


@dataclass
class _PlateRead:
    line: OCRLine | None = None
    crop: np.ndarray | None = None
    plate_box: tuple[float, float, float, float] | None = None
    quality: float = 0.0
    score: float = 0.0
    source: str = ""


@dataclass
class _FrameCandidate:
    detection: Detection
    plate: _PlateRead
    attributes: FrameAttributes | None = None


class TraceXPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        registry: Registry | None = None,
        store: JobStore | None = None,
    ):
        self.config = config or load_config()
        seed_everything(self.config.project.seed)
        self.registry = registry or load_registry(self.config.paths.registry)
        self.store = store
        self.vehicle_detector = VehicleDetector(self.config.detector)
        self.plate_detector = PlateDetector(self.config.detector)
        self.ocr: AutoOCR = build_ocr(self.config.ocr)
        self.attribute_extractor = AttributeExtractor(self.config.attributes)
        self._iou_tracker = SimpleIoUTracker()
        self._warned: set[str] = set()
        self._last_frameset: FrameSet | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        source: str | Path,
        hints: Hints | dict[str, Any] | None = None,
        job_id: str | None = None,
        output_dir: str | Path | None = None,
        progress: ProgressCallback | None = None,
    ) -> AnalysisResult:
        job_id = job_id or new_job_id()
        hints = hints if isinstance(hints, Hints) else Hints.from_dict(hints)
        run_dir = Path(output_dir) if output_dir else Path(self.config.paths.runs) / job_id
        for sub in ("frames", "plates", "annotated"):
            ensure_dir(run_dir / sub)

        result = AnalysisResult(
            job_id=job_id,
            status="running",
            source_name=Path(source).name,
            source_path=str(source),
            disclaimer=self.config.project.disclaimer,
            pipeline_version=self.config.project.version,
        )
        warnings: list[str] = []
        self._warned = set()

        def report_progress(stage: str, pct: float, message: str) -> None:
            if progress:
                try:
                    progress(stage, pct, message)
                except Exception:
                    pass
            if self.store:
                self.store.update(job_id, status="running", stage=stage, progress=pct, message=message)

        report_progress("sampling", 0.02, "Sampling frames")
        try:
            frameset = sample_frames(
                source,
                max_frames=self.config.video.max_frames,
                stride=self.config.video.stride,
            )
        except Exception as exc:
            result.status = "failed"
            result.decision = "NO_CONFIDENT_MATCH"
            result.decision_reason = f"Could not read the source clip: {exc}"
            result.warnings = [result.decision_reason]
            result.completed_at = utc_now_iso()
            if self.store:
                self.store.fail(job_id, result.decision_reason)
            return result

        self._last_frameset = frameset
        result.source_kind = frameset.kind
        result.duration_s = round(float(frameset.duration_s), 3)
        result.fps = round(float(frameset.fps), 3)
        result.n_frames_sampled = len(frameset.frames)

        observations: list[PlateObservation] = []
        frame_attributes: list[FrameAttributes] = []
        frame_evidence: list[FrameEvidence] = []
        if hints.plate:
            observations.append(observation_from_hint(hints.plate))
            warnings.append(f"Investigator plate hint used as evidence: {clean_text(hints.plate)}")

        total = max(1, len(frameset.frames))
        for i, frame in enumerate(frameset.frames):
            pct = 0.08 + 0.62 * (i / total)
            report_progress("ocr", pct, f"Analysing frame {i + 1}/{total}")
            try:
                evidence, observation, attrs = self._process_frame(frame, i, run_dir, warnings)
            except Exception as exc:
                warnings.append(f"Frame {frame.index}: stage failed ({exc}); frame skipped.")
                evidence, observation, attrs = None, None, None
            if evidence is not None:
                frame_evidence.append(evidence)
            if observation is not None:
                observations.append(observation)
            if attrs is not None:
                frame_attributes.append(attrs)

        report_progress("fusion", 0.74, "Fusing multi-frame character evidence")
        try:
            fused = fuse_plate(observations, self.config.fusion)
        except Exception as exc:
            warnings.append(f"Plate fusion failed ({exc}); no plate characters available.")
            fused = fuse_plate([], self.config.fusion)
        warnings.extend(fused.warnings)

        # Fill per-frame aligned evidence using the fused length/pattern.
        for ev in frame_evidence:
            if not ev.ocr_text:
                continue
            try:
                aligned = align_reading(
                    ev.ocr_text,
                    len(fused.text),
                    pattern=fused.pattern,
                    char_confidences=ev.char_confidences,
                    line_confidence=ev.ocr_confidence,
                    class_correction=self.config.fusion.class_correction,
                )
                ev.aligned_text = aligned.text
                ev.aligned_confidences = [round(c, 4) for c in aligned.confidences]
            except Exception:
                ev.aligned_text = ev.ocr_text
                ev.aligned_confidences = list(ev.char_confidences)

        report_progress("attributes", 0.80, "Fusing colour, body type and plate style")
        try:
            fused_attrs = self.attribute_extractor.fuse(frame_attributes)
        except Exception as exc:
            warnings.append(f"Attribute fusion failed ({exc}); attributes marked unknown.")
            fused_attrs = FusedAttributes()
        fused_attrs = self._apply_hints(fused_attrs, hints, warnings)
        attributes = self._to_vehicle_attributes(fused_attrs)

        report_progress("matching", 0.88, "Ranking registry candidates")
        try:
            outcome = match_registry(fused, attributes, self.registry, self.config.matching)
        except Exception as exc:
            warnings.append(f"Registry matching failed ({exc}).")
            outcome = match_registry(fused, attributes, self.registry, self.config.matching)

        warnings.extend(self._collect_warnings(fused, attributes))
        result.status = "completed"
        result.completed_at = utc_now_iso()
        result.fused_plate = fused
        result.attributes = attributes
        result.frames = frame_evidence
        result.matches = outcome.matches
        result.near_misses = outcome.near_misses
        result.decision = outcome.decision
        result.decision_reason = outcome.decision_reason
        result.top_confidence = outcome.top_confidence
        result.warnings = _unique(warnings)
        result.config_summary = self._config_summary()

        report_progress("report", 0.94, "Building HTML/PDF report")
        self._write_reports(result, run_dir, warnings)
        result.warnings = _unique(warnings)
        self._save_result(result, run_dir)
        if self.store:
            self.store.complete(job_id, result)
        report_progress("done", 1.0, "Completed")
        return result

    def analyze_pair(
        self,
        source_a: str | Path,
        source_b: str | Path,
        hints_a: Hints | dict[str, Any] | None = None,
        hints_b: Hints | dict[str, Any] | None = None,
        job_id_a: str | None = None,
        job_id_b: str | None = None,
    ) -> tuple[AnalysisResult, AnalysisResult, Any]:
        a = self.analyze(source_a, hints_a, job_id=job_id_a)
        b = self.analyze(source_b, hints_b, job_id=job_id_b)
        tracking = compare_results(a, b, self.config.tracking)
        a.tracking = tracking
        b.tracking = tracking
        for result in (a, b):
            run_dir = Path(self.config.paths.runs) / result.job_id
            self._write_reports(result, run_dir, result.warnings)
            self._save_result(result, run_dir)
            if self.store:
                self.store.complete(result.job_id, result)
        return a, b, tracking

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame: Frame,
        ordinal: int,
        run_dir: Path,
        warnings: list[str],
    ) -> tuple[FrameEvidence | None, PlateObservation | None, FrameAttributes | None]:
        detections = self._detect(frame.image, warnings)
        detections = detections[: max(1, self.config.detector.max_vehicles)]

        best_det: Detection | None = None
        best_plate = _PlateRead()
        best_score = -1.0
        for det in detections:
            plate = self._read_plate_for_detection(frame.image, det, warnings)
            area = max(1.0, (det.box[2] - det.box[0]) * (det.box[3] - det.box[1]))
            score = plate.score + 0.08 * det.confidence + 1e-9 * area
            if score > best_score:
                best_score = score
                best_det = det
                best_plate = plate

        if best_det is None:
            return None, None, None

        quality = best_plate.quality or frame_quality(frame.image, best_det.box)["quality"]
        attrs = self.attribute_extractor.extract_frame(
            frame.image,
            best_det.box,
            best_plate.plate_box,
            best_det.label,
            best_det.confidence,
            quality,
            plate_crop=best_plate.crop,
        )

        # Save raw + annotated frames and the plate crop.
        frame_rel = None
        annotated_rel = None
        crop_rel = None
        if self.config.save_frames:
            raw_path = run_dir / "frames" / f"frame_{ordinal:04d}.jpg"
            if imwrite(raw_path, frame.image, self.config.jpeg_quality):
                frame_rel = relative_posix(raw_path, run_dir)
            annotated = frame.image.copy()
            draw_label(
                annotated,
                best_det.box,
                f"{best_det.label} {best_det.confidence:.2f}"
                + (f" track {best_det.track_id}" if best_det.track_id is not None else ""),
                (0, 220, 120),
            )
            if best_plate.plate_box is not None:
                text = best_plate.line.text if best_plate.line else "plate?"
                draw_label(annotated, best_plate.plate_box, text, (60, 220, 255), 2)
            ann_path = run_dir / "annotated" / f"frame_{ordinal:04d}.jpg"
            if imwrite(ann_path, annotated, self.config.jpeg_quality):
                annotated_rel = relative_posix(ann_path, run_dir)
            if best_plate.crop is not None:
                crop_path = run_dir / "plates" / f"frame_{ordinal:04d}_plate.jpg"
                if imwrite(crop_path, best_plate.crop, 94):
                    crop_rel = relative_posix(crop_path, run_dir)

        line = best_plate.line
        used = line is not None
        if used and line is not None:
            obs = observation_from_ocr(frame.index, frame.timestamp, line, quality)
            evidence = FrameEvidence(
                frame_index=frame.index,
                timestamp=round(float(frame.timestamp), 3),
                frame_path=frame_rel,
                annotated_path=annotated_rel,
                plate_crop_path=crop_rel,
                vehicle_bbox=[round(float(v), 2) for v in best_det.box],
                plate_bbox=[round(float(v), 2) for v in best_plate.plate_box] if best_plate.plate_box else None,
                ocr_text=line.text,
                ocr_confidence=round(float(line.confidence), 4),
                char_confidences=[round(float(c), 4) for c in line.char_confidences],
                quality=round(float(quality), 4),
                used=True,
                reason=f"Plate reading passed the likelihood gate ({best_plate.score:.2f}).",
                detector_label=best_det.label,
                detector_confidence=round(float(best_det.confidence), 4),
                track_id=best_det.track_id,
                plate_style=attrs.plate_style.style,
            )
            return evidence, obs, attrs

        evidence = FrameEvidence(
            frame_index=frame.index,
            timestamp=round(float(frame.timestamp), 3),
            frame_path=frame_rel,
            annotated_path=annotated_rel,
            plate_crop_path=crop_rel,
            vehicle_bbox=[round(float(v), 2) for v in best_det.box],
            plate_bbox=[round(float(v), 2) for v in best_plate.plate_box] if best_plate.plate_box else None,
            ocr_text=line.text if line else "",
            ocr_confidence=round(float(line.confidence), 4) if line else 0.0,
            char_confidences=[round(float(c), 4) for c in line.char_confidences] if line else [],
            quality=round(float(quality), 4),
            used=False,
            reason="No plate reading passed the likelihood gate."
            if best_plate.plate_box is not None
            else "No plate candidate found in this frame.",
            detector_label=best_det.label,
            detector_confidence=round(float(best_det.confidence), 4),
            track_id=best_det.track_id,
            plate_style=attrs.plate_style.style,
        )
        return evidence, None, attrs

    def _detect(self, image: np.ndarray, warnings: list[str]) -> list[Detection]:
        if self.vehicle_detector.available:
            detections = (
                self.vehicle_detector.track(image)
                if self.config.detector.use_tracking
                else self.vehicle_detector.detect(image)
            )
            if detections:
                return detections
            self._warn_once(warnings, "no_vehicle", "No vehicle was detected in at least one frame; using the fallback ROI.")
        else:
            self._warn_once(
                warnings,
                "detector",
                f"Vehicle detector unavailable ({self.vehicle_detector.status}); using fallback ROI and unknown body type.",
            )
        return [
            Detection(
                label="unknown",
                confidence=0.0,
                box=fallback_vehicle_box(image),
                class_id=-1,
                source="fallback",
                track_id=None,
            )
        ]

    def _read_plate_for_detection(
        self, image: np.ndarray, detection: Detection, warnings: list[str]
    ) -> _PlateRead:
        candidates = self.plate_detector.detect(image, detection.box)
        best = _PlateRead()
        for cand in candidates[: max(1, self.config.ocr.max_plate_candidates)]:
            try:
                crop, refined_box, quality = self.plate_detector.crop_and_refine(image, cand.box)
            except Exception:
                continue
            if crop is None or crop.size == 0:
                continue
            try:
                line = read_plate(crop, self.ocr, self.config.ocr)
            except Exception as exc:
                self._warn_once(warnings, "ocr", f"OCR failed on a plate crop ({exc}); continuing.")
                line = None
            if line is None:
                continue
            likeness = plate_likeness(line.text)
            if likeness < self.config.ocr.min_plate_likeness:
                continue
            score = (
                likeness
                * (0.45 + 0.55 * max(0.0, min(1.0, line.confidence)))
                * (0.50 + 0.50 * max(0.0, min(1.0, quality)))
                * (0.40 + 0.60 * max(0.0, min(1.0, cand.score)))
            )
            if score > best.score:
                best = _PlateRead(
                    line=line,
                    crop=crop,
                    plate_box=refined_box,
                    quality=float(quality),
                    score=float(score),
                    source=cand.source,
                )
            if line.known_chars >= 8 and line.confidence >= 0.55:
                break
        return best

    # ------------------------------------------------------------------
    # Hints, attributes, warnings
    # ------------------------------------------------------------------
    def _apply_hints(self, fused: FusedAttributes, hints: Hints, warnings: list[str]) -> FusedAttributes:
        if hints.color:
            color = canonical_color(hints.color)
            if color != "unknown":
                fused.color = ColorResult(
                    color=color,
                    confidence=max(0.90, fused.color.confidence),
                    distribution={**fused.color.distribution, color: 1.0},
                    histogram=fused.color.histogram,
                    evidence=f"investigator-provided colour: {color}",
                )
                fused.evidence.append(f"Investigator colour hint: {color}")
                warnings.append("Investigator colour hint used; extracted colour retained as secondary evidence.")
        if hints.body_type:
            body = hints.body_type.strip().lower()
            fused.body = BodyResult(body, 0.90, f"investigator-provided body type: {body}")
            fused.evidence.append(f"Investigator body-type hint: {body}")
        if hints.plate_style:
            style = hints.plate_style.strip().lower()
            if style in ("white", "yellow", "green"):
                fused.plate_style = PlateStyleResult(style, 0.90, f"investigator-provided plate style: {style}")
                fused.evidence.append(f"Investigator plate-style hint: {style}")
        if hints.make:
            fused.make = hints.make.strip()
            fused.model = hints.model.strip() or "unknown"
            fused.make_model_confidence = max(0.90, fused.make_model_confidence)
            fused.make_model_evidence = "investigator-provided make/model"
            fused.evidence.append(f"Investigator make/model hint: {fused.make} {fused.model}".strip())
        return fused

    def _to_vehicle_attributes(self, fused: FusedAttributes) -> VehicleAttributes:
        return VehicleAttributes(
            color=fused.color.color,
            color_confidence=round(float(fused.color.confidence), 4),
            color_distribution={k: round(float(v), 4) for k, v in fused.color.distribution.items()},
            body_type=fused.body.body_type,
            body_type_confidence=round(float(fused.body.confidence), 4),
            plate_style=fused.plate_style.style,
            plate_style_confidence=round(float(fused.plate_style.confidence), 4),
            make=fused.make,
            model=fused.model,
            make_model_confidence=round(float(fused.make_model_confidence), 4),
            evidence=list(fused.evidence),
            color_histogram=[round(float(v), 6) for v in fused.color.histogram],
        )

    def _collect_warnings(self, fused, attributes: VehicleAttributes) -> list[str]:
        warnings: list[str] = []
        if self.ocr.name == "template":
            warnings.append(
                "Template OCR fallback in use. Install PaddleOCR for real CCTV footage; "
                "the template backend is intended for the synthetic demo and tests."
            )
        if not self.vehicle_detector.available:
            warnings.append(
                f"Vehicle detector not active ({self.vehicle_detector.status}); body type is unknown "
                "unless an investigator hint is supplied."
            )
        if fused.known_chars == 0:
            warnings.append("No readable plate characters were recovered from this clip.")
        if attributes.color == "unknown":
            warnings.append("Colour could not be extracted; matching ignores colour.")
        if attributes.body_type == "unknown":
            warnings.append("Body type could not be determined; matching ignores body type.")
        warnings.append("Super-resolution was not used. Plate crops are upscaled with interpolation only.")
        return warnings

    def _config_summary(self) -> dict[str, Any]:
        return {
            "vehicle_detector": self.vehicle_detector.status,
            "plate_detector": self.plate_detector.status,
            "ocr_backend": self.ocr.name,
            "tracker": self.config.detector.tracker if self.config.detector.use_tracking else "disabled",
            "fusion": {
                "min_char_confidence": self.config.fusion.min_char_confidence,
                "min_vote_share": self.config.fusion.min_vote_share,
                "min_support_weight": self.config.fusion.min_support_weight,
            },
            "matching": {
                "min_confidence": self.config.matching.min_confidence,
                "min_score": self.config.matching.min_score,
                "min_known_chars": self.config.matching.min_known_chars,
                "min_margin": self.config.matching.min_margin,
            },
            "disclaimer": self.config.project.disclaimer,
        }

    def _warn_once(self, warnings: list[str], key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.append(message)

    # ------------------------------------------------------------------
    # Reports / persistence
    # ------------------------------------------------------------------
    def _write_reports(self, result: AnalysisResult, run_dir: Path, warnings: list[str]) -> None:
        try:
            from .report import build_reports

            html_path, pdf_path = build_reports(result, run_dir, self.config)
            result.report_html = relative_posix(html_path, run_dir) if html_path else None
            result.report_pdf = relative_posix(pdf_path, run_dir) if pdf_path else None
        except Exception as exc:
            warnings.append(f"Report generation failed: {exc}")
            result.report_html = None
            result.report_pdf = None

    def _save_result(self, result: AnalysisResult, run_dir: Path) -> None:
        try:
            path = run_dir / "result.json"
            path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            result.result_json = "result.json"
        except Exception:
            result.result_json = None


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
