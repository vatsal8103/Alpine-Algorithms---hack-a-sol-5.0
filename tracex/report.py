"""HTML and PDF analysis reports.

The HTML report is print-ready and self-contained (images embedded as data
URIs).  The PDF is generated with ReportLab so the demo works on Windows
without native GTK/WeasyPrint dependencies.  Both show:

* the decision (shortlist vs no confident match);
* the fused plate with per-character confidence and support;
* every sampled frame, its OCR reading and per-character confidence;
* the top-5 ranked candidates, reasons and conflicts;
* methodology, warnings and limitations.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import PipelineConfig
from .io_utils import image_to_data_uri
from .schemas import AnalysisResult, FrameEvidence, MatchResult

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _generated_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _select_frames(frames: list[FrameEvidence], limit: int) -> list[FrameEvidence]:
    used = [f for f in frames if f.used]
    unused = [f for f in frames if not f.used]
    selected = used[:limit]
    if len(selected) < limit:
        selected.extend(unused[: limit - len(selected)])
    return selected


def _frame_rows(result: AnalysisResult, run_dir: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in _select_frames(result.frames, config.report.max_frames):
        annotated = run_dir / ev.annotated_path if ev.annotated_path else None
        raw = run_dir / ev.frame_path if ev.frame_path else None
        crop = run_dir / ev.plate_crop_path if ev.plate_crop_path else None
        main = annotated if annotated and annotated.exists() else raw
        aligned_text = ev.aligned_text or ev.ocr_text
        confs = ev.aligned_confidences or ev.char_confidences
        pairs = [
            {"char": ch, "confidence": float(confs[i]) if i < len(confs) else 0.0}
            for i, ch in enumerate(aligned_text)
        ]
        rows.append(
            {
                "index": ev.frame_index,
                "timestamp": ev.timestamp,
                "ocr_text": ev.ocr_text or "—",
                "ocr_confidence": ev.ocr_confidence,
                "quality": ev.quality,
                "used": ev.used,
                "reason": ev.reason,
                "track_id": ev.track_id,
                "detector_label": ev.detector_label,
                "plate_style": ev.plate_style,
                "image_uri": image_to_data_uri(main, max_width=config.report.thumbnail_width) if main and main.exists() else None,
                "plate_uri": image_to_data_uri(crop, max_width=config.report.plate_crop_width) if crop and crop.exists() else None,
                "pairs": pairs,
            }
        )
    return rows


def _match_rows(matches: list[MatchResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in matches:
        rows.append(
            {
                "rank": m.rank,
                "plate": m.vehicle.plate,
                "make": m.vehicle.make,
                "model": m.vehicle.model,
                "color": m.vehicle.color,
                "body_type": m.vehicle.body_type,
                "plate_style": m.vehicle.plate_style,
                "category": m.vehicle.category,
                "state": m.vehicle.state,
                "confidence": m.confidence,
                "score": m.score,
                "plate_score": m.plate_score,
                "color_score": m.color_score,
                "body_score": m.body_score,
                "plate_style_score": m.plate_style_score,
                "make_model_score": m.make_model_score,
                "known_chars": m.known_chars,
                "unknown_chars": m.unknown_chars,
                "exact_plate": m.exact_plate,
                "reasons": m.reasons,
                "conflicts": m.conflicts,
                "rejected": m.rejected,
                "rejection_reason": m.rejection_reason,
            }
        )
    return rows


def build_html_report(result: AnalysisResult, run_dir: Path, config: PipelineConfig) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    plate = result.fused_plate
    plate_chars = []
    for i, ch in enumerate(plate.text):
        alts = plate.alternatives[i] if i < len(plate.alternatives) else []
        plate_chars.append(
            {
                "position": i + 1,
                "char": ch,
                "confidence": plate.char_confidences[i] if i < len(plate.char_confidences) else 0.0,
                "support": plate.support[i] if i < len(plate.support) else 0,
                "unknown": ch == "?",
                "alternatives": [
                    {"char": a.char, "weight": a.weight, "confidence": a.confidence, "support": a.support}
                    for a in alts
                ],
            }
        )
    context = {
        "result": result,
        "generated_at": _generated_at(),
        "disclaimer": result.disclaimer or config.project.disclaimer,
        "decision_title": (
            "SHORTLIST — highest-ranked candidates based on the evidence"
            if result.decision == "SHORTLIST"
            else "NO CONFIDENT MATCH"
        ),
        "decision_class": "ok" if result.decision == "SHORTLIST" else "warn",
        "plate_chars": plate_chars,
        "frames": _frame_rows(result, run_dir, config),
        "matches": _match_rows(result.matches),
        "near_misses": _match_rows(result.near_misses),
        "config_summary": result.config_summary,
        "warnings": result.warnings,
        "tracking": result.tracking,
        "has_frames": bool(result.frames),
    }
    html_text = template.render(**context)
    path = run_dir / "report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PDF (ReportLab)
# ---------------------------------------------------------------------------
def build_pdf_report(result: AnalysisResult, run_dir: Path, config: PipelineConfig) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        HRFlowable,
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7, leading=9, textColor=colors.HexColor("#555555")))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8, leading=11))
    styles.add(ParagraphStyle(name="MonoSmall", parent=styles["BodyText"], fontName="Courier", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Reason", parent=styles["BodyText"], fontSize=8, leading=10, leftIndent=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#123b4a")))

    pdf_path = run_dir / "report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"TRACE-X report {result.job_id}",
        author="TRACE-X",
    )
    story: list[Any] = []

    def p(text: str, style: str = "BodyText") -> Paragraph:
        return Paragraph(html.escape(str(text)).replace("\n", "<br/>"), styles[style])

    def section(title: str) -> None:
        story.append(Paragraph(html.escape(title), styles["Section"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#9fb7bf")))
        story.append(Spacer(1, 3))

    story.append(Paragraph("TRACE-X — Vehicle Shortlist Analysis Report", styles["Title"]))
    story.append(p(f"Case ID: {result.job_id}   |   Generated: {_generated_at()}   |   Source: {result.source_name}", "Tiny"))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Decision-support only.</b> "
            + html.escape(result.disclaimer or config.project.disclaimer),
            styles["SmallBody"],
        )
    )
    story.append(Spacer(1, 8))

    decision_text = (
        "SHORTLIST — highest-ranked candidates based on the evidence"
        if result.decision == "SHORTLIST"
        else "NO CONFIDENT MATCH"
    )
    decision_color = colors.HexColor("#d9f2e5") if result.decision == "SHORTLIST" else colors.HexColor("#ffe3d6")
    decision_table = Table([[Paragraph(f"<b>{html.escape(decision_text)}</b>", styles["Heading3"])]], colWidths=[doc.width])
    decision_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), decision_color),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#4d7c8a")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(decision_table)
    story.append(Spacer(1, 4))
    story.append(p(result.decision_reason, "SmallBody"))
    story.append(Spacer(1, 6))

    summary = [
        ["Source type", result.source_kind, "Frames sampled", str(result.n_frames_sampled)],
        ["Duration", f"{result.duration_s:.2f} s", "FPS", f"{result.fps:.2f}"],
        ["Fused plate", result.fused_plate.text or "—", "Known characters", f"{result.fused_plate.known_chars}/{len(result.fused_plate.text)}"],
        ["Colour", f"{result.attributes.color} ({result.attributes.color_confidence:.2f})", "Body type", f"{result.attributes.body_type} ({result.attributes.body_type_confidence:.2f})"],
        ["Plate style", f"{result.attributes.plate_style} ({result.attributes.plate_style_confidence:.2f})", "Top confidence", f"{result.top_confidence:.2f}"],
    ]
    summary_table = Table(summary, colWidths=[doc.width * 0.18, doc.width * 0.32, doc.width * 0.20, doc.width * 0.30])
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d5da")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5f7")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef5f7")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 8))

    section("Fused plate evidence")
    if result.fused_plate.text:
        plate_cells = []
        conf_cells = []
        support_cells = []
        for i, ch in enumerate(result.fused_plate.text):
            plate_cells.append(Paragraph(f"<b>{html.escape(ch)}</b>", styles["MonoSmall"]))
            conf = result.fused_plate.char_confidences[i] if i < len(result.fused_plate.char_confidences) else 0.0
            conf_cells.append(Paragraph(f"{conf:.2f}", styles["Tiny"]))
            support_cells.append(Paragraph(f"{result.fused_plate.support[i] if i < len(result.fused_plate.support) else 0}", styles["Tiny"]))
        width = doc.width / max(1, len(plate_cells))
        plate_table = Table([plate_cells, conf_cells, support_cells], colWidths=[width] * len(plate_cells))
        plate_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9fb7bf")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f7f8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(plate_table)
        story.append(p("Rows: character / calibrated confidence / supporting frames. '?' means unknown, never invented.", "Tiny"))
    else:
        story.append(p("No plate characters were recovered.", "SmallBody"))
    if result.fused_plate.class_corrections:
        story.append(Spacer(1, 3))
        story.append(p("Grammar-based normalisations (observed character reinterpreted, never invented):", "SmallBody"))
        for correction in result.fused_plate.class_corrections[:8]:
            story.append(p(f"Position {int(correction['position']) + 1}: {correction['rule']}", "Tiny"))
    story.append(Spacer(1, 8))

    section("Frame evidence")
    frames = _select_frames(result.frames, min(config.report.max_frames, 8))
    if not frames:
        story.append(p("No frames were sampled.", "SmallBody"))
    for ev in frames:
        rows = []
        image_path = None
        if ev.annotated_path:
            image_path = run_dir / ev.annotated_path
        elif ev.frame_path:
            image_path = run_dir / ev.frame_path
        if image_path and image_path.exists():
            try:
                reader = ImageReader(str(image_path))
                iw, ih = reader.getSize()
                width = 52 * mm
                height = width * ih / max(1, iw)
                rows.append([Image(str(image_path), width=width, height=height)])
            except Exception:
                rows.append([p("image unavailable", "Tiny")])
        else:
            rows.append([p("image unavailable", "Tiny")])
        text = (
            f"<b>Frame {ev.frame_index}</b> · t={ev.timestamp:.2f}s · quality={ev.quality:.2f} · "
            f"{'USED' if ev.used else 'not used'}<br/>"
            f"OCR: <font name='Courier'>{html.escape(ev.ocr_text or '—')}</font> "
            f"(mean {ev.ocr_confidence:.2f})<br/>"
            f"Aligned: <font name='Courier'>{html.escape(ev.aligned_text or '—')}</font><br/>"
            f"{html.escape(ev.reason)}"
        )
        rows.append([Paragraph(text, styles["SmallBody"])])
        table = Table([rows], colWidths=[58 * mm, doc.width - 58 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d5da")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 4))

    if result.decision == "SHORTLIST" and result.matches:
        story.append(PageBreak())
        section("Top-5 shortlist")
        for m in result.matches:
            block = []
            block.append(
                Paragraph(
                    f"<b>#{m.rank} · {html.escape(m.vehicle.plate)}</b> — "
                    f"{html.escape(m.vehicle.make)} {html.escape(m.vehicle.model)} · "
                    f"{html.escape(m.vehicle.color)} · {html.escape(m.vehicle.body_type)} · "
                    f"confidence <b>{m.confidence:.2f}</b>",
                    styles["Heading3"],
                )
            )
            block.append(
                p(
                    f"Score {m.score:.2f} | plate {m.plate_score:.2f} | colour "
                    f"{'—' if m.color_score is None else f'{m.color_score:.2f}'} | body "
                    f"{'—' if m.body_score is None else f'{m.body_score:.2f}'} | style "
                    f"{'—' if m.plate_style_score is None else f'{m.plate_style_score:.2f}'} | "
                    f"known {m.known_chars}/{m.known_chars + m.unknown_chars}",
                    "SmallBody",
                )
            )
            for reason in m.reasons[:8]:
                block.append(p(f"• {reason}", "Reason"))
            if m.conflicts:
                block.append(p("Conflicts:", "SmallBody"))
                for conflict in m.conflicts[:6]:
                    block.append(p(f"• {conflict}", "Reason"))
            block.append(Spacer(1, 6))
            story.append(KeepTogether(block))
    else:
        section("Why there is no confident match")
        story.append(p(result.decision_reason, "SmallBody"))
        if result.near_misses:
            story.append(p("Near misses are shown for transparency only; they are not a confident identification.", "SmallBody"))
            for m in result.near_misses[:5]:
                story.append(
                    p(
                        f"#{m.rank} {m.vehicle.plate} — confidence {m.confidence:.2f} "
                        f"(plate {m.plate_score:.2f}). {m.rejection_reason or ''}",
                        "Reason",
                    )
                )

    section("Warnings and limitations")
    for warning in result.warnings:
        story.append(p(f"• {warning}", "Tiny"))
    story.append(Spacer(1, 4))
    story.append(
        p(
            "Method: pretrained YOLO vehicle detection (+ ByteTrack when available), plate localisation, "
            "PaddleOCR/template OCR, team-authored confidence-weighted multi-frame character fusion, "
            "and team-authored partial-plate + attribute ranking. Unknown characters are '?' and are never "
            "filled from the registry. No super-resolution is used. Data splits for evaluation are by clip/vehicle.",
            "Tiny",
        )
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(16 * mm, 8 * mm, "TRACE-X · decision support, not proof")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path


def build_reports(result: AnalysisResult, run_dir: Path, config: PipelineConfig) -> tuple[Path, Path | None]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    html_path = build_html_report(result, run_dir, config)
    pdf_path: Path | None = None
    if (config.report.pdf_engine or "reportlab").lower() != "print-html-only":
        try:
            pdf_path = build_pdf_report(result, run_dir, config)
        except Exception:
            pdf_path = None
    return html_path, pdf_path
