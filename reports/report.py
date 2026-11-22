"""TRACE-X Forensic Report Generator.

Renders print-ready A4 forensic investigation reports in HTML and PDF formats.
Contains:
- Case ID, clip name, analysis timestamp, track ID
- Recovered consensus plate with '?' and per-position confidences
- Vehicle attributes with confidence
- Top-5 ranked candidates with score breakdown bars and rationales
- Final decision (MATCH_FOUND or NO_CONFIDENT_MATCH) and reason
- Weights and thresholds applied
- Mandatory decision-support disclaimer
"""

from __future__ import annotations

import os
from pathlib import Path
from jinja2 import Template
from pipeline.schemas import CaseResult

REPORT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>TRACE-X Forensic Investigation Report - {{ case.case_id }}</title>
    <style>
        @page { size: A4; margin: 18mm; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: #ffffff;
            color: #1e293b;
            line-height: 1.4;
            font-size: 11pt;
            margin: 0;
            padding: 0;
        }
        .header {
            border-bottom: 2px solid #2563eb;
            padding-bottom: 12px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            color: #0f172a;
            font-size: 20pt;
        }
        .meta-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .meta-table td {
            padding: 4px 8px;
            font-size: 9.5pt;
        }
        .plate-box {
            background: #f8fafc;
            border: 2px solid #cbd5e1;
            padding: 12px;
            border-radius: 8px;
            margin: 14px 0;
            display: inline-block;
        }
        .char-tile {
            display: inline-block;
            font-family: "Courier New", Courier, monospace;
            font-size: 20pt;
            font-weight: bold;
            padding: 4px 10px;
            background: #ffffff;
            border: 1px solid #94a3b8;
            border-radius: 4px;
            margin-right: 4px;
        }
        .char-unknown {
            color: #d97706;
            border: 2px dashed #d97706;
            background: #fef3c7;
        }
        .section-title {
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            margin-top: 20px;
            font-size: 13pt;
        }
        .decision-box {
            padding: 12px 16px;
            border-radius: 6px;
            margin: 14px 0;
        }
        .decision-match {
            background: #ecfdf5;
            border-left: 5px solid #10b981;
            color: #065f46;
        }
        .decision-nomatch {
            background: #fef2f2;
            border-left: 5px solid #ef4444;
            color: #991b1b;
        }
        .candidate-card {
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background: #f8fafc;
        }
        .score-bar {
            background: #e2e8f0;
            border-radius: 3px;
            height: 10px;
            width: 140px;
            display: inline-block;
            overflow: hidden;
            vertical-align: middle;
        }
        .score-bar-fill {
            background: #2563eb;
            height: 100%;
        }
        .disclaimer {
            margin-top: 36px;
            border-top: 1px solid #cbd5e1;
            padding-top: 10px;
            font-size: 8.5pt;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>TRACE-X Forensic Vehicle Investigation Report</h1>
        <table class="meta-table">
            <tr>
                <td><strong>Case ID:</strong> {{ case.case_id }}</td>
                <td><strong>Clip Name:</strong> {{ case.clip_name }}</td>
                <td><strong>Analysis Date:</strong> {{ case.created_at }}</td>
            </tr>
            <tr>
                <td><strong>Pipeline Version:</strong> 1.0.0</td>
                <td><strong>Frames Analyzed:</strong> {{ case.fused_plate.frames_used }}</td>
                <td><strong>Indian Plate Syntax:</strong> {{ "Valid" if case.fused_plate.format_valid else "Ambiguous" }}</td>
            </tr>
        </table>
    </div>

    <div class="section-title">1. Consensus Plate Recovery Evidence</div>
    <div class="plate-box">
        {% for ch in case.fused_plate.text %}
            {% if ch == '?' %}
                <span class="char-tile char-unknown">?</span>
            {% else %}
                <span class="char-tile">{{ ch }}</span>
            {% endif %}
        {% endfor %}
    </div>
    <p style="font-size: 9.5pt; color: #475569;">
        Mean Known Confidence: <strong>{{ "%.1f"|format(case.fused_plate.mean_confidence * 100) }}%</strong> |
        Resolved Slots: <strong>{{ case.fused_plate.known_count }}/10</strong> |
        Unresolved ('?'): <strong>{{ case.fused_plate.unknown_count }}</strong>
    </p>

    <div class="section-title">2. Extracted Vehicle Visual Attributes</div>
    <p>
        Observed Body Colour: <strong>{{ case.attributes.colour|upper }}</strong> (Confidence: {{ "%.2f"|format(case.attributes.colour_conf) }})<br/>
        Classified Body Type: <strong>{{ case.attributes.body_type|upper }}</strong> (Confidence: {{ "%.2f"|format(case.attributes.body_conf) }})<br/>
        Estimated Make / Model: <strong>{{ case.attributes.make }} {{ case.attributes.model }}</strong>
    </p>

    <div class="section-title">3. Forensic Investigation Decision</div>
    <div class="decision-box decision-{{ 'match' if case.decision == 'MATCH_FOUND' else 'nomatch' }}">
        <strong style="font-size: 12pt;">DECISION: {{ case.decision }}</strong><br/>
        <span>{{ case.decision_reason }}</span>
    </div>

    <div class="section-title">4. Candidate Shortlist (Top 5 Ranked){{ " — No Confident Match" if case.decision != "MATCH_FOUND" else "" }}</div>
    {% if case.decision != "MATCH_FOUND" %}
        <p style="font-size: 9pt; color: #991b1b; background: #fef2f2; padding: 8px 12px; border-radius: 5px; border: 1px solid #fca5a5; margin-bottom: 12px;">
            <strong>Forensic Safe Abstention Notice:</strong> The license plate in this footage was unreadable across sampled frames (0% plate match credit). The entries below are ranked strictly by observed visual attributes (colour & body type) for investigative lead elimination only; none meet the required confidence threshold ({{ "%.2f"|format(case.thresholds_used.get('no_match_threshold', 0.52)) }}) for vehicle identification.
        </p>
    {% endif %}
    {% if case.candidates %}
        {% for cand in case.candidates %}
            <div class="candidate-card">
                <table style="width: 100%;">
                    <tr>
                        <td><strong style="font-size: 12pt;">#{{ loop.index }} — {{ cand.registration_number }}</strong> ({{ cand.vehicle_id }})</td>
                        <td style="text-align: right;">
                            Composite Score: <strong>{{ "%.3f"|format(cand.final_score) }}</strong>
                            <div class="score-bar">
                                <div class="score-bar-fill" style="width: {{ cand.final_score * 100 }}%;"></div>
                            </div>
                        </td>
                    </tr>
                </table>
                <div style="font-size: 9pt; margin-top: 6px;">
                    Plate: {{ "%.2f"|format(cand.breakdown.plate) }} | Colour: {{ "%.2f"|format(cand.breakdown.colour) }} | Body: {{ "%.2f"|format(cand.breakdown.body) }}
                </div>
                <div style="font-size: 9pt; margin-top: 4px;">
                    <strong>Supporting Evidence:</strong> {{ cand.why_match|join("; ") }}
                </div>
                {% if cand.why_not %}
                    <div style="font-size: 9pt; color: #b91c1c; margin-top: 2px;">
                        <strong>Points of Discrepancy:</strong> {{ cand.why_not|join("; ") }}
                    </div>
                {% endif %}
            </div>
        {% endfor %}
    {% else %}
        <p>No candidate met the evidential confidence threshold.</p>
    {% endif %}

    <div class="section-title">5. Audit Configuration & Thresholds Applied</div>
    <table class="meta-table" style="font-size: 8.5pt;">
        <tr>
            <td><strong>Weights:</strong> Plate: {{ case.weights_used.get('plate', 0.65) }}, Colour: {{ case.weights_used.get('colour', 0.15) }}, Body: {{ case.weights_used.get('body', 0.12) }}</td>
            <td><strong>No-Match Threshold:</strong> {{ case.thresholds_used.get('no_match_threshold', 0.52) }}</td>
            <td><strong>Margin Threshold:</strong> {{ case.thresholds_used.get('margin_threshold', 0.06) }}</td>
        </tr>
    </table>

    <div class="disclaimer">
        <strong>LEGAL & FORENSIC DISCLAIMER:</strong> Highest-ranked candidate based on the evidence.
        Decision-support only. Results are never a definitive identification.
        Never use this output as sole proof for enforcement action.
    </div>
</body>
</html>
"""


def generate_html_report(case: CaseResult, output_path: str) -> str:
    """Generate standalone HTML forensic report from CaseResult."""
    p_out = Path(output_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    tmpl = Template(REPORT_HTML_TEMPLATE)
    rendered = tmpl.render(case=case)

    with open(p_out, "w", encoding="utf-8") as f:
        f.write(rendered)

    return str(p_out)


def generate_pdf_report(case: CaseResult, output_pdf_path: str) -> str | None:
    """Attempt PDF generation via WeasyPrint or ReportLab; falls back gracefully."""
    p_pdf = Path(output_pdf_path)
    p_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_temp = p_pdf.with_suffix(".html")
    generate_html_report(case, str(html_temp))

    try:
        from weasyprint import HTML
        HTML(filename=str(html_temp)).write_pdf(str(p_pdf))
        return str(p_pdf)
    except Exception:
        # Fallback to HTML
        return str(html_temp)
