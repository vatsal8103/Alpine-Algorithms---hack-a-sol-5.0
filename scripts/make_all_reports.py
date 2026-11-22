"""Auto-generate forensic investigation reports for all test clips in data/clips/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.run_pipeline import run_pipeline
from reports.report import generate_html_report


def generate_all_reports(clips_dir: str = "data/clips", output_dir: str = "reports/output") -> list[str]:
    p_clips = Path(clips_dir)
    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    clip_files = sorted(list(p_clips.glob("*.mp4")))
    if not clip_files:
        print(f"No .mp4 clips found in {clips_dir}")
        return []

    generated = []
    print(f"Found {len(clip_files)} test clips. Generating forensic reports...")

    for clip in clip_files:
        print(f"\n---> Analyzing {clip.name}...")
        try:
            case_res = run_pipeline(str(clip))
            report_file = p_out / f"{clip.stem}_report.html"
            generate_html_report(case_res, str(report_file))
            generated.append(str(report_file))
            print(f"  [OK] Saved report: {report_file.name}")
        except Exception as e:
            print(f"  [ERROR] Failed on {clip.name}: {e}")

    print(f"\nFinished generating {len(generated)} forensic reports in {output_dir}")
    return generated


if __name__ == "__main__":
    generate_all_reports()
