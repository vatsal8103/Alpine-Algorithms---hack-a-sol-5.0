"""TRACE-X Batch Test Runner for 'test/' directory.

Processes all videos and images placed in the 'test/' folder using GPU-accelerated
vehicle tracking, continuous 30-attempt per-vehicle OCR digit scanning,
consensus evidential fusion, and vehicle registry matching.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from pipeline.run_pipeline import run_pipeline


def get_test_files(test_dir: Path) -> list[Path]:
    """Find all supported video and image files in test directory."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
    files = [
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in video_exts
    ]
    return sorted(files, key=lambda p: p.name)


def process_test_folder(
    test_dir: str = "test",
    registry_csv: str = "data/registry/vehicles.csv",
    watch: bool = False
) -> None:
    test_p = Path(test_dir)
    test_p.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  TRACE-X Test Folder Processing Engine")
    print(f"  Target Folder: {test_p.resolve()}")
    print("  Mode:          Continuous 30-attempt per-vehicle digit scanning & fusion")
    print("=" * 70)

    files = get_test_files(test_p)
    if not files and not watch:
        print(f"\n[test] Folder '{test_p.name}/' is currently empty.")
        print(f"[test] Drop any traffic videos (.mp4, .avi, etc.) into '{test_p.resolve()}'")
        print("[test] and rerun this command, or use --watch to auto-process dropped files.")
        return

    processed: set[str] = set()

    while True:
        current_files = get_test_files(test_p)
        for f in current_files:
            if f.name in processed:
                continue
            
            print(f"\n>>> [test] Processing new test video: {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
            t0 = time.time()
            try:
                result = run_pipeline(
                    video_clip_path=str(f),
                    registry_csv=registry_csv,
                    device=None
                )
                elapsed = time.time() - t0
                plate_txt = result.fused_plate.text if result.fused_plate else "???????????"
                print(f">>> [test] Finished {f.name} in {elapsed:.1f}s | Target Plate: {plate_txt} | Vehicles: {len(result.all_vehicles)}")
            except Exception as e:
                print(f"[test] Error processing {f.name}: {e}", file=sys.stderr)
            
            processed.add(f.name)

        if not watch:
            break
            
        time.sleep(2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process test/ directory videos through TRACE-X pipeline")
    parser.add_argument("--test-dir", default="test", help="Path to test directory")
    parser.add_argument("--registry", default="data/registry/vehicles.csv", help="Path to vehicles.csv registry")
    parser.add_argument("--watch", action="store_true", help="Continuously watch folder for new videos")
    args = parser.parse_args()

    process_test_folder(test_dir=args.test_dir, registry_csv=args.registry, watch=args.watch)
