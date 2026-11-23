#!/usr/bin/env python3
"""TRACE-X SOTA License Plate Detector Training Pipeline (YOLO11s).

Trains Ultralytics YOLO11 (State-of-the-Art, released late 2024) across ALL
Indian road datasets:
1. data/plate_det (2,550 real-world Indian vehicle crops - cars, bikes, buses)
2. data/raw/truck_plates (1,060 commercial heavy vehicles - trucks, lorries, trailers)

Features:
- State-of-the-Art YOLO11 architecture with C3k2 & C2PSA Pointwise Spatial Attention
- Superior small-object mAP and low parameter footprint (9.4M vs 11.2M in YOLOv8)
- Real-world augmentation: motion blur, mosaic, perspective tilt, HSV jitter
- Automatic hardware detection (CUDA FP16 Tensor Cores or CPU fallback)
- Auto-exports champion weights to models/plate_yolo11s.pt

Usage:
    # 1. Train on GPU with all datasets and periodic checkpoints (saves every 5 epochs):
    python scripts/train_plate_yolo11.py --model yolo11s.pt --epochs 50 --save-period 5

    # 2. Fast trial on GPU with per-epoch checkpoints:
    python scripts/train_plate_yolo11.py --model yolo11n.pt --epochs 10 --save-period 1

    # 3. Resume training from last checkpoint:
    python scripts/train_plate_yolo11.py --resume

    # 4. Resume or fine-tune from a specific saved checkpoint:
    python scripts/train_plate_yolo11.py --checkpoint models/checkpoints/plate_yolo11n_epoch5_best.pt --epochs 30
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
import yaml

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def prepare_combined_yaml(root: Path) -> Path:
    """Create unified YOLO dataset configuration combining all plate data sources."""
    cfg_dir = root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = cfg_dir / "combined_plates.yaml"

    truck_images = root / "data" / "raw" / "truck_plates" / "images"
    train_sources = ["data/plate_det/images/train"]
    if truck_images.exists() and len(list(truck_images.glob("*"))) > 0:
        train_sources.append("data/raw/truck_plates/images")

    config = {
        "path": str(root.resolve()),
        "train": train_sources,
        "val": "data/plate_det/images/val",
        "test": "data/plate_det/images/test",
        "names": {
            0: "license_plate"
        }
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"[*] Generated unified dataset config: {yaml_path}")
    print(f"    Train sources: {train_sources}")
    return yaml_path


def train_yolo11_detector(
    model_name: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    save_period: int = 5,
    resume: bool = False,
    checkpoint: str | None = None,
    device: str | None = None
) -> Path | None:
    """Train YOLO11 license plate detector on all merged Indian road datasets."""
    import torch
    from ultralytics import YOLO

    cuda_avail = torch.cuda.is_available()
    dev = device or ("0" if cuda_avail else "cpu")

    project_dir = ROOT_DIR / "runs" / "detect"
    run_name = "plate_yolo11"
    weights_dir = project_dir / run_name / "weights"
    checkpoints_dir = ROOT_DIR / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("      TRACE-X SOTA LICENSE PLATE TRAINING PIPELINE (YOLO11)")
    print("=" * 70)
    print(f"  Base Architecture: {model_name} (SOTA Ultralytics YOLO11)")
    print(f"  Target Resolution: {imgsz}x{imgsz}")
    print(f"  Target Epochs:     {epochs}")
    print(f"  Batch Size:        {batch_size}")
    print(f"  Save Period:       Every {save_period} epoch(s)")
    print(f"  Resume Mode:       {resume} (checkpoint: {checkpoint or 'auto-last'})")
    print(f"  Compute Device:    {dev} (CUDA available: {cuda_avail})")
    print("=" * 70)

    # Prepare combined dataset
    data_yaml = prepare_combined_yaml(ROOT_DIR)

    # Determine model / checkpoint to load
    if resume or checkpoint:
        resume_pt = Path(checkpoint) if checkpoint else (weights_dir / "last.pt")
        if not resume_pt.exists():
            raise FileNotFoundError(f"Checkpoint for resume not found at: {resume_pt}")
        print(f"\n[*] Resuming training from checkpoint: {resume_pt}")
        model = YOLO(str(resume_pt))
        train_kwargs = {"resume": True}
    else:
        print(f"\n[*] Initializing {model_name}...")
        model = YOLO(model_name)
        train_kwargs = {}

    print(f"\n[*] Launching training run on {dev}...")
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=dev,
            workers=min(4, os.cpu_count() or 2),
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            patience=15,
            save=True,
            save_period=save_period,  # Save checkpoint every N epochs (e.g. epoch1.pt, epoch5.pt)
            val=True,
            mosaic=1.0,        # Excellent for contextual vehicle-to-plate scales
            degrees=10.0,      # Invariant to tilted CCTV camera angles
            shear=5.0,         # Invariant to oblique road perspective
            scale=0.5,         # Multiscale scale jitter (near and far vehicles)
            plots=True,
            verbose=True,
            **train_kwargs
        )

        # Sync intermediate and champion checkpoints to models/checkpoints/
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"

        if best_pt.exists():
            model_stem = Path(model_name).stem
            target_out = ROOT_DIR / "models" / f"plate_{model_stem}.pt"
            shutil.copy2(str(best_pt), str(target_out))
            # Also maintain active plate detector path
            active_out = ROOT_DIR / "models" / "plate_yolov8s.pt"
            shutil.copy2(str(best_pt), str(active_out))

            # Store copies in models/checkpoints/
            cp_best = checkpoints_dir / f"plate_{model_stem}_best.pt"
            shutil.copy2(str(best_pt), str(cp_best))
            if last_pt.exists():
                cp_last = checkpoints_dir / f"plate_{model_stem}_last.pt"
                shutil.copy2(str(last_pt), str(cp_last))

            # Also copy any epoch checkpoints generated by save_period
            for ep_pt in weights_dir.glob("epoch*.pt"):
                shutil.copy2(str(ep_pt), str(checkpoints_dir / f"plate_{model_stem}_{ep_pt.name}"))

            print("\n" + "=" * 70)
            print("[+] SOTA Training Complete!")
            print(f"    Best Checkpoint:  {best_pt}")
            print(f"    Exported Target:  {target_out}")
            print(f"    Active Pipeline:  {active_out}")
            print(f"    Checkpoints Dir:  {checkpoints_dir}")
            print("=" * 70)
            return target_out
        else:
            print(f"\n[!] Training finished but best.pt not found at expected path: {best_pt}")
            return None

    except Exception as exc:
        print(f"\n[X] Training failed with error: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SOTA YOLO11 License Plate Detector across all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        help="Base YOLO11 model variant ('yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt'). Default: 'yolo11s.pt'"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save checkpoint every X epochs (e.g. 1 saves after every epoch). Default: 5"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint (runs/detect/plate_yolo11/weights/last.pt)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path (.pt) to resume or fine-tune from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training ('0', 'cpu', etc.). Auto-detects GPU if omitted."
    )

    args = parser.parse_args()

    train_yolo11_detector(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        save_period=args.save_period,
        resume=args.resume,
        checkpoint=args.checkpoint,
        device=args.device
    )


if __name__ == "__main__":
    main()
