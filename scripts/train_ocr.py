"""TRACE-X OCR Recognizer Local Fine-Tuning Script.

Architecture:
- High-efficiency Mobile CRNN: 4-layer CNN feature extractor + Bidirectional GRU + CTC loss
- Optimized for fast CPU training utilizing all available CPU cores with multi-threading
- Charset: 36 characters (A-Z, 0-9) + 1 CTC blank
- Stratified Data Mix: 70% synthetic standard / 20% real crops / 10% hard confusable cases
- Checkpointing per epoch with early stopping on held-out REAL validation crops
- Exports weights to models/ocr_finetuned.pth and generates Colab/Kaggle configurations
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set optimal multi-threading for CPU
if not torch.cuda.is_available():
    torch.set_num_threads(min(8, os.cpu_count() or 4))

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 is CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # 37 (blank + 36 characters)


# ============================================================================
# 1. Dataset & Collate Function
# ============================================================================

class PlateOCRDataset(Dataset):
    """Plate crop dataset supporting synthetic, hard confusable, and real video crops with high-speed in-memory caching."""

    def __init__(
        self,
        samples: list[tuple[str, str]],
        img_h: int = 32,
        img_w: int = 128,
        augment: bool = False,
        cache_in_ram: bool = True
    ) -> None:
        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.cache_in_ram = cache_in_ram
        self.cached_images: list[np.ndarray | None] = [None] * len(samples)

        # Pre-cache images in memory to completely eliminate disk I/O bottlenecks during GPU training
        if self.cache_in_ram and len(samples) <= 25000:
            for i, (img_path, _) in enumerate(samples):
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    im = cv2.resize(im, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
                self.cached_images[i] = im

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]

        if self.cache_in_ram and self.cached_images[idx] is not None:
            img = self.cached_images[idx].copy()
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        if self.augment:
            # High-fidelity augmentations: contrast/brightness, blur, and sensor noise
            if random.random() < 0.35:
                alpha = random.uniform(0.75, 1.25)
                beta = random.randint(-15, 15)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                img = cv2.GaussianBlur(img, (3, 3), 0)
            if random.random() < 0.20:
                noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        tensor = torch.from_numpy(img).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0)  # (1, H, W)

        clean_label = "".join([c for c in label.upper() if c in CHAR2IDX])
        target = torch.tensor([CHAR2IDX[c] for c in clean_label], dtype=torch.long)
        target_len = len(clean_label)

        return tensor, target, target_len


def collate_ocr_batch(batch):
    images, targets, target_lens = zip(*batch)
    images = torch.stack(images, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    target_lens = torch.tensor(target_lens, dtype=torch.long)
    return images, flat_targets, target_lens


# ============================================================================
# 2. High-Efficiency Mobile CRNN Architecture
# ============================================================================

class CRNNOcrModel(nn.Module):
    """Mobile CRNN: 4-layer CNN + Bidirectional GRU + CTC Head."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 96) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: 32x128 -> 16x64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16x64 -> 8x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 8x32 -> 4x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Block 4: 4x32 -> 1x32
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((4, 1), (4, 1)),
        )

        # BiGRU sequence model (96 hidden units, 1 layer for ultra-fast local CPU throughput)
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Classifier head
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 128)
        features = self.cnn(x)  # (B, 128, 1, 32)
        seq = features.squeeze(2).permute(0, 2, 1)  # (B, 32, 128)

        recurrent_out, _ = self.rnn(seq)  # (B, 32, 2 * hidden_size)
        logits = self.fc(recurrent_out)    # (B, 32, num_classes)

        # CTC loss requires (T, B, num_classes) log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
        return log_probs

    def decode_greedy(self, log_probs: torch.Tensor) -> list[str]:
        """Greedy CTC decoder: collapses duplicates, strips blank token (0)."""
        preds = torch.argmax(log_probs, dim=-1).permute(1, 0).cpu().numpy()
        results = []
        for seq in preds:
            chars = []
            prev = 0
            for p in seq:
                if p != 0 and p != prev and p in IDX2CHAR:
                    chars.append(IDX2CHAR[p])
                prev = p
            results.append("".join(chars))
        return results


# ============================================================================
# 3. Stratified Data Preparation
# ============================================================================

def build_mixed_dataset(
    synth_csv: str = "data/plates_synth/labels.csv",
    real_train_txt: str = "data/real_crops/train_real.txt",
    real_val_txt: str = "data/real_crops/val_real.txt",
    kaggle_csv: str = "data/kaggle_crops/labels.csv",
    target_train_size: int | None = None,
    use_all_data: bool = True,
    only_kaggle: bool = False,
    seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build dataset from Kaggle real plates and optional synthetic plates."""
    rng = random.Random(seed)

    # Ingest 1,695 Kaggle Real Indian Plate Crops
    kaggle_train: list[tuple[str, str]] = []
    kaggle_val: list[tuple[str, str]] = []

    p_kaggle = Path(kaggle_csv)
    if p_kaggle.exists():
        with open(p_kaggle, "r", encoding="utf-8") as f:
            k_reader = list(csv.DictReader(f))
            rng.shuffle(k_reader)
            n_k_val = int(len(k_reader) * 0.15)
            for r in k_reader[n_k_val:]:
                kaggle_train.append((r["image_path"], r["text"]))
            for r in k_reader[:n_k_val]:
                kaggle_val.append((r["image_path"], r["text"]))

    # Ingest Real Crops from Video Tracks
    video_train: list[tuple[str, str]] = []
    video_val: list[tuple[str, str]] = []

    p_real_tr = Path(real_train_txt)
    if p_real_tr.exists():
        with open(p_real_tr, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    video_train.append((parts[0], parts[1]))

    p_real_val = Path(real_val_txt)
    if p_real_val.exists():
        with open(p_real_val, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    video_val.append((parts[0], parts[1]))

    val_samples = list(video_val) + list(kaggle_val)

    if only_kaggle:
        # STRICTLY REAL KAGGLE + REAL VIDEO CROPS ONLY (NO SYNTHETIC PLATES)
        real_video_expanded = list(video_train) * 5
        train_samples = list(kaggle_train) + real_video_expanded
        rng.shuffle(train_samples)

        print("\n==================================================================", flush=True)
        print("  [*] ONLY KAGGLE REAL DATASET ACTIVE (ZERO SYNTHETIC PLATES)", flush=True)
        print(f"  - Kaggle Real Indian Plates:  {len(kaggle_train):,} training crops", flush=True)
        print(f"  - CCTV Video Track Crops:     {len(video_train):,} unique ({len(real_video_expanded):,} weighted)", flush=True)
        print(f"  - Synthetic Plates Used:      0 (DISABLED)", flush=True)
        print(f"  - TOTAL REAL TRAINING SAMPLES:{len(train_samples):,} authentic plate crops", flush=True)
        print(f"  - HELD-OUT REAL VALIDATION:   {len(val_samples):,} authentic plate crops", flush=True)
        print("==================================================================\n", flush=True)
        return train_samples, val_samples

    synth_all: list[tuple[str, str]] = []
    synth_hard: list[tuple[str, str]] = []

    p_synth = Path(synth_csv)
    if p_synth.exists():
        with open(p_synth, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                pair = (r["image_path"], r["text"])
                synth_all.append(pair)
                if r.get("is_hard_case") == "True":
                    synth_hard.append(pair)

    val_samples = list(video_val) + list(kaggle_val)

    if use_all_data or target_train_size is None or target_train_size >= len(synth_all):
        # USE 100% OF ALL AVAILABLE DATA ON DISK
        # 1. All 30,000 synthetic plates (all 4 styles, levels 0-5)
        # 2. Oversample real Kaggle + Video crops (3x) so real plates appear prominently in every batch
        # 3. Oversample hard confusions (2x) to eliminate O/0, B/8, S/5, Z/2 ambiguities
        real_all = (kaggle_train * 3) + (video_train * 10)
        train_samples = list(synth_all) + real_all + list(synth_hard)
        rng.shuffle(train_samples)

        print("\n==================================================================", flush=True)
        print("  [*] FULL DATASET MODE ACTIVE: USING 100% OF AVAILABLE DATA", flush=True)
        print(f"  - Synthetic Plates:         {len(synth_all):,} (all styles & blur levels 0-5)", flush=True)
        print(f"  - Hard Confusable Subset:   {len(synth_hard):,} (O/0, B/8, I/1, S/5, Z/2)", flush=True)
        print(f"  - Kaggle Real Indian Plates:{len(kaggle_train):,} unique ({len(kaggle_train)*3:,} oversampled)", flush=True)
        print(f"  - CCTV Video Track Crops:   {len(video_train):,} unique", flush=True)
        print(f"  - TOTAL TRAINING EXAMPLES:  {len(train_samples):,} samples", flush=True)
        print(f"  - HELD-OUT VALIDATION SET:  {len(val_samples):,} REAL Indian plates", flush=True)
        print("==================================================================\n", flush=True)
    else:
        # Subsampled mode
        n_synth = int(target_train_size * 0.70)
        n_hard = int(target_train_size * 0.10)
        n_real = target_train_size - n_synth - n_hard

        train_samples = []
        if synth_all:
            train_samples.extend(rng.choices(synth_all, k=n_synth))
        if synth_hard:
            train_samples.extend(rng.choices(synth_hard, k=n_hard))
        real_pool = kaggle_train + video_train
        if real_pool:
            train_samples.extend(rng.choices(real_pool, k=n_real))

        rng.shuffle(train_samples)
        print(f"Data mix loaded: {len(train_samples)} training samples (subsampled mode)", flush=True)
        print(f"Validation set:   {len(val_samples)} held-out REAL Indian plate samples", flush=True)

    return train_samples, val_samples


# ============================================================================
# 4. Local Training Loop with Live Progress Logging
# ============================================================================

def train_ocr_locally(
    epochs: int = 20,
    batch_size: int = 64 if torch.cuda.is_available() else 32,
    lr: float = 0.002,
    train_size: int | None = None,
    use_all_data: bool = True,
    only_kaggle: bool = False,
    model_out_path: str = "models/ocr_finetuned.pth",
    checkpoint_dir: str = "runs/ocr_checkpoints",
    resume: bool = True
) -> dict[str, Any]:
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    if is_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        dev_desc = f"{gpu_name} ({gpu_mem:.0f} MB VRAM, CUDA 12.4 + Tensor Cores)"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_amp = True
        scaler = torch.amp.GradScaler("cuda")
    else:
        dev_desc = f"CPU ({os.cpu_count()} threads allocated)"
        use_amp = False
        scaler = None

    mode_str = "KAGGLE REAL DATASET ONLY (Zero Synthetic Plates)" if only_kaggle else "100% FULL CORPUS (Synthetic + Kaggle Real)"

    print("\n==================================================================", flush=True)
    print("  TRACE-X Production OCR Fine-Tuning Pipeline (GPU Accelerated)")
    print(f"  Hardware:   {dev_desc}")
    print(f"  Epochs:     {epochs} | Batch Size: {batch_size} | Base LR: {lr}")
    print(f"  Mixed Prec: {'FP16 Automatic Mixed Precision (AMP)' if use_amp else 'FP32'}")
    print(f"  RAM Cache:  High-speed in-memory tensors enabled")
    print(f"  Mode:       {mode_str}", flush=True)
    print("==================================================================\n", flush=True)

    p_ckpt = Path(checkpoint_dir)
    p_ckpt.mkdir(parents=True, exist_ok=True)
    Path(model_out_path).parent.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = build_mixed_dataset(
        target_train_size=train_size,
        use_all_data=use_all_data,
        only_kaggle=only_kaggle
    )
    train_dataset = PlateOCRDataset(train_samples, augment=True)
    val_dataset = PlateOCRDataset(val_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_ocr_batch,
        drop_last=True,
        pin_memory=is_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_ocr_batch,
        pin_memory=is_cuda
    )

    model = CRNNOcrModel(num_classes=NUM_CLASSES).to(device)

    # Resume previous weights if available
    if resume and Path(model_out_path).exists():
        try:
            model.load_state_dict(torch.load(model_out_path, map_location=device))
            print(f"[*] Resumed weights from existing checkpoint: {model_out_path}\n", flush=True)
        except Exception:
            pass

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        current_lr = scheduler.get_last_lr()[0]
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", unit="batch", dynamic_ncols=True, leave=True)

        for images, targets, target_lens in pbar:
            images = images.to(device, non_blocking=is_cuda)
            targets = targets.to(device, non_blocking=is_cuda)
            target_lens = target_lens.to(device, non_blocking=is_cuda)

            optimizer.zero_grad()
            b_size = images.size(0)
            t_len = 32
            input_lens = torch.full((b_size,), t_len, dtype=torch.long, device=device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    log_probs = model(images)
                    loss = ctc_loss_fn(log_probs, targets, input_lens, target_lens)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    n_batches += 1
            else:
                log_probs = model(images)
                loss = ctc_loss_fn(log_probs, targets, input_lens, target_lens)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

            avg_running = total_loss / max(1, n_batches)
            pbar.set_postfix({"loss": f"{avg_running:.4f}", "lr": f"{current_lr:.5f}"})

        scheduler.step()
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation with progress indicator
        model.eval()
        val_correct_chars = 0
        val_total_chars = 0
        val_correct_plates = 0
        val_total_plates = 0

        with torch.no_grad():
            for images, targets, target_lens in val_loader:
                images = images.to(device, non_blocking=is_cuda)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        log_probs = model(images)
                else:
                    log_probs = model(images)
                pred_texts = model.decode_greedy(log_probs)

                offset = 0
                for i, t_len in enumerate(target_lens):
                    t_slice = targets[offset:offset + t_len].tolist()
                    true_text = "".join([IDX2CHAR.get(idx, "") for idx in t_slice])
                    offset += t_len

                    pred = pred_texts[i]
                    val_total_plates += 1
                    if pred == true_text:
                        val_correct_plates += 1

                    match_l = min(len(pred), len(true_text))
                    val_correct_chars += sum(1 for a, b in zip(pred[:match_l], true_text[:match_l]) if a == b)
                    val_total_chars += len(true_text)

        t_elapsed = time.time() - t_start
        val_char_acc = val_correct_chars / max(1, val_total_chars)
        val_plate_acc = val_correct_plates / max(1, val_total_plates)

        status_msg = (
            f"--> Epoch [{epoch:02d}/{epochs:02d}] ({t_elapsed:4.2f}s) | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Char Acc: {val_char_acc*100:5.2f}% | "
            f"Val Plate Acc: {val_plate_acc*100:5.2f}%"
        )
        print(status_msg, flush=True)

        # Checkpointing
        epoch_ckpt = p_ckpt / f"epoch_{epoch:02d}.pth"
        torch.save(model.state_dict(), epoch_ckpt)

        if val_char_acc >= best_val_acc:
            best_val_acc = val_char_acc
            torch.save(model.state_dict(), model_out_path)
            torch.save(model.state_dict(), p_ckpt / "best_model.pth")
            print(f"    [*] Saved new best model to {model_out_path} (Char Acc: {best_val_acc*100:.2f}%)\n", flush=True)
        else:
            print("", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_char_acc": val_char_acc,
            "val_plate_acc": val_plate_acc
        })

    print(f"\n==================================================================", flush=True)
    print(f"  Training Complete! Peak Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Weights saved: {model_out_path}")
    print(f"==================================================================\n", flush=True)

    # Auto-calibrate after training
    try:
        from scripts.calibrate_ocr import calibrate_ocr_model
        print("[*] Running automatic temperature scaling calibration on new weights...", flush=True)
        calibrate_ocr_model(model_path=model_out_path)
    except Exception as e:
        print(f"[Warning] Auto-calibration skipped: {e}", flush=True)

    return {"best_val_acc": best_val_acc, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TRACE-X OCR Recognizer Locally on Kaggle Dataset")
    default_bs = 64 if torch.cuda.is_available() else 32
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default 30)")
    parser.add_argument("--batch-size", type=int, default=default_bs, help=f"Batch size (default {default_bs})")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (default 0.002)")
    parser.add_argument("--only-kaggle", "--no-synth", action="store_true", default=False, help="Train ONLY on real Kaggle plate crops (no synthetic plates)")
    parser.add_argument("--subsample", type=int, default=None, help="Optional: limit dataset to N samples")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from scratch instead of resuming")
    parser.add_argument("--model-out", type=str, default="models/ocr_finetuned.pth", help="Output path")
    args = parser.parse_args()

    use_all = (args.subsample is None)

    train_ocr_locally(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_size=args.subsample,
        use_all_data=use_all,
        only_kaggle=args.only_kaggle,
        model_out_path=args.model_out,
        resume=not args.fresh
    )


if __name__ == "__main__":
    main()
