"""TRACE-X Comprehensive OCR Benchmark (Phase 2).

Evaluates candidate OCR engines on the SAME held-out real Indian plate crops:
1. Fine-Tuned Mobile CRNN (models/ocr_finetuned.pth) - Recognition Only
2. EasyOCR Baseline
3. Preprocessing Variants (Deskew, Cubic vs Lanczos Upscale, Mild CLAHE)

Metrics Reported:
- Character Accuracy (1 - CER)
- Full-Plate Accuracy (Exact Match %)
- '?' - Honesty (Accuracy of characters emitted with confident certainty)
- Confidence Calibration Error (Mean confidence vs actual accuracy)
- Inference Latency per crop (ms)
- Performance stratified by sharpness / blur degradation

Outputs:
- results/ocr_benchmark.csv
- results/ocr_benchmark.png
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from pipeline.ocr import LicensePlateOCR, TemplateOCRBackend
from pipeline.plate_stage import deskew_plate
from scripts.train_ocr import CRNNOcrModel, NUM_CLASSES, IDX2CHAR


def measure_sharpness(img_bgr: np.ndarray) -> float:
    """Compute variance of Laplacian as proxy for image sharpness."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ==============================================================================
# Preprocessing Variants (NO AI SUPER-RESOLUTION)
# ==============================================================================

def preprocess_variant_baseline(img: np.ndarray) -> np.ndarray:
    return img


def preprocess_variant_deskew(img: np.ndarray) -> np.ndarray:
    return deskew_plate(img)


def preprocess_variant_cubic(img: np.ndarray, target_h: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / max(1, h)
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)


def preprocess_variant_lanczos(img: np.ndarray, target_h: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / max(1, h)
    return cv2.resize(img, (int(w * scale), target_h), interpolation=cv2.INTER_LANCZOS4)


def preprocess_variant_clahe(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        return clahe.apply(img)


def preprocess_variant_optimal(img: np.ndarray) -> np.ndarray:
    """Combined: Deskew + Lanczos upscale + mild CLAHE."""
    d = deskew_plate(img)
    u = preprocess_variant_lanczos(d, target_h=48)
    c = preprocess_variant_clahe(u)
    return c


# ==============================================================================
# Inference Wrappers
# ==============================================================================

class BenchmarkFastPlateOCR:
    """fast-plate-ocr (CCT-XS Conv-Transformer ONNX) wrapper."""
    def __init__(self, model_name: str = "cct-xs-v2-global-model") -> None:
        from fast_plate_ocr import LicensePlateRecognizer
        self.rec = LicensePlateRecognizer(model_name)

    def predict(self, img_bgr: np.ndarray) -> tuple[str, list[float], list[str]]:
        try:
            res = self.rec.run_one(img_bgr, return_confidence=True)
            text = res.plate.strip()
            confs = [float(p) for p in res.char_probs] if res.has_confidence and res.char_probs is not None else [0.9] * len(text)
            return text, confs, list(text)
        except Exception:
            return "", [], []


class BenchmarkCRNN:
    """Recognition-only PyTorch CRNN wrapper."""
    def __init__(self, weights_path: str = "models/ocr_finetuned.pth", device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CRNNOcrModel(num_classes=NUM_CLASSES).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, img_bgr: np.ndarray, temp: float = 0.5) -> tuple[str, list[float], list[str]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float() / 127.5 - 1.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs = self.model(tensor)
            scaled = log_probs / max(0.1, temp)
            probs = torch.softmax(scaled, dim=-1).squeeze(1).cpu().numpy()

        preds = np.argmax(probs, axis=-1)
        chars = []
        confidences = []
        prev = 0

        for t, p in enumerate(preds):
            if p != 0 and p != prev and p in IDX2CHAR:
                ch = IDX2CHAR[p]
                conf = float(probs[t, p])
                chars.append(ch)
                confidences.append(conf)
            prev = p

        text = "".join(chars)
        return text, confidences, chars


class BenchmarkEasyOCR:
    """EasyOCR recognition wrapper."""
    def __init__(self) -> None:
        import easyocr
        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    def predict(self, img_bgr: np.ndarray) -> tuple[str, list[float], list[str]]:
        results = self.reader.readtext(img_bgr, detail=1)
        if not results:
            return "", [], []
        # Take the best detected box
        best = max(results, key=lambda x: len(re.sub(r"[^A-Za-z0-9]", "", x[1])))
        text = re.sub(r"[^A-Za-z0-9]", "", best[1]).upper()
        conf = float(best[2])
        confs = [conf] * len(text)
        return text, confs, list(text)


class BenchmarkTemplateOCR:
    """Template baseline."""
    def __init__(self) -> None:
        self.engine = TemplateOCRBackend()

    def predict(self, img_bgr: np.ndarray) -> tuple[str, list[float], list[str]]:
        segments = self.engine.segment_characters(img_bgr)
        chars = []
        confs = []
        for sx, sy, sw, sh, patch in segments:
            ch, prob, _ = self.engine.recognize_character(patch)
            chars.append(ch)
            confs.append(prob)
        return "".join(chars), confs, chars


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def evaluate_predictions(
    ground_truth: str,
    pred_text: str,
    pred_confs: list[float],
    conf_threshold: float = 0.50
) -> dict[str, Any]:
    """Calculate character accuracy, full match, honesty, and calibration gap."""
    gt_clean = re.sub(r"[^A-Z0-9]", "", ground_truth.upper())
    pred_clean = re.sub(r"[^A-Z0-9]", "", pred_text.upper())

    # 1. Exact Match
    exact_match = (gt_clean == pred_clean) and len(gt_clean) >= 6

    # 2. Positional Character Accuracy
    m_len = min(len(gt_clean), len(pred_clean))
    char_matches = sum(1 for a, b in zip(gt_clean[:m_len], pred_clean[:m_len]) if a == b)
    char_acc = char_matches / max(1, len(gt_clean))

    # 3. '?' - Honesty (Accuracy of characters emitted as known / confident)
    # If the model emits a char with conf >= conf_threshold, is it correct?
    confident_emitted = 0
    confident_correct = 0
    for i in range(min(len(pred_clean), len(pred_confs))):
        c_prob = pred_confs[i]
        if c_prob >= conf_threshold:
            confident_emitted += 1
            if i < len(gt_clean) and pred_clean[i] == gt_clean[i]:
                confident_correct += 1

    honesty = (confident_correct / confident_emitted) if confident_emitted > 0 else 1.0

    # 4. Calibration: Mean confidence vs actual char accuracy
    mean_conf = float(np.mean(pred_confs)) if pred_confs else 0.0
    calib_gap = abs(mean_conf - char_acc)

    return {
        "exact_match": exact_match,
        "char_acc": char_acc,
        "char_matches": char_matches,
        "gt_len": len(gt_clean),
        "honesty": honesty,
        "confident_emitted": confident_emitted,
        "confident_correct": confident_correct,
        "mean_conf": mean_conf,
        "calib_gap": calib_gap
    }


def run_benchmark(
    test_csv: str = "data/ocr/test/labels.csv",
    video_crops_csv: str = "data/real_crops/labels.csv",
    out_dir: str = "results"
) -> dict[str, Any]:
    p_out = Path(out_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    # Ingest test dataset
    test_samples: list[tuple[str, str, str]] = []  # (img_path, label, clip_source)

    p_test = Path(test_csv)
    if p_test.exists():
        with open(p_test, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                test_samples.append((r["image_path"], r["text"], "heldout_test_set"))

    p_v = Path(video_crops_csv)
    if p_v.exists():
        with open(p_v, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("text"):
                    test_samples.append((r["image_path"], r["text"], r.get("clip_id", "real_video_clip")))

    print(f"\n=================================================================")
    print(f"  TRACE-X Empirical OCR Benchmark on Held-Out Real Crops")
    print(f"  Total Labeled Test Samples: {len(test_samples)}")
    print(f"  Device: {'cuda (GPU)' if torch.cuda.is_available() else 'cpu'}")
    print(f"=================================================================\n")

    # Load engines
    fast_engine = BenchmarkFastPlateOCR("cct-xs-v2-global-model")
    crnn_engine = BenchmarkCRNN()
    easy_engine = BenchmarkEasyOCR()
    tmpl_engine = BenchmarkTemplateOCR()

    configs_to_test = [
        {"name": "fast-plate CCT-XS (Raw)", "engine": fast_engine, "prep": preprocess_variant_baseline},
        {"name": "fast-plate CCT-XS + Deskew", "engine": fast_engine, "prep": preprocess_variant_deskew},
        {"name": "fast-plate CCT-XS + Cubic Upscale", "engine": fast_engine, "prep": preprocess_variant_cubic},
        {"name": "fast-plate CCT-XS + Lanczos Upscale", "engine": fast_engine, "prep": preprocess_variant_lanczos},
        {"name": "fast-plate CCT-XS + Mild CLAHE", "engine": fast_engine, "prep": preprocess_variant_clahe},
        {"name": "CRNN Fine-Tuned (Baseline)", "engine": crnn_engine, "prep": preprocess_variant_baseline},
        {"name": "EasyOCR Baseline", "engine": easy_engine, "prep": preprocess_variant_baseline},
        {"name": "Template OCR Baseline", "engine": tmpl_engine, "prep": preprocess_variant_baseline},
    ]

    benchmark_summary: list[dict[str, Any]] = []

    for cfg in configs_to_test:
        model_name = cfg["name"]
        engine = cfg["engine"]
        prep_fn = cfg["prep"]

        total_exact = 0
        total_char_matches = 0
        total_gt_chars = 0
        total_conf_emitted = 0
        total_conf_correct = 0
        total_calib_gap = 0.0
        latencies_ms = []

        # Sharpness breakdown (sharp > 100 vs blurred <= 100)
        sharp_correct = 0
        sharp_total = 0
        blur_correct = 0
        blur_total = 0

        for img_rel, gt_text, clip_src in test_samples:
            img_p = Path(img_rel)
            if not img_p.exists():
                img_p = PROJECT_ROOT / img_rel
            if not img_p.exists():
                continue

            im = cv2.imread(str(img_p))
            if im is None:
                continue

            sharpness = measure_sharpness(im)
            preprocessed = prep_fn(im)

            t0 = time.perf_counter()
            pred_text, pred_confs, _ = engine.predict(preprocessed)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            metrics = evaluate_predictions(gt_text, pred_text, pred_confs)

            if metrics["exact_match"]:
                total_exact += 1
            total_char_matches += metrics["char_matches"]
            total_gt_chars += metrics["gt_len"]
            total_conf_emitted += metrics["confident_emitted"]
            total_conf_correct += metrics["confident_correct"]
            total_calib_gap += metrics["calib_gap"]

            if sharpness > 100.0:
                sharp_total += 1
                if metrics["exact_match"]:
                    sharp_correct += 1
            else:
                blur_total += 1
                if metrics["exact_match"]:
                    blur_correct += 1

        n_eval = len(latencies_ms)
        char_acc = (total_char_matches / max(1, total_gt_chars)) * 100.0
        plate_acc = (total_exact / max(1, n_eval)) * 100.0
        honesty = (total_conf_correct / max(1, total_conf_emitted)) * 100.0
        avg_calib_gap = total_calib_gap / max(1, n_eval)
        avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        sharp_acc = (sharp_correct / max(1, sharp_total)) * 100.0
        blur_acc = (blur_correct / max(1, blur_total)) * 100.0

        res_row = {
            "model_configuration": model_name,
            "char_accuracy_pct": round(char_acc, 2),
            "plate_accuracy_pct": round(plate_acc, 2),
            "honesty_pct": round(honesty, 2),
            "calibration_gap": round(avg_calib_gap, 4),
            "latency_ms": round(avg_latency, 2),
            "sharp_plate_acc_pct": round(sharp_acc, 2),
            "blur_plate_acc_pct": round(blur_acc, 2),
            "samples_evaluated": n_eval
        }
        benchmark_summary.append(res_row)

        print(f"[*] {model_name:<38} | Plate Acc: {plate_acc:5.1f}% | Char Acc: {char_acc:5.1f}% | Latency: {avg_latency:5.1f}ms | Honesty: {honesty:5.1f}%")

    # Save CSV
    csv_file = p_out / "ocr_benchmark.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_configuration", "char_accuracy_pct", "plate_accuracy_pct",
            "honesty_pct", "calibration_gap", "latency_ms",
            "sharp_plate_acc_pct", "blur_plate_acc_pct", "samples_evaluated"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_summary)

    print(f"\n[✓] Saved benchmark data: {csv_file}")

    # Generate Comparative Bar Chart
    chart_file = p_out / "ocr_benchmark.png"
    try:
        names = [r["model_configuration"].replace("CRNN + ", "+ ").replace("CRNN Fine-Tuned ", "CRNN ") for r in benchmark_summary]
        p_accs = [r["plate_accuracy_pct"] for r in benchmark_summary]
        c_accs = [r["char_accuracy_pct"] for r in benchmark_summary]
        lats = [r["latency_ms"] for r in benchmark_summary]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy comparison
        x = np.arange(len(names))
        width = 0.35
        rects1 = ax1.bar(x - width/2, p_accs, width, label='Full Plate Acc (%)', color='#2563eb')
        rects2 = ax1.bar(x + width/2, c_accs, width, label='Char Acc (%)', color='#10b981')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Plate & Character Accuracy on Real Indian Crops')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.5)

        # Latency comparison
        colors = ['#3b82f6' if 'CRNN' in n else '#ef4444' for n in names]
        ax2.bar(names, lats, color=colors, width=0.5)
        ax2.set_ylabel('Latency per Crop (ms)')
        ax2.set_title('Inference Latency (Lower is Better)')
        ax2.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(str(chart_file), dpi=150)
        plt.close()
        print(f"[✓] Saved benchmark visualization: {chart_file}\n")
    except Exception as e:
        print(f"[!] Chart generation error: {e}")

    return {"summary": benchmark_summary, "csv_file": str(csv_file), "chart_file": str(chart_file)}


if __name__ == "__main__":
    run_benchmark()
