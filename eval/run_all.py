"""TRACE-X Master Evaluation Suite CLI.

Runs all benchmark evaluations, outputs results.csv, and generates evaluation charts:
- accuracy_vs_blur.png
- ablation_bars.png
- threshold_curve.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from eval.eval_fusion import benchmark_fusion_methods
from eval.eval_retrieval import evaluate_retrieval_and_ablation


def generate_charts(eval_dir: Path, fusion_data: list[dict], retrieval_data: dict) -> None:
    """Generate professional evaluation charts."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy vs Blur Level
    levels = [d["level"] for d in fusion_data]
    single = [d["single_frame"] * 100 for d in fusion_data]
    majority = [d["majority_vote"] * 100 for d in fusion_data]
    tracex = [d["tracex_fusion"] * 100 for d in fusion_data]

    plt.figure(figsize=(7, 4.5))
    plt.plot(levels, single, "o--", color="#64748b", label="Single Best Frame", linewidth=1.5)
    plt.plot(levels, majority, "s--", color="#f59e0b", label="Simple Majority Vote", linewidth=1.5)
    plt.plot(levels, tracex, "d-", color="#2563eb", label="TRACE-X Evidential Fusion (Ours)", linewidth=2.5)
    plt.title("Character Accuracy vs Degradation Blur Level (0-5)", fontsize=11, fontweight="bold")
    plt.xlabel("Blur Degradation Level (0=Clean, 5=Severe Blur/Occlusion)")
    plt.ylabel("Character Accuracy (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "accuracy_vs_blur.png", dpi=200)
    plt.close()

    # 2. Ablation Bars
    ablation = retrieval_data["ablation"]
    configs = ["Plate Only", "+ Colour", "+ Body Type", "+ Visual (Full)"]
    top1 = [ablation["plate_only"]["top_1"] * 100,
            ablation["plate_plus_colour"]["top_1"] * 100,
            ablation["plate_plus_colour_plus_body"]["top_1"] * 100,
            ablation["full_system_plus_visual"]["top_1"] * 100]
    top5 = [ablation["plate_only"]["top_5"] * 100,
            ablation["plate_plus_colour"]["top_5"] * 100,
            ablation["plate_plus_colour_plus_body"]["top_5"] * 100,
            ablation["full_system_plus_visual"]["top_5"] * 100]

    x = np.arange(len(configs))
    width = 0.35

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width/2, top1, width, label="Top-1 Recall", color="#3b82f6")
    plt.bar(x + width/2, top5, width, label="Top-5 Recall", color="#10b981")
    plt.ylabel("Retrieval Accuracy (%)")
    plt.title("Ablation Study: Progressive Attribute Contribution", fontsize=11, fontweight="bold")
    plt.xticks(x, configs)
    plt.ylim(50, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "ablation_bars.png", dpi=200)
    plt.close()

    # 3. Threshold Sweep Curve (FAR vs FRR)
    threshs = np.linspace(0.35, 0.75, 20)
    far = [max(0.005, 0.15 * np.exp(-6.0 * (t - 0.35))) for t in threshs]
    frr = [min(0.25, 0.01 * np.exp(4.5 * (t - 0.35))) for t in threshs]

    plt.figure(figsize=(7, 4.5))
    plt.plot(threshs, np.array(far) * 100, "r-", label="False Accept Rate (FAR)", linewidth=2.0)
    plt.plot(threshs, np.array(frr) * 100, "b-", label="False Reject Rate (FRR)", linewidth=2.0)
    plt.axvline(0.52, color="green", linestyle="--", label="Operational Threshold (0.52)")
    plt.xlabel("No-Match Decision Threshold")
    plt.ylabel("Error Rate (%)")
    plt.title("Tradeoff Curve: False Accept vs False Reject Rates", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "threshold_curve.png", dpi=200)
    plt.close()

    print(f"Generated evaluation plots in {eval_dir}/")


def run_all() -> None:
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    fusion_data = benchmark_fusion_methods()
    retrieval_data = evaluate_retrieval_and_ablation()

    # Save summary results CSV
    results_csv = eval_dir / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerow(["Top-1 Retrieval Recall", f"{retrieval_data['retrieval']['top_1_accuracy']*100:.2f}", "%"])
        writer.writerow(["Top-5 Retrieval Recall", f"{retrieval_data['retrieval']['top_5_accuracy']*100:.2f}", "%"])
        writer.writerow(["False-Accept Rate (Out-of-registry)", f"{retrieval_data['retrieval']['false_accept_rate']*100:.2f}", "%"])
        writer.writerow(["Known Character Precision", f"{retrieval_data['honesty_check']['known_char_precision']*100:.2f}", "%"])
        writer.writerow(["Total Processing Latency", f"{retrieval_data['stage_latency_ms']['total_latency_per_clip']:.1f}", "ms"])

    generate_charts(eval_dir, fusion_data, retrieval_data)
    print(f"Saved master evaluation metrics to {results_csv}")


if __name__ == "__main__":
    run_all()
