"""OCR Temperature Scaling and Probability Calibration.

Calibrates OCR character-level softmax confidences using temperature scaling:
    p_calibrated = softmax(logits / T)

Evaluates:
- Reliability diagram before and after calibration
- Expected Calibration Error (ECE) before and after
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def find_optimal_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    initial_t: float = 1.0,
    lr: float = 0.05,
    max_iters: int = 50
) -> float:
    """Find optimal temperature T using gradient-free line search / grid search."""
    best_t = 1.0
    best_nll = float("inf")

    t_candidates = np.linspace(0.5, 3.0, 100)
    for t in t_candidates:
        scaled_logits = logits / t
        # Log-sum-exp
        lse = np.log(np.sum(np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)), axis=-1)) + np.max(scaled_logits, axis=-1)
        # NLL for true classes
        true_logits = scaled_logits[np.arange(len(labels)), labels]
        nll = np.mean(lse - true_logits)

        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t


def calibrate_validation_set() -> dict[str, float]:
    """Simulate validation calibration on synthetic samples."""
    np.random.seed(42)
    n_samples = 1000
    n_classes = 36

    # Generate synthetic uncalibrated logits with typical overconfidence
    true_labels = np.random.randint(0, n_classes, n_samples)
    logits = np.random.normal(0, 1.0, (n_samples, n_classes))
    # Make true label higher
    logits[np.arange(n_samples), true_labels] += 3.5

    # Uncalibrated probabilities (T = 1.0)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs_uncal = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    top_probs_uncal = np.max(probs_uncal, axis=-1)
    preds = np.argmax(probs_uncal, axis=-1)
    correct = (preds == true_labels).astype(int)

    ece_before = compute_ece(top_probs_uncal, correct)

    optimal_t = find_optimal_temperature(logits, true_labels)

    # Calibrated probabilities
    logits_cal = logits / optimal_t
    exp_logits_cal = np.exp(logits_cal - np.max(logits_cal, axis=-1, keepdims=True))
    probs_cal = exp_logits_cal / np.sum(exp_logits_cal, axis=-1, keepdims=True)
    top_probs_cal = np.max(probs_cal, axis=-1)

    ece_after = compute_ece(top_probs_cal, correct)

    results = {
        "optimal_temperature": round(optimal_t, 3),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "calibration_improvement_pct": round((ece_before - ece_after) / ece_before * 100, 2)
    }

    print("\n[OCR Confidence Temperature Calibration Results]")
    print(f"Optimal Temperature (T): {results['optimal_temperature']}")
    print(f"ECE Before Calibration:   {results['ece_before']:.4f}")
    print(f"ECE After Calibration:    {results['ece_after']:.4f}")
    print(f"Relative Improvement:     {results['calibration_improvement_pct']}%\n")

    return results


def main() -> None:
    calibrate_validation_set()


if __name__ == "__main__":
    main()
