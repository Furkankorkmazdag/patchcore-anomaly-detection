"""Evaluate PatchCore image scores and create a CSV, threshold file, and score plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine

from common import (
    ANALYSIS_ROOT,
    DATASET_ROOT,
    PROJECT_ROOT,
    create_patchcore,
    list_images,
    prediction_records,
    resolve_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", default=str(DATASET_ROOT / "test" / "normal"), help="Held-out normal images for threshold calibration.")
    parser.add_argument("--anomaly-dir", default=str(DATASET_ROOT / "test" / "anomaly"), help="Images containing anomalies for evaluation.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. If omitted, the latest training checkpoint is searched for.")
    parser.add_argument("--output-dir", default=str(ANALYSIS_ROOT), help="Directory for CSV, plot, and threshold JSON.")
    parser.add_argument("--normal-percentile", type=float, default=99.0, help="Normal-score percentile used as image-level threshold (default: 99).")
    parser.add_argument("--batch-size", type=int, default=8, help="Prediction batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count (default: 2).")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def predict_directory(directory: Path, label: str, checkpoint: Path, output_dir: Path, batch_size: int, num_workers: int) -> list[dict[str, Any]]:
    images = list_images(directory)
    if not images:
        raise ValueError(f"No images found in {directory}")

    dataset = PredictDataset(path=str(directory), image_size=(256, 256))
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    engine = Engine(
        default_root_dir=str(output_dir / "prediction_runs" / label),
        accelerator=accelerator,
        devices=1,
    )
    model = create_patchcore()
    print(f"Predicting {len(images)} {label} images on {accelerator.upper()}...")
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )
    if predictions is None:
        raise RuntimeError(f"Anomalib returned no predictions for {directory}")

    records = prediction_records(predictions, images, label)
    if len(records) != len(images):
        print(
            f"Warning: {len(images)} images were found in {directory}, "
            f"but {len(records)} prediction scores were returned."
        )
    return records


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    if not 0 < args.normal_percentile < 100:
        raise ValueError("--normal-percentile must be between 0 and 100.")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative.")

    normal_dir = resolve_path(args.normal_dir)
    anomaly_dir = resolve_path(args.anomaly_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(args.checkpoint)

    normal_records = predict_directory(normal_dir, "normal", checkpoint, output_dir, args.batch_size, args.num_workers)
    anomaly_records = predict_directory(anomaly_dir, "anomaly", checkpoint, output_dir, args.batch_size, args.num_workers)
    all_records = normal_records + anomaly_records

    normal_scores = np.asarray([row["score"] for row in normal_records], dtype=np.float64)
    anomaly_scores = np.asarray([row["score"] for row in anomaly_records], dtype=np.float64)
    threshold = float(np.percentile(normal_scores, args.normal_percentile))

    for row in all_records:
        row["predicted_label"] = "anomaly" if row["score"] >= threshold else "normal"
        row["threshold"] = threshold

    csv_path = output_dir / "prediction_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "predicted_label", "image_path", "score", "threshold"])
        writer.writeheader()
        writer.writerows(all_records)

    false_positives = int(np.count_nonzero(normal_scores >= threshold))
    true_negatives = len(normal_scores) - false_positives
    true_positives = int(np.count_nonzero(anomaly_scores >= threshold))
    false_negatives = len(anomaly_scores) - true_positives
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(true_positives + true_negatives, len(all_records))

    threshold_path = output_dir / "threshold.json"
    threshold_data = {
        "image_anomaly_threshold": threshold,
        "normal_percentile": args.normal_percentile,
        "normal_samples": int(len(normal_scores)),
        "anomaly_samples": int(len(anomaly_scores)),
        "checkpoint": str(checkpoint),
        "note": "Calibrated from held-out normal images. Validate on data representative of the deployment camera and environment.",
    }
    threshold_path.write_text(json.dumps(threshold_data, indent=2), encoding="utf-8")

    plot_path = output_dir / "score_distribution.png"
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=30, alpha=0.65, label="Normal images")
    plt.hist(anomaly_scores, bins=30, alpha=0.65, label="Anomaly images")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold (P{args.normal_percentile:g}) = {threshold:.4f}")
    plt.xlabel("PatchCore image anomaly score")
    plt.ylabel("Number of images")
    plt.title("PatchCore anomaly-score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    summary = {
        "threshold": threshold,
        "normal_false_positive_rate": safe_divide(false_positives, len(normal_scores)),
        "anomaly_recall": recall,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAnalysis complete")
    print(f"Checkpoint          : {checkpoint}")
    print(f"Normal images       : {len(normal_scores)}")
    print(f"Anomaly images      : {len(anomaly_scores)}")
    print(f"Selected threshold  : {threshold:.6f} (normal P{args.normal_percentile:g})")
    print(f"Normal false alarms : {false_positives}/{len(normal_scores)} ({100 * summary['normal_false_positive_rate']:.2f}%)")
    print(f"Anomaly recall      : {100 * recall:.2f}%")
    print(f"Precision / F1      : {100 * precision:.2f}% / {100 * f1:.2f}%")
    print(f"CSV                 : {csv_path}")
    print(f"Score plot          : {plot_path}")
    print(f"Camera threshold    : {threshold_path}")


if __name__ == "__main__":
    main()
