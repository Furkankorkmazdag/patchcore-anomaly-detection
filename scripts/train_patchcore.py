"""Train PatchCore from normal images and export a model for live inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.deploy import ExportType
from anomalib.engine import Engine

from common import DATASET_ROOT, PATCHCORE_RESULTS_ROOT, PROJECT_ROOT, create_patchcore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-dir", default=str(DATASET_ROOT / "train" / "normal"), help="Folder containing only normal training images.")
    parser.add_argument("--output-dir", default=str(PATCHCORE_RESULTS_ROOT), help="Directory for checkpoints and exported model.")
    parser.add_argument("--normal-split", type=float, default=0.2, help="Fraction of normal images held out for internal validation (default: 0.2).")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count (default: 2).")
    parser.add_argument("--skip-export", action="store_true", help="Do not export the TorchInferencer model after training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normal_dir = Path(args.normal_dir).expanduser()
    if not normal_dir.is_absolute():
        normal_dir = (PROJECT_ROOT / normal_dir).resolve()
    if not normal_dir.is_dir():
        raise NotADirectoryError(f"Normal training directory not found: {normal_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_count = sum(1 for item in normal_dir.iterdir() if item.is_file() and item.suffix.lower() in valid_extensions)
    if image_count < 2:
        raise ValueError(f"At least 2 normal training images are required; found {image_count} in {normal_dir}")
    if not 0 <= args.normal_split < 1:
        raise ValueError("--normal-split must be in the range [0, 1).")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers cannot be negative.")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "exported"
    export_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    print("=" * 68)
    print("PATCHCORE TRAINING")
    print("=" * 68)
    print(f"Normal images : {image_count}")
    print(f"Training data : {normal_dir}")
    print(f"Device        : {accelerator}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU           : not available; using CPU")

    # Folder automatically reserves part of the normal images for internal validation.
    datamodule = Folder(
        name="normal_training_data",
        root=normal_dir.parent,
        normal_dir=normal_dir.name,
        abnormal_dir=None,
        normal_test_dir=None,
        normal_split_ratio=args.normal_split,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        val_split_ratio=0.0,
        seed=42,
    )

    model = create_patchcore()
    engine = Engine(
        default_root_dir=str(output_dir),
        accelerator=accelerator,
        devices=1,
    )
    engine.fit(model=model, datamodule=datamodule)

    checkpoint = Path(engine.best_model_path) if engine.best_model_path else None
    if checkpoint is None or not checkpoint.is_file():
        candidates = list(output_dir.rglob("*.ckpt"))
        if not candidates:
            raise FileNotFoundError(
                f"Training finished but no checkpoint was found under {output_dir}. "
                "Check the Anomalib training output above."
            )
        checkpoint = max(candidates, key=lambda item: item.stat().st_mtime)

    print(f"Checkpoint saved: {checkpoint}")

    metadata = {
        "model": "PatchCore",
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "training_images": image_count,
        "normal_split_ratio": args.normal_split,
        "checkpoint": str(checkpoint.resolve()),
    }
    (output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.skip_export:
        print("Exporting model for camera inference...")
        exported_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=str(export_dir),
            model_file_name="patchcore",
            datamodule=datamodule,
            ckpt_path=str(checkpoint),
        )
        if exported_path is None:
            expected_path = export_dir / "patchcore.pt"
            if not expected_path.is_file():
                raise RuntimeError("Anomalib did not return an export path and patchcore.pt was not created.")
            exported_path = expected_path
        print(f"Exported model  : {exported_path}")

    print("Training workflow completed.")
    print("Next: run scripts/analyze_results.py, then scripts/detect_camera.py --camera 0")


if __name__ == "__main__":
    main()
