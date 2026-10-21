"""Shared paths and helpers for the PatchCore anomaly-detection project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"
PATCHCORE_RESULTS_ROOT = RESULTS_ROOT / "patchcore"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"
EXPORTED_MODEL = PATCHCORE_RESULTS_ROOT / "exported" / "patchcore.pt"
THRESHOLD_FILE = ANALYSIS_ROOT / "threshold.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def create_patchcore():
    """Create PatchCore with the same architecture used for training."""
    from anomalib.models import Patchcore

    return Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )


def resolve_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Resolve a user-provided checkpoint or find one produced by training."""
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not PATCHCORE_RESULTS_ROOT.exists():
        raise FileNotFoundError(
            "No PatchCore results were found. Train a model first or pass --checkpoint."
        )

    candidates = list(PATCHCORE_RESULTS_ROOT.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .ckpt checkpoint found under {PATCHCORE_RESULTS_ROOT}. "
            "Run scripts/train_patchcore.py first."
        )

    # Prefer the conventional best model file, otherwise use the most recently modified.
    preferred = [path for path in candidates if path.name == "model.ckpt"]
    search_space = preferred or candidates
    return max(search_space, key=lambda path: path.stat().st_mtime)


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensors and array-like model outputs to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_outputs(value: Any) -> Iterator[Any]:
    """Recursively flatten nested prediction outputs returned by Anomalib."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_outputs(item)
    elif value is not None:
        yield value


def list_images(directory: str | Path) -> list[Path]:
    """Return supported images in a directory, sorted by filename."""
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Image directory not found: {path}")
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda item: item.name.lower(),
    )


def normalize_paths(value: Any) -> list[str]:
    """Normalize Anomalib image_path values to a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(normalize_paths(item))
        return result
    return [str(value)]


def prediction_records(outputs: Any, fallback_paths: list[Path], label: str) -> list[dict[str, Any]]:
    """Extract image-level scores and paths from varying Anomalib batch shapes."""
    records: list[dict[str, Any]] = []
    fallback_index = 0

    for item in flatten_outputs(outputs):
        if not hasattr(item, "pred_score"):
            continue

        scores = to_numpy(item.pred_score).astype(float).reshape(-1)
        paths = normalize_paths(getattr(item, "image_path", None))

        if len(paths) != len(scores):
            paths = []
            for _ in scores:
                if fallback_index < len(fallback_paths):
                    paths.append(str(fallback_paths[fallback_index]))
                fallback_index += 1
        else:
            fallback_index += len(scores)

        for index, score in enumerate(scores):
            path = paths[index] if index < len(paths) else f"unknown_{len(records):06d}"
            records.append({"label": label, "image_path": path, "score": float(score)})

    # In some Anomalib versions each item has an image path, but scores are exposed via a wrapper.
    # If no output could be parsed, fail clearly rather than create misleading analysis files.
    if not records:
        raise RuntimeError(
            "Could not extract pred_score values from Anomalib predictions. "
            "Check the installed anomalib version and checkpoint compatibility."
        )
    return records
