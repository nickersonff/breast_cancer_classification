"""Training and evaluation utilities for AU-Net pectoral-muscle segmentation."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from numpy.typing import NDArray
from pydicom import FileDataset, dcmread
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pt.models.au_net import AU_Net


@dataclass(frozen=True)
class AnnotationSample:
    annotation_path: Path
    image_stem: str
    patient_id: str
    split: str


@dataclass(frozen=True)
class LetterboxInfo:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


GrayArray = NDArray[np.uint8]

OUTPUT_DIR_DEFAULT: str = "runs/au_net_pectoral"
METADATA_DIR_DEFAULT: str = "data/metadata/cbis-ddsm"


def _csv_image_key(image_path: str) -> str:
    return image_path.replace("\\", "/").split("/", maxsplit=1)[0]


def load_cbis_samples(
    annotations_dir: Path, metadata_dir: Path
) -> list[AnnotationSample]:
    """Match LabelMe pectoral polygons to the official CBIS train/test CSVs."""
    csv_records: dict[str, set[tuple[str, str, str]]] = {}
    csv_paths = sorted(metadata_dir.glob("*_case_description_*_set.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CBIS case-description CSVs found in {metadata_dir}"
        )

    for csv_path in csv_paths:
        csv_split: str = "test" if "_test_set" in csv_path.name else "train"
        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row.get("image view", "").upper() != "MLO":
                    continue
                image_key: str = _csv_image_key(row["image file path"])
                csv_records.setdefault(image_key, set()).add(
                    (row["patient_id"], row["image view"].upper(), csv_split)
                )

    annotation_paths = sorted(annotations_dir.glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(
            f"No LabelMe JSON annotations found in {annotations_dir}"
        )

    samples: list[AnnotationSample] = []
    unmatched: list[str] = []
    for annotation_path in annotation_paths:
        image_stem: str = annotation_path.stem.removesuffix("_FULL")
        if image_stem.endswith("_CC"):
            continue
        matches: set[tuple[str, str, str]] = csv_records.get(image_stem, set())
        if not matches:
            unmatched.append(annotation_path.name)
            continue

        patient_splits: set[tuple[str, str]] = {
            (patient_id, split) for patient_id, _, split in matches
        }
        if len(patient_splits) != 1:
            raise ValueError(
                f"CBIS image {image_stem} maps to multiple patients or splits: "
                f"{sorted(patient_splits)}"
            )
        patient_id, split = next(iter(patient_splits))
        samples.append(AnnotationSample(annotation_path, image_stem, patient_id, split))

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} pectoral annotation(s) did not match a CBIS MLO CSV "
            f"record; examples: {unmatched[:5]}"
        )
    if not samples:
        raise ValueError("No matching CBIS pectoral-muscle annotations were found")
    return samples


def split_cbis_samples(
    samples: list[AnnotationSample], validation_fraction: float, seed: int
) -> tuple[list[AnnotationSample], list[AnnotationSample], list[AnnotationSample]]:
    """Create patient-disjoint training/validation sets from official CBIS train."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    official_train: list[AnnotationSample] = [
        sample for sample in samples if sample.split == "train"
    ]
    official_test: list[AnnotationSample] = [
        sample for sample in samples if sample.split == "test"
    ]
    patients: list[str] = sorted({sample.patient_id for sample in official_train})
    if len(patients) < 2:
        raise ValueError(
            "At least two CBIS training patients are needed for validation"
        )

    random.Random(seed).shuffle(patients)
    validation_count: int = min(
        len(patients) - 1, max(1, round(len(patients) * validation_fraction))
    )
    validation_patients: set[str] = set(patients[:validation_count])
    train_samples: list[AnnotationSample] = [
        sample
        for sample in official_train
        if sample.patient_id not in validation_patients
    ]
    validation_samples: list[AnnotationSample] = [
        sample for sample in official_train if sample.patient_id in validation_patients
    ]
    return train_samples, validation_samples, official_test


def _decode_annotation_image(
    annotation: dict[str, Any], annotation_path: Path
) -> GrayArray:
    image_data = annotation.get("imageData")
    if image_data:
        encoded: str = str(image_data).split(",", maxsplit=1)[-1]
        image_bytes: bytes = base64.b64decode(encoded)
        decoded_image: MatLike = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        image = np.asarray(decoded_image, dtype=np.uint8)
    else:
        image_path = annotation.get("imagePath")
        source_path: Path = annotation_path.parent / str(image_path)
        if not image_path or not source_path.is_file():
            raise FileNotFoundError(
                f"{annotation_path} has no embedded image and the referenced image "
                f"does not exist: {source_path}"
            )
        decoded_image: MatLike = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        image = np.asarray(decoded_image, dtype=np.uint8)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image in {annotation_path}, got {image.shape}"
        )

    expected_shape: tuple[int, int] = (
        int(annotation.get("imageHeight", image.shape[0])),
        int(annotation.get("imageWidth", image.shape[1])),
    )
    if image.shape != expected_shape:
        raise ValueError(
            f"Image dimensions in {annotation_path} are {image.shape}, "
            f"but JSON declares {expected_shape}"
        )
    return image


def rasterize_pectoral_mask(
    annotation: dict[str, Any], image_shape: tuple[int, int]
) -> GrayArray:
    """Rasterize LabelMe polygons labeled ``muscle`` into a binary mask."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    polygons = [
        shape
        for shape in annotation.get("shapes", [])
        if shape.get("label", "").strip().lower() == "muscle"
    ]
    if not polygons:
        raise ValueError("Annotation contains no shape labeled 'muscle'")

    for polygon in polygons:
        if polygon.get("shape_type") != "polygon":
            raise ValueError(
                "Pectoral annotations must use polygon shapes; got "
                f"{polygon.get('shape_type')!r}"
            )
        points: NDArray[np.float32] = np.asarray(
            polygon.get("points", []), dtype=np.float32
        )
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError(
                "A pectoral polygon must contain at least three (x, y) points"
            )
        if not np.isfinite(points).all():
            raise ValueError("Pectoral polygon contains non-finite coordinates")
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], color=(1,))

    if not np.any(mask):
        raise ValueError("Pectoral polygon rasterized to an empty mask")
    return mask


def _normalize_image(image: NDArray[Any]) -> GrayArray:
    image: NDArray[np.float32] = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("Mammogram contains NaN or infinite pixel values")
    low, high = np.percentile(image, (1.0, 99.5))
    if high <= low:
        low = float(image.min())
        high = float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized: NDArray[np.float32] = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def _letterbox_image(
    image: GrayArray, size: int, interpolation: int
) -> tuple[GrayArray, LetterboxInfo]:
    if size < 16:
        raise ValueError("image_size must be at least 16 pixels")
    height, width = image.shape
    scale: float = min(size / height, size / width)
    resized_height: int = max(1, round(height * scale))
    resized_width: int = max(1, round(width * scale))
    resized: GrayArray = np.asarray(
        cv2.resize(image, (resized_width, resized_height), interpolation=interpolation),
        dtype=np.uint8,
    )
    top: int = (size - resized_height) // 2
    left: int = (size - resized_width) // 2
    canvas: GrayArray = np.zeros((size, size), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, LetterboxInfo(
        height, width, resized_height, resized_width, top, left
    )


def prepare_pair(
    image: GrayArray, mask: GrayArray, size: int
) -> tuple[GrayArray, GrayArray]:
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask dimensions do not align: {image.shape} != {mask.shape}"
        )
    normalized: GrayArray = _normalize_image(image)
    image_canvas, _ = _letterbox_image(
        normalized,
        size,
        cv2.INTER_AREA if max(image.shape) > size else cv2.INTER_LINEAR,
    )
    mask_canvas, _ = _letterbox_image(mask.astype(np.uint8), size, cv2.INTER_NEAREST)
    return image_canvas, (mask_canvas > 0).astype(np.uint8)


def cache_cbis_samples(
    samples: list[AnnotationSample], cache_dir: Path, image_size: int
) -> dict[Path, tuple[Path, Path]]:
    """Decode JSON-embedded mammograms and cache resized image/mask PNGs."""
    image_dir: Path = cache_dir / "images"
    mask_dir: Path = cache_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[Path, tuple[Path, Path]] = {}

    for sample in samples:
        image_path: Path = image_dir / f"{sample.annotation_path.stem}.png"
        mask_path: Path = mask_dir / f"{sample.annotation_path.stem}.png"
        if not image_path.is_file() or not mask_path.is_file():
            with sample.annotation_path.open(encoding="utf-8") as file:
                annotation = json.load(file)
            image_stem: str = Path(
                str(annotation.get("imagePath", ""))
            ).stem.removesuffix("_FULL")
            if image_stem != sample.image_stem:
                raise ValueError(
                    f"JSON filename and imagePath differ in {sample.annotation_path}: "
                    f"{sample.image_stem!r} != {image_stem!r}"
                )
            image: GrayArray = _decode_annotation_image(
                annotation, sample.annotation_path
            )
            mask: GrayArray = rasterize_pectoral_mask(
                annotation, (int(image.shape[0]), int(image.shape[1]))
            )
            image, mask = prepare_pair(image, mask, image_size)
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Could not write cached image: {image_path}")
            if not cv2.imwrite(str(mask_path), mask * 255):
                raise OSError(f"Could not write cached mask: {mask_path}")
        paths[sample.annotation_path] = (image_path, mask_path)
    return paths


class CBISPectoralDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        samples: list[AnnotationSample],
        cache_paths: dict[Path, tuple[Path, Path]],
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.cache_paths = cache_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample: AnnotationSample = self.samples[index]
        image_path, mask_path = self.cache_paths[sample.annotation_path]
        image_data: MatLike = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask_data: MatLike = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image: GrayArray = np.asarray(image_data, dtype=np.uint8)
        mask: GrayArray = np.asarray(mask_data, dtype=np.uint8)
        if image.shape != mask.shape:
            raise ValueError(
                f"Cached image/mask dimensions differ for {sample.image_stem}"
            )

        if self.augment and random.random() < 0.5:
            image: GrayArray = np.asarray(cv2.flip(image, 1), dtype=np.uint8)
            mask: GrayArray = np.asarray(cv2.flip(mask, 1), dtype=np.uint8)
        image_tensor: Tensor = (
            torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
        )
        mask_tensor: Tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(
            0
        )
        return image_tensor, mask_tensor


def _run_epoch(
    model: AU_Net,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: BCEWithLogitsLoss,
    device: torch.device,
    optimizer: Adam | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss: float = 0.0
    sample_count: int = 0
    true_positive = false_positive = false_negative = true_negative = 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sample_count += images.shape[0]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits: AU_Net = model(images)
            loss: BCEWithLogitsLoss = criterion(logits, masks)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach()) * images.shape[0]
            predictions = logits.detach() >= 0
            targets = masks >= 0.5
            true_positive += int((predictions & targets).sum())
            false_positive += int((predictions & ~targets).sum())
            false_negative += int((~predictions & targets).sum())
            true_negative += int((~predictions & ~targets).sum())

    denominator: int = max(sample_count, 1)
    dice_denominator: int = 2 * true_positive + false_positive + false_negative
    sensitivity_denominator: int = true_positive + false_negative
    pixel_count: int = true_positive + false_positive + false_negative + true_negative
    return {
        "loss": total_loss / denominator,
        "dice": 2 * true_positive / max(dice_denominator, 1),
        "sensitivity": true_positive / max(sensitivity_denominator, 1),
        "accuracy": (true_positive + true_negative) / max(pixel_count, 1),
    }


def _loader(
    samples: list[AnnotationSample],
    cache_paths: dict[Path, tuple[Path, Path]],
    batch_size: int,
    workers: int,
    shuffle: bool,
    augment: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    if not samples:
        raise ValueError("A required CBIS split is empty")
    return DataLoader(
        CBISPectoralDataset(samples, cache_paths, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _save_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def train_au_net(
    annotations_dir: Path,
    metadata_dir: Path,
    output_dir: Path,
    image_size: int = 512,
    base_channels: int = 16,
    batch_size: int = 1,
    epochs: int = 100,
    learning_rate: float = 1e-5,
    patience: int = 15,
    validation_fraction: float = 0.1,
    workers: int = 2,
    seed: int = 42,
) -> tuple[AU_Net, Path, dict[str, Any]]:
    if epochs < 1 or batch_size < 1 or patience < 1 or workers < 0:
        raise ValueError(
            "epochs, batch_size and patience must be positive; workers nonnegative"
        )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[AnnotationSample] = load_cbis_samples(annotations_dir, metadata_dir)
    train_samples, validation_samples, test_samples = split_cbis_samples(
        samples, validation_fraction, seed
    )
    train_validation_samples = train_samples + validation_samples
    cache_paths: dict[Path, tuple[Path, Path]] = cache_cbis_samples(
        train_validation_samples, output_dir / f"cache_{image_size}", image_size
    )

    train_loader: DataLoader[tuple[Tensor, Tensor]] = _loader(
        train_samples, cache_paths, batch_size, workers, shuffle=True, augment=True
    )
    validation_loader: DataLoader[tuple[Tensor, Tensor]] = _loader(
        validation_samples, cache_paths, batch_size, workers, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AU_Net(base_channels=base_channels).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    settings: dict[str, Any] = {
        "image_size": image_size,
        "base_channels": base_channels,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "device": str(device),
        "sample_counts": {
            "train": len(train_samples),
            "validation": len(validation_samples),
            "cbis_test": len(test_samples),
        },
        "annotation_jsons": str(annotations_dir),
        "metadata_dir": str(metadata_dir),
    }
    _save_json(output_dir / "training_config.json", settings)

    best_dice: float = -1.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = []
    best_checkpoint: Path = output_dir / "au_net_pectoral_best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics: dict[str, float] = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_metrics: dict[str, float] = _run_epoch(
            model, validation_loader, criterion, device
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_dice={validation_metrics['dice']:.4f}"
        )

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_dice": max(best_dice, validation_metrics["dice"]),
            "settings": settings,
        }
        torch.save(checkpoint, output_dir / "au_net_pectoral_last.pt")
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint["best_validation_dice"] = best_dice
            torch.save(checkpoint, best_checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    _save_json(output_dir / "training_history.json", history)
    best_state: dict[str, Any] = torch.load(
        best_checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(best_state["model_state_dict"])
    results: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_validation_dice": best_dice,
    }
    _save_json(output_dir / "training_metrics.json", results)
    return model, best_checkpoint, results


def load_au_net_checkpoint(
    checkpoint_path: Path, device: torch.device | None = None
) -> AU_Net:
    """Load a trained AU-Net from its best or last training checkpoint."""
    target_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    settings = checkpoint.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Checkpoint has no AU-Net settings: {checkpoint_path}")
    base_channels = int(settings.get("base_channels", 16))
    model = AU_Net(base_channels=base_channels).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_au_net_cbis_test(
    annotations_dir: Path,
    metadata_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 1,
    workers: int = 2,
) -> dict[str, Any]:
    """Evaluate the official CBIS test split using a saved AU-Net checkpoint."""
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers nonnegative")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AU-Net checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    settings = checkpoint.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"Checkpoint has no AU-Net settings: {checkpoint_path}")
    image_size = int(settings.get("image_size", 512))
    model = AU_Net(base_channels=int(settings.get("base_channels", 16))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_samples: list[AnnotationSample] = [
        sample
        for sample in load_cbis_samples(annotations_dir, metadata_dir)
        if sample.split == "test"
    ]
    if not test_samples:
        raise ValueError("No annotated CBIS test MLO images were found")

    cache_paths: dict[Path, tuple[Path, Path]] = cache_cbis_samples(
        test_samples, checkpoint_path.parent / f"cache_{image_size}", image_size
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = _loader(
        test_samples,
        cache_paths,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
    )
    metrics: dict[str, float] = _run_epoch(
        model, test_loader, BCEWithLogitsLoss(), device
    )
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": "official_cbis_test",
        "sample_count": len(test_samples),
        "image_size": image_size,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path: Path = output_dir / "cbis_test_metrics.json"
    _save_json(metrics_path, results)
    print(f"CBIS test metrics saved to {metrics_path}")
    return results


def train_au_net_from_config(
    config: dict[str, Any], project_root: Path
) -> tuple[AU_Net, Path, dict[str, Any]]:
    """Run training from the ``au_net`` section in config/config.yaml."""
    annotations_value = config.get("annotations_dir")
    if not annotations_value:
        raise ValueError("config au_net.annotations_dir is required")

    def resolve_path(value: str | Path, default: str) -> Path:
        path = Path(value or default).expanduser()
        return path if path.is_absolute() else project_root / path

    output_dir: Path = resolve_path(config.get("output_dir", ""), OUTPUT_DIR_DEFAULT)
    vindr_config = config.get("vindr_test", {})
    vindr_paths: tuple[Path, Path] | None = None
    if vindr_config.get("enabled", False):
        image_root_value = vindr_config.get("image_root")
        csv_value = vindr_config.get("breast_level_csv")
        if not image_root_value or not csv_value:
            raise ValueError(
                "config au_net.vindr_test.image_root and "
                "au_net.vindr_test.breast_level_csv are required when enabled"
            )
        image_root: Path = resolve_path(image_root_value, "")
        breast_level_csv: Path = resolve_path(csv_value, "")
        if not image_root.is_dir():
            raise FileNotFoundError(f"VinDr image root does not exist: {image_root}")
        if not breast_level_csv.is_file():
            raise FileNotFoundError(
                f"VinDr breast-level CSV does not exist: {breast_level_csv}"
            )
        vindr_paths = image_root, breast_level_csv

    model, checkpoint, results = train_au_net(
        annotations_dir=resolve_path(annotations_value, ""),
        metadata_dir=resolve_path(config.get("metadata_dir", ""), METADATA_DIR_DEFAULT),
        output_dir=output_dir,
        image_size=int(config.get("image_size", 512)),
        base_channels=int(config.get("base_channels", 16)),
        batch_size=int(config.get("batch_size", 1)),
        epochs=int(config.get("epochs", 100)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        patience=int(config.get("patience", 15)),
        validation_fraction=float(config.get("validation_fraction", 0.1)),
        workers=int(config.get("workers", 2)),
        seed=int(config.get("seed", 42)),
    )

    if vindr_paths is not None:
        image_root, breast_level_csv = vindr_paths
        predict_vindr_test(
            model,
            image_root,
            breast_level_csv,
            output_dir / "vindr_test_predictions",
            image_size=int(config.get("image_size", 512)),
            limit=int(vindr_config.get("limit", 0)),
        )
    return model, checkpoint, results


def test_au_net_from_config(
    config: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    """Load the configured best checkpoint and evaluate CBIS test separately."""
    annotations_value = config.get("annotations_dir")
    if not annotations_value:
        raise ValueError("config au_net.annotations_dir is required")

    def resolve_path(value: str | Path, default: str) -> Path:
        path = Path(value or default).expanduser()
        return path if path.is_absolute() else project_root / path

    output_dir = resolve_path(config.get("test_output_dir", ""), OUTPUT_DIR_DEFAULT)
    checkpoint_path = resolve_path(
        config.get("checkpoint_path", ""),
        f"{OUTPUT_DIR_DEFAULT}/au_net_pectoral_best.pt",
    )
    results = evaluate_au_net_cbis_test(
        annotations_dir=resolve_path(annotations_value, ""),
        metadata_dir=resolve_path(config.get("metadata_dir", ""), METADATA_DIR_DEFAULT),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        batch_size=int(config.get("batch_size", 1)),
        workers=int(config.get("workers", 2)),
    )

    vindr_config = config.get("vindr_test", {})
    if vindr_config.get("enabled", False):
        image_root_value = vindr_config.get("image_root")
        csv_value = vindr_config.get("breast_level_csv")
        if not image_root_value or not csv_value:
            raise ValueError(
                "config au_net.vindr_test.image_root and "
                "config au_net.vindr_test.breast_level_csv are required when enabled"
            )
        image_root = resolve_path(image_root_value, "")
        breast_level_csv = resolve_path(csv_value, "")
        if not image_root.is_dir():
            raise FileNotFoundError(f"VinDr image root does not exist: {image_root}")
        if not breast_level_csv.is_file():
            raise FileNotFoundError(
                f"VinDr breast-level CSV does not exist: {breast_level_csv}"
            )
        model = load_au_net_checkpoint(checkpoint_path)
        predict_vindr_test(
            model,
            image_root,
            breast_level_csv,
            output_dir / "vindr_test_predictions",
            image_size=int(results["image_size"]),
            limit=int(vindr_config.get("limit", 0)),
        )
    return results


def _read_vindr_image(path: Path) -> NDArray[np.float32]:
    dataset: FileDataset = dcmread(path)
    image: NDArray[np.float32] = np.asarray(dataset.pixel_array, dtype=np.float32)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = float(image.max() + image.min()) - image
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D VinDr mammogram at {path}, got {image.shape}")
    return image


def predict_vindr_test(
    model: AU_Net,
    image_root: Path,
    breast_level_csv: Path,
    output_dir: Path,
    image_size: int,
    limit: int = 0,
) -> int:
    """Export masks for the official VinDr test MLO images (no pixel metrics)."""
    records: list[dict[str, str]] = []
    with breast_level_csv.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if (
                row.get("split", "").strip().lower() == "test"
                and row.get("view_position", "").strip().upper() == "MLO"
            ):
                records.append(row)
    if limit > 0:
        records: list[dict[str, str]] = records[:limit]
    if not records:
        raise ValueError("No VinDr test MLO images found in the breast-level CSV")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    index: list[dict[str, str]] = []
    with torch.inference_mode():
        for record in records:
            image_id: str = record["image_id"]
            image_path: Path = image_root / record["study_id"] / f"{image_id}.dicom"
            if not image_path.is_file():
                raise FileNotFoundError(f"VinDr DICOM image not found: {image_path}")
            original: NDArray[np.float32] = _read_vindr_image(image_path)
            normalized: GrayArray = _normalize_image(original)
            canvas, info = _letterbox_image(normalized, image_size, cv2.INTER_AREA)
            tensor: Tensor = (
                torch.from_numpy(canvas.copy())
                .float()
                .div_(255.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            predicted = (
                (model(tensor).sigmoid()[0, 0] >= 0.5).cpu().numpy().astype(np.uint8)
            )
            unpadded = predicted[
                info.top : info.top + info.resized_height,
                info.left : info.left + info.resized_width,
            ]
            restored: MatLike = cv2.resize(
                unpadded,
                (info.original_width, info.original_height),
                interpolation=cv2.INTER_NEAREST,
            )
            output_path: Path = (
                output_dir / f"{record['study_id']}_{image_id}_pectoral.png"
            )
            if not cv2.imwrite(str(output_path), restored * 255):
                raise OSError(f"Could not write VinDr prediction: {output_path}")
            index.append(
                {
                    "study_id": record["study_id"],
                    "image_id": image_id,
                    "image": str(image_path),
                    "prediction_mask": str(output_path),
                }
            )
            if len(index) % 100 == 0:
                print(f"VinDr test predictions: {len(index)}/{len(records)}")

    index_path: Path = output_dir / "predictions.csv"
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print(f"Saved {len(index)} VinDr test masks to {output_dir}")
    return len(index)


def main(argv: list[str] | None = None) -> None:
    project_root: Path = Path(__file__).resolve().parents[3]
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Train AU-Net for pectoral segmentation"
    )
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=project_root / METADATA_DIR_DEFAULT,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / OUTPUT_DIR_DEFAULT
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vindr-root", type=Path)
    parser.add_argument("--vindr-csv", type=Path)
    parser.add_argument("--vindr-limit", type=int, default=0)
    args: argparse.Namespace = parser.parse_args(argv)
    if (args.vindr_root is None) != (args.vindr_csv is None):
        parser.error("--vindr-root and --vindr-csv must be supplied together")

    model, _, _ = train_au_net(
        annotations_dir=args.annotations_dir,
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        workers=args.workers,
        seed=args.seed,
    )
    if args.vindr_root is not None and args.vindr_csv is not None:
        predict_vindr_test(
            model,
            args.vindr_root,
            args.vindr_csv,
            args.output_dir / "vindr_test_predictions",
            args.image_size,
            limit=args.vindr_limit,
        )


if __name__ == "__main__":
    main()
