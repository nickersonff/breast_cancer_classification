import base64
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch
from cv2.typing import MatLike
from pydicom import FileDataset, FileMetaDataset, dcmwrite
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pt.learners import au_net_trainer
from pt.learners.au_net_trainer import (
    AnnotationSample,
    CBISPectoralDataset,
    cache_cbis_samples,
    evaluate_au_net_cbis_test,
    load_au_net_checkpoint,
    load_cbis_samples,
    predict_vindr_test,
    prepare_pair,
    rasterize_pectoral_mask,
    split_cbis_samples,
)
from pt.models.au_net import AU_Net


def _write_annotation(path: Path, image: MatLike) -> None:
    success, encoded = cv2.imencode(".png", image)
    assert success
    annotation: dict[str, Any] = {
        "imagePath": path.stem.removesuffix("_FULL") + ".png",
        "imageData": base64.b64encode(encoded).decode("ascii"),
        "imageHeight": image.shape[0],
        "imageWidth": image.shape[1],
        "shapes": [
            {
                "label": "muscle",
                "shape_type": "polygon",
                "points": [[1, 1], [image.shape[1] - 2, 1], [1, image.shape[0] - 2]],
            }
        ],
    }
    path.write_text(json.dumps(annotation), encoding="utf-8")


def test_au_net_preserves_input_shape() -> None:
    model = AU_Net(base_channels=2).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 65, 81))
    assert output.shape == (1, 1, 65, 81)


def test_au_net_checkpoint_round_trip(tmp_path: Path) -> None:
    model = AU_Net(base_channels=2)
    checkpoint_path: Path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "settings": {"base_channels": 2},
        },
        checkpoint_path,
    )

    loaded = load_au_net_checkpoint(checkpoint_path, torch.device("cpu"))

    assert not loaded.training
    assert loaded(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def test_rasterize_pectoral_mask_fills_polygon() -> None:
    mask = rasterize_pectoral_mask(
        {
            "shapes": [
                {
                    "label": "muscle",
                    "shape_type": "polygon",
                    "points": [[1, 1], [7, 1], [1, 7]],
                }
            ]
        },
        (9, 9),
    )
    assert mask.dtype == np.uint8
    assert mask[2, 2] == 1
    assert mask[7, 7] == 0


def test_prepare_pair_keeps_mask_binary_and_aligned() -> None:
    image = np.full((64, 32), 127, dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[8:24, 4:16] = 1

    image_result, mask_result = prepare_pair(image, mask, 32)

    assert image_result.shape == (32, 32)
    assert mask_result.shape == (32, 32)
    assert set(np.unique(mask_result)) == {0, 1}
    assert 0 < int(mask_result.sum()) < int(mask.sum())


def test_cbis_json_is_matched_to_official_csv_and_cached(tmp_path: Path) -> None:
    annotations_dir: Path = tmp_path / "annotations"
    metadata_dir: Path = tmp_path / "metadata"
    cache_dir: Path = tmp_path / "cache"
    annotations_dir.mkdir()
    metadata_dir.mkdir()
    stem: str = "Mass-Training_P_00001_LEFT_MLO"
    csv_path: Path = metadata_dir / "mass_case_description_train_set.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["patient_id", "image view", "image file path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "patient_id": "P_00001",
                "image view": "MLO",
                "image file path": f"{stem}/1/000000.dcm",
            }
        )
    annotation_path: Path = annotations_dir / f"{stem}_FULL.json"
    _write_annotation(annotation_path, np.arange(64, dtype=np.uint8).reshape(8, 8))

    samples: list[AnnotationSample] = load_cbis_samples(annotations_dir, metadata_dir)
    assert samples == [AnnotationSample(annotation_path, stem, "P_00001", "train")]
    cache_paths: dict[Path, tuple[Path, Path]] = cache_cbis_samples(
        samples, cache_dir, image_size=32
    )
    dataset = CBISPectoralDataset(samples, cache_paths)
    image, mask = dataset[0]
    assert image.shape == mask.shape == (1, 32, 32)
    assert torch.all((mask == 0) | (mask == 1))


def test_cbis_test_evaluation_uses_saved_checkpoint_and_only_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    annotations_dir: Path = tmp_path / "annotations"
    metadata_dir: Path = tmp_path / "metadata"
    output_dir: Path = tmp_path / "test-output"
    annotations_dir.mkdir()
    metadata_dir.mkdir()
    samples = [
        ("Mass-Training_P_00001_LEFT_MLO", "train"),
        ("Mass-Test_P_00002_LEFT_MLO", "test"),
    ]
    for stem, split in samples:
        csv_path = metadata_dir / f"mass_case_description_{split}_set.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["patient_id", "image view", "image file path"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "patient_id": stem.split("_")[2],
                    "image view": "MLO",
                    "image file path": f"{stem}/1/image.dcm",
                }
            )
        _write_annotation(
            annotations_dir / f"{stem}_FULL.json",
            np.arange(64, dtype=np.uint8).reshape(8, 8),
        )

    checkpoint_path: Path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": AU_Net(base_channels=2).state_dict(),
            "settings": {"base_channels": 2, "image_size": 32},
        },
        checkpoint_path,
    )
    evaluated_samples: list[str] = []
    original_run_epoch = au_net_trainer._run_epoch

    def track_evaluation(
        model: AU_Net,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.BCEWithLogitsLoss,
        device: torch.device,
        optimizer: torch.optim.Adam | None = None,
    ) -> dict[str, float]:
        dataset = loader.dataset
        assert isinstance(dataset, CBISPectoralDataset)
        evaluated_samples.extend(sample.image_stem for sample in dataset.samples)
        return original_run_epoch(model, loader, criterion, device, optimizer)

    monkeypatch.setattr(au_net_trainer, "_run_epoch", track_evaluation)

    results = evaluate_au_net_cbis_test(
        annotations_dir,
        metadata_dir,
        checkpoint_path,
        output_dir,
        batch_size=1,
        workers=0,
    )

    assert evaluated_samples == ["Mass-Test_P_00002_LEFT_MLO"]
    assert results["sample_count"] == 1
    assert results["split"] == "official_cbis_test"
    assert (output_dir / "cbis_test_metrics.json").is_file()


def test_split_cbis_samples_is_patient_disjoint() -> None:
    samples: list[AnnotationSample] = [
        AnnotationSample(
            Path(f"p{patient}.json"), f"img{patient}", f"p{patient}", "train"
        )
        for patient in range(10)
    ]
    samples.append(AnnotationSample(Path("test.json"), "test", "p10", "test"))

    train, validation, test = split_cbis_samples(samples, 0.2, seed=7)

    assert {sample.patient_id for sample in train}.isdisjoint(
        sample.patient_id for sample in validation
    )
    assert len(test) == 1
    assert test[0].split == "test"


def test_missing_muscle_polygon_is_reported() -> None:
    with pytest.raises(ValueError, match="no shape labeled 'muscle'"):
        rasterize_pectoral_mask({"shapes": []}, (8, 8))


def test_vindr_test_exports_full_size_prediction_without_mask_metrics(
    tmp_path: Path,
) -> None:
    image_root: Path = tmp_path / "vindr"
    study_dir: Path = image_root / "study-1"
    study_dir.mkdir(parents=True)
    dicom_path: Path = study_dir / "image-1.dicom"
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    dicom = FileDataset(str(dicom_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dicom.Rows = 32
    dicom.Columns = 16
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME2"
    dicom.BitsAllocated = 8
    dicom.BitsStored = 8
    dicom.HighBit = 7
    dicom.PixelRepresentation = 0
    dicom.PixelData = np.arange(512, dtype=np.uint8).tobytes()
    dcmwrite(dicom_path, dicom)
    csv_path: Path = tmp_path / "breast-level.csv"
    csv_path.write_text(
        "study_id,image_id,view_position,split\n"
        "study-1,image-1,MLO,test\n"
        "study-1,other-image,CC,test\n"
        "study-1,train-image,MLO,training\n",
        encoding="utf-8",
    )
    model = AU_Net(base_channels=2).eval()

    count: int = predict_vindr_test(
        model, image_root, csv_path, tmp_path / "predictions", image_size=32
    )

    assert count == 1
    output_mask: Path = tmp_path / "predictions" / "study-1_image-1_pectoral.png"
    prediction: MatLike = cv2.imread(str(output_mask), cv2.IMREAD_GRAYSCALE)
    assert prediction.shape == (32, 16)
    assert (tmp_path / "predictions" / "predictions.csv").is_file()
