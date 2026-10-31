import random
from json import dump
from logging import info
from os import makedirs
from os.path import basename, join
from typing import Any

from matplotlib.pyplot import boxplot, close, figure, savefig, title, xticks, ylabel

from pt.learners.local_mammo_learner import MammoLearner
from pt.preprocessing.preprocess_json import (
    preprocess_db,
    preprocess_json,
    preprocess_multiview_db,
)
from pt.utils.constants import Constants
from pt.utils.parser import CBISDDSMParser, VinDrParser

PROJECT_ROOT: str = Constants.get_absolute_project_path()
_run_sequence: int = 0


def _run_single_train(
    dataset_root: str,
    datalist_prefix: str,
    config: dict[str, Any],
    fold: int | None = None,
    batch: int = 64,
    cnn: str = "resnet",
    train_datalist: list[dict[str, str | int]] | None = None,
    valid_datalist: list[dict[str, str | int]] | None = None,
    run_name: str = "default",
) -> tuple[float | None, float | None, float | None, float | None]:
    info(f"Testing MammoLearner...model: {cnn} / batch: {batch}")
    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        conf=config,
        datalist=datalist_prefix,
        aggregation_epochs=int(config["hyperparameters"].get("aggregation_epochs", 60)),
        lr=float(config["hyperparameters"].get("lr", 0.001)),
        batch_size=batch,
        architecture=cnn,
        train_datalist=train_datalist,
        valid_datalist=valid_datalist,
        run_name=run_name,
    )
    info("test initialize...")
    learner.initialize()
    info("test train...")
    learner.train(train_loader=learner.train_loader)
    learner.save_model("final-model.safetensors")
    metrics = learner.local_valid(
        valid_loader=learner.valid_loader, is_final=True, fold=fold
    )
    learner.writer.close()
    return metrics


def init_datalist_parser(config: dict[str, Any]) -> list[dict[str, Any]]:
    databases = config.get("dataset_config", {}).get("databases", [])
    parsers: dict[str, type] = {"vindr": VinDrParser, "cbis-ddsm": CBISDDSMParser}
    datalist: list[dict[str, Any]] = []
    for database in databases:
        name = database["name"]
        if name not in parsers:
            raise ValueError(
                f"Dataset '{name}' não suportado. Escolha entre: {list(parsers)}"
            )
        datalist.extend(parsers[name](metadata_file=database["metadata_file"]).parse())
    return datalist


def run_train(
    dataset_root: str,
    datalist_prefix: str,
    config: dict[str, Any],
    batch: int = 64,
    cnn: str = "resnet",
    run_prefix: str | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    global _run_sequence
    if run_prefix is None:
        _run_sequence += 1
        run_prefix = (
            f"{basename(datalist_prefix).replace('.json', '')}_run_{_run_sequence:02d}"
        )
    return _run_single_train(
        dataset_root,
        datalist_prefix,
        config,
        batch=batch,
        cnn=cnn,
        run_name=run_prefix,
    )


def preprocessing(
    config: dict[str, Any],
    debug_datalist: str = "/home/nfferreira/data/dataset_site-1.json",
) -> list[dict[str, Any]]:
    out_path = join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix", ""))
    cnn = str(config["hyperparameters"].get("architecture", "resnet"))
    pipeline_specs = [
        ("raw_224", "", "", 224, 64),
        ("min-max_1024", "min-max", "", 1024, 16),
        ("z-score_1024", "z-score", "", 1024, 16),
        ("min-max_clahe_1024", "min-max", "CLAHE", 1024, 16),
        ("min-max_gaussian_1024", "min-max", "GAUSSIAN", 1024, 16),
        ("min-max_bilateral_1024", "min-max", "BILATERAL", 1024, 16),
        ("min-max_wiener_1024", "min-max", "WIENER", 1024, 16),
        ("min-max_median_1024", "min-max", "MEDIAN", 1024, 16),
        ("min-max_clahe-bilateral_1024", "min-max", "CLAHE+BILATERAL", 1024, 16),
        ("min-max_clahe-gaussian_1024", "min-max", "CLAHE+GAUSSIAN", 1024, 16),
        ("min-max_clahe-wiener_1024", "min-max", "CLAHE+WIENER", 1024, 16),
        ("min-max_clahe-median_1024", "min-max", "CLAHE+MEDIAN", 1024, 16),
        ("raw_384", "", "", 384, 32),
        ("raw_512", "", "", 512, 32),
        ("raw_1024", "", "", 1024, 16),
        ("raw_2048", "", "", 2048, 4),
    ]
    results: list[dict[str, Any]] = []
    for pipeline, norm, filter_name, size, batch in pipeline_specs:
        manifest_path = preprocess_json(
            out_path=out_path,
            norm=norm,
            filter=filter_name,
            size=size,
            datalist=debug_datalist,
            config=config,
        )
        metrics = run_train(
            out_path,
            manifest_path,
            config,
            batch=batch,
            cnn=cnn,
        )
        results.append(
            {
                "pipeline": pipeline,
                "accuracy": metrics[0],
                "kappa": metrics[1],
                "roc_auc": metrics[2],
                "f1": metrics[3],
            }
        )
    return results


def save_fold_metrics(
    fold_metrics: list[dict[str, float | int | None]],
    config: dict[str, Any],
) -> None:
    results_dir = config["io_dirs"].get("results_dir")
    if not results_dir or not fold_metrics:
        return

    makedirs(results_dir, exist_ok=True)
    with open(join(results_dir, "fold_metrics.json"), "w") as metrics_file:
        dump(fold_metrics, metrics_file, indent=2)

    metric_labels = {
        "accuracy": "Accuracy",
        "kappa": "Kappa",
        "roc_auc": "ROC AUC",
        "f1": "F1-score",
    }
    for metric_name, metric_label in metric_labels.items():
        values = [
            float(metrics[metric_name])
            for metrics in fold_metrics
            if metrics[metric_name] is not None
        ]
        if not values:
            continue
        figure(figsize=(6, 5))
        boxplot(values, showmeans=True)
        xticks([1], [metric_label])
        ylabel("Score")
        title(f"{metric_label} across folds")
        savefig(
            join(results_dir, f"folds_{metric_name}_boxplot.png"),
            bbox_inches="tight",
        )
        close()


def pipelines(
    config: dict[str, Any],
    debug_datalist: str = "/home/nfferreira/data/dataset_site-1.json",
    cnn: str = "resnet",
) -> None:
    if str(config["hyperparameters"].get("architecture", "")).lower() in {
        "mv_swin_t",
        "mvswin",
    } and config["io_dirs"].get("dicom_root_VINDR"):
        architecture_pipeline(config)
        return

    norms = ["min-max", "z-score"]
    filters = [
        "CLAHE",
        "BILATERAL",
        "WIENER",
        "GAUSSIAN",
        "MEDIAN",
        "CLAHE+BILATERAL",
        "CLAHE+WIENER",
        "CLAHE+GAUSSIAN",
        "CLAHE+MEDIAN",
    ]
    sizes = [224, 384, 512, 1024, 2048]
    requested = int(config["hyperparameters"].get("num_pipelines", 25))
    if requested > len(norms) * len(filters) * len(sizes):
        raise ValueError("num_pipelines exceeds the number of unique combinations")

    random.seed(42)
    combinations: list[tuple[int, int, int]] = []
    while len(combinations) < requested:
        combination = (
            random.randrange(len(norms)),
            random.randrange(len(filters)),
            random.randrange(len(sizes)),
        )
        if combination not in combinations:
            combinations.append(combination)

    out_path = join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix", ""))
    for norm_index, filter_index, size_index in combinations:
        manifest_path = preprocess_json(
            out_path=out_path,
            norm=norms[norm_index],
            filter=filters[filter_index],
            size=sizes[size_index],
            datalist=debug_datalist,
            config=config,
        )
        run_train(
            out_path,
            manifest_path,
            config,
            batch=int(config["hyperparameters"].get("batch_size", 32)),
            cnn=cnn,
        )


def architecture_pipeline(config: dict[str, Any]) -> None:
    datalist = init_datalist_parser(config)
    out_path = join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix", ""))
    makedirs(out_path, exist_ok=True)
    if str(config["hyperparameters"].get("architecture", "")).lower() == "mv_swin_t":
        paired = preprocess_multiview_db(
            out_path=out_path,
            size=224,
            norm="z-score",
            filter="CLAHE",
            datalist=datalist,
        )
        records = {"train": paired, "test": []}
    else:
        preprocess_db(
            out_path=out_path,
            size=1024,
            norm="z-score",
            filter="CLAHE",
            datalist=datalist,
        )
        records = {
            "train": [
                {
                    "image": item["npy"],
                    "label": item["label"],
                    "patient_id": item["patient_id"],
                }
                for item in datalist
            ],
            "test": [],
        }
    manifest_path = join(out_path, "architecture.json")
    with open(manifest_path, "w") as manifest_file:
        dump(records, manifest_file, indent=2)
    run_train(
        out_path,
        manifest_path,
        config,
        batch=int(config["hyperparameters"].get("batch_size", 16)),
        cnn=str(config["hyperparameters"].get("architecture", "resnet")),
    )
