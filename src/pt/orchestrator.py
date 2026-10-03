from json import dump
from os import makedirs
from os.path import basename, join
from statistics import mean, stdev
from typing import Any, cast

from matplotlib.pyplot import (
    close,
    figure,
    legend,
    plot,
    savefig,
    title,
    xlabel,
    ylabel,
)

from pt.learners.local_mammo_learner import MammoLearner
from pt.preprocessing.preprocess_json import preprocess_db, preprocess_json
from pt.utils.constants import Constants
from pt.utils.parser import CBISDDSMParser, VinDrParser
from pt.utils.records import get_records
from pt.validation.kfold import iter_kfold_splits, validation_config

PROJECT_ROOT: str = Constants.get_absolute_project_path()
_run_sequence: int = 0


def _plot_fold_metrics(
    fold_metrics: list[dict[str, float | int | None]], output_path: str
) -> None:
    metric_names = ("accuracy", "kappa", "roc_auc")
    folds = [cast(int, metrics["fold"]) for metrics in fold_metrics]

    figure(figsize=(9, 5))
    for metric_name in metric_names:
        values = [
            cast(float, metrics[metric_name])
            for metrics in fold_metrics
            if metrics[metric_name] is not None
        ]
        metric_folds = [
            cast(int, metrics["fold"])
            for metrics in fold_metrics
            if metrics[metric_name] is not None
        ]
        if values:
            plot(metric_folds, values, marker="o", linestyle="", label=metric_name)
            plot(
                folds,
                [mean(values)] * len(folds),
                linestyle="--",
                alpha=0.5,
            )

    xlabel("Fold")
    ylabel("Score")
    title("Classification metrics by fold")
    legend()
    savefig(output_path, bbox_inches="tight")
    close()


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
) -> tuple[float | None, float | None, float | None]:
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
    learner.initialize()
    learner.train(train_loader=learner.train_loader)
    learner.save_model("final-model.safetensors")
    metrics = learner.local_valid(
        valid_loader=learner.valid_loader, is_final=True, fold=fold
    )
    learner.writer.close()
    return metrics


def init_datalist_parser(config: dict[str, Any]) -> list[dict[str, Any]]:
    databases = config.get("dataset_config", {}).get("databases", [])
    parsers = {"vindr": VinDrParser, "cbis-ddsm": CBISDDSMParser}
    datalist: list[dict[str, Any]] = []
    for database in databases:
        name = database["name"]
        if name not in parsers:
            raise ValueError(
                f"Dataset '{name}' não suportado. Escolha entre: {list(parsers)}"
            )
        datalist.extend(parsers[name](metadata_file=database["metadata_file"]).parse())
    return datalist


def run_kfold(
    dataset_root: str,
    datalist_prefix: str,
    config: dict[str, Any],
    batch: int = 64,
    cnn: str = "resnet",
    run_prefix: str | None = None,
) -> None:
    settings = validation_config(config)
    if run_prefix is not None:
        settings["run_prefix"] = run_prefix

    records = get_records(datalist_prefix, settings, dataset_root)
    fold_metrics: list[dict[str, float | int | None]] = []
    for fold_index, train_records, valid_records in iter_kfold_splits(
        records,
        n_splits=settings["n_splits"],
        shuffle=settings["shuffle"],
        random_state=settings["random_state"],
    ):
        fold = fold_index + 1
        run_name = f"{settings['run_prefix']}_fold_{fold:02d}"
        metrics = _run_single_train(
            dataset_root,
            datalist_prefix,
            config,
            batch=batch,
            cnn=cnn,
            fold=fold,
            train_datalist=train_records,
            valid_datalist=valid_records,
            run_name=run_name,
        )
        fold_metrics.append(
            {
                "fold": fold,
                "train_size": len(train_records),
                "validation_size": len(valid_records),
                "accuracy": metrics[0],
                "kappa": metrics[1],
                "roc_auc": metrics[2],
            }
        )

    aggregate: dict[str, float | None] = {}
    for metric_name in ("accuracy", "kappa", "roc_auc"):
        values = [
            float(value)
            for metrics in fold_metrics
            for value in [metrics[metric_name]]
            if value is not None
        ]
        aggregate[f"{metric_name}_mean"] = mean(values) if values else None
        aggregate[f"{metric_name}_std"] = stdev(values) if len(values) > 1 else 0.0

    results_dir = config["io_dirs"].get(
        "results_dir", join(PROJECT_ROOT, "logs", "kfold")
    )
    makedirs(results_dir, exist_ok=True)
    result_path = join(results_dir, f"{settings['run_prefix']}_metrics.json")
    with open(result_path, "w") as result_file:
        dump(
            {
                "strategy": "kfold",
                "n_splits": settings["n_splits"],
                "shuffle": settings["shuffle"],
                "random_state": settings["random_state"],
                "data_list_key": settings["data_list_key"],
                "folds": fold_metrics,
                "aggregate": aggregate,
            },
            result_file,
            indent=2,
        )
    _plot_fold_metrics(
        fold_metrics,
        join(results_dir, f"{settings['run_prefix']}_metrics_scatter.png"),
    )


def run_train(
    dataset_root: str,
    datalist_prefix: str,
    config: dict[str, Any],
    batch: int = 64,
    cnn: str = "resnet",
    run_prefix: str | None = None,
) -> None:
    global _run_sequence
    settings = validation_config(config)
    if run_prefix is None:
        _run_sequence += 1
        run_prefix = (
            f"{basename(datalist_prefix).replace('.json', '')}_run_{_run_sequence:02d}"
        )
    if settings["enabled"]:
        run_kfold(dataset_root, datalist_prefix, config, batch, cnn, run_prefix)
    else:
        _run_single_train(
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
) -> None:
    out_path = join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix", ""))
    manifest_path = preprocess_json(
        out_path=out_path, datalist=debug_datalist, config=config
    )
    run_train(
        out_path,
        manifest_path,
        config,
        batch=int(config["hyperparameters"].get("batch_size", 32)),
        cnn=str(config["hyperparameters"].get("architecture", "resnet")),
    )


def pipelines(
    config: dict[str, Any],
    debug_datalist: str = "/home/nfferreira/data/dataset_site-1.json",
    cnn: str = "resnet",
) -> None:
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

    combinations = [
        (norm_index, filter_index, size_index)
        for norm_index in range(len(norms))
        for filter_index in range(len(filters))
        for size_index in range(len(sizes))
    ][:requested]

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
    preprocess_db(
        out_path=out_path,
        size=1024,
        norm="z-score",
        filter="CLAHE",
        datalist=datalist,
    )
    manifest_path = join(out_path, "architecture.json")
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
    with open(manifest_path, "w") as manifest_file:
        dump(records, manifest_file, indent=2)
    run_train(
        out_path,
        manifest_path,
        config,
        batch=int(config["hyperparameters"].get("batch_size", 16)),
        cnn=str(config["hyperparameters"].get("architecture", "resnet")),
    )
