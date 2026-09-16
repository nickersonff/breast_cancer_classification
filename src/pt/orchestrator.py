from json import dump, load
from os import makedirs
from os.path import basename, join
from random import sample, seed
from statistics import mean, stdev
from typing import Any

from pt.learners.local_mammo_learner import MammoLearner
from pt.preprocessing.preprocess_json import preprocess_db
from pt.utils.constants import Constants
from pt.validation.kfold import iter_kfold_splits, validation_config

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT: str = Constants.get_absolute_project_path()
_run_sequence: int = 0


def _run_single_train(
    dataset_root: str,
    datalist_prefix: str,
    config: dict[str, Any],
    batch: int = 64,
    cnn: str = "resnet",
    train_datalist: list[dict[str, str | int]] | None = None,
    valid_datalist: list[dict[str, str | int]] | None = None,
    run_name: str = "default",
) -> tuple[float | None, float | None, float | None]:
    print("Testing MammoLearner...")
    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        aggregation_epochs=config["hyperparameters"].get("aggregation_epochs", 60),
        lr=float(config["hyperparameters"].get("lr", 0.001)),
        batch_size=batch,
        architecture=cnn,
        conf=config,
        train_datalist=train_datalist,
        valid_datalist=valid_datalist,
        run_name=run_name,
    )
    print("test initialize...")
    learner.initialize()

    print("test train...")
    learner.train(train_loader=learner.train_loader)

    learner.save_model("final-model.safetensors")

    print("test valid...")
    acc, kappa, roc = learner.local_valid(
        valid_loader=learner.valid_loader, is_final=True
    )

    print("debug acc", acc)
    print("debug kappa", kappa)
    print("debug ROC AUC", roc)
    learner.writer.close()
    return acc, kappa, roc


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
    with open(datalist_prefix, "r") as manifest_file:
        manifest: dict[str, list[dict[str, str | int]]] = load(manifest_file)

    records = manifest.get(settings["data_list_key"], [])
    if not records:
        raise ValueError(
            f"No records found under manifest key '{settings['data_list_key']}'"
        )

    fold_metrics: list[dict[str, float | int | None]] = []
    for fold_index, train_records, valid_records in iter_kfold_splits(
        records,
        n_splits=settings["n_splits"],
        shuffle=settings["shuffle"],
        random_state=settings["random_state"],
    ):
        run_name = f"{settings['run_prefix']}_fold_{fold_index + 1:02d}"
        print(
            f"**** KFold {fold_index + 1}/{settings['n_splits']} "
            f"(train={len(train_records)}, validation={len(valid_records)}) ****"
        )
        learner_metrics = _run_single_train(
            dataset_root,
            datalist_prefix,
            config,
            batch=batch,
            cnn=cnn,
            train_datalist=train_records,
            valid_datalist=valid_records,
            run_name=run_name,
        )
        fold_metrics.append(
            {
                "fold": fold_index + 1,
                "train_size": len(train_records),
                "validation_size": len(valid_records),
                "accuracy": learner_metrics[0],
                "kappa": learner_metrics[1],
                "roc_auc": learner_metrics[2],
            }
        )

    aggregate: dict[str, float | None] = {}
    for metric_name in ("accuracy", "kappa", "roc_auc"):
        values = [
            float(metrics[metric_name] or 0.0)
            for metrics in fold_metrics
            if metrics[metric_name] is not None
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
    print(f"KFold results written to {result_path}")


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
        run_kfold(
            dataset_root,
            datalist_prefix,
            config,
            batch=batch,
            cnn=cnn,
            run_prefix=run_prefix,
        )
        return
    _run_single_train(
        dataset_root,
        datalist_prefix,
        config,
        batch=batch,
        cnn=cnn,
        run_name=run_prefix or "default",
    )


def preprocessing(
    config: dict[str, Any],
    debug_datalist: str = "/home/nfferreira/data/dataset_site-1.json",
) -> None:

    cnn: str = config["hyperparameters"].get("architecture")
    debug_dataset_root: str = join(
        PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix")
    )

    print(f"FILE: {debug_datalist}")
    """
    DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224
    """
    print(
        "**** Pipeline: DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224 ****"
    )
    preprocess_db(out_path=debug_dataset_root, datalist=debug_datalist, config=config)
    run_train(debug_dataset_root, debug_datalist, batch=64, cnn=cnn, config=config)

    """
    MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    Z-SCORE NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024
    """
    print(
        "**** Pipeline: Z-SCORE NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="z-score",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE - 1024 X 1024
    """
    print("**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE - 1024 X 1024 ****")
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="CLAHE",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - GAUSSIAN - 1024 X 1024
    """
    print("**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - GAUSSIAN - 1024 X 1024 ****")
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="GAUSSIAN",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - BILATERAL - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - BILATERAL - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="BILATERAL",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - WIENER - 1024 X 1024
    """
    print("**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - WIENER - 1024 X 1024 ****")
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="WIENER",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - MEDIAN - 1024 X 1024
    """
    print("**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - MEDIAN - 1024 X 1024 ****")
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="MEDIAN",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+BILATERAL - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+BILATERAL - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="CLAHE+BILATERAL",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+GAUSSIAN - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+GAUSSIAN - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="CLAHE+GAUSSIAN",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+WIENER - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+WIENER - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="CLAHE+WIENER",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+MEDIAN - 1024 X 1024
    """
    print(
        "**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+MEDIAN - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root,
        size=1024,
        norm="min-max",
        filter="CLAHE+MEDIAN",
        datalist=debug_datalist,
        config=config,
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 384 X 384
    """
    print(
        "**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 384 X 384 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=384, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=32, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512
    """
    print(
        "**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=512, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=32, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024
    """
    print(
        "**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=1024, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048
    """
    print(
        "**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=2048, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=4, cnn=cnn, config=config)


def pipelines(
    config: dict[str, Any],
    debug_datalist: str = "/home/nfferreira/data/dataset_site-1.json",
    cnn: str = "resnet",
) -> None:

    norm: list[str] = ["min-max", "z-score"]
    filters: list[str] = [
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
    sizes: list[int] = [224, 384, 512, 1024, 2048]
    pipe: list[tuple[int, int, int]] = []

    seed(42)
    qt_exec: int = config["hyperparameters"].get("num_pipelines", 25)

    while len(pipe) < qt_exec:
        t = (
            sample(range(len(norm)), 1)[0],
            sample(range(len(filters)), 1)[0],
            sample(range(len(sizes)), 1)[0],
        )
        if t not in pipe:
            pipe.append(t)

    for pipeline_index, i in enumerate(pipe, start=1):
        outpath: str = join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix"))

        print(f"**** Pipeline: {norm[i[0]]} - {filters[i[1]]} - {sizes[i[2]]} ****")
        preprocess_db(
            out_path=outpath,
            norm=norm[i[0]],
            filter=filters[i[1]],
            size=sizes[i[2]],
            datalist=debug_datalist,
            config=config,
        )

        run_train(
            outpath,
            debug_datalist,
            batch=config["hyperparameters"].get("batch_size", 32),
            cnn=cnn,
            config=config,
            run_prefix=(
                f"{basename(debug_datalist).replace('.json', '')}_"
                f"pipeline_{pipeline_index:02d}"
            ),
        )
