import os
import random

from src.pt.learners.local_mammo_learner import MammoLearner
from src.pt.preprocessing.preprocess_json import preprocess_db
from src.pt.utils.constants import Constants

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT = Constants.get_absolute_project_path()


def run_train(
    dataset_root, datalist_prefix, batch=64, cnn="resnet", config: dict = None
):
    print("Testing MammoLearner...")
    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        aggregation_epochs=config["hyperparameters"].get("aggregation_epochs", 60),
        lr=config["hyperparameters"].get("lr", 0.001),
        batch_size=batch,
        architecture=cnn,
        conf=config,
    )
    print("test initialize...")
    learner.initialize()

    print("test train...")
    learner.train(train_loader=learner.train_loader)

    learner.save_model("final-model.pt")

    print("test valid...")
    acc, kappa, roc = learner.local_valid(
        valid_loader=learner.valid_loader, is_final=True
    )

    print("debug acc", acc)
    print("debug kappa", kappa)
    print("debug ROC AUC", roc)


def preprocessing(
    debug_datalist="/home/nfferreira/data/dataset_site-1.json", config: dict = None
):

    cnn = config["hyperparameters"].get("architecture")
    debug_dataset_root = os.path.join(
        PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix")
    )

    print(f"FILE: {debug_datalist}")
    """
    DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224
    """
    print(
        f"**** Pipeline: DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224 ****"
    )
    preprocess_db(out_path=debug_dataset_root, datalist=debug_datalist, config=config)
    run_train(debug_dataset_root, debug_datalist, batch=64, cnn=cnn, config=config)

    """
    MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024
    """
    print(
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****"
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
        f"**** Pipeline: Z-SCORE NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****"
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
    print(f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE - 1024 X 1024 ****")
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
    print(
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - GAUSSIAN - 1024 X 1024 ****"
    )
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
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - BILATERAL - 1024 X 1024 ****"
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
    print(f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - WIENER - 1024 X 1024 ****")
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
    print(f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - MEDIAN - 1024 X 1024 ****")
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
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+BILATERAL - 1024 X 1024 ****"
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
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+GAUSSIAN - 1024 X 1024 ****"
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
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+WIENER - 1024 X 1024 ****"
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
        f"**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+MEDIAN - 1024 X 1024 ****"
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
        f"**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 384 X 384 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=384, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=32, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512
    """
    print(
        f"**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=512, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=32, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024
    """
    print(
        f"**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=1024, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, config=config)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048
    """
    print(
        f"**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048 ****"
    )
    preprocess_db(
        out_path=debug_dataset_root, size=2048, datalist=debug_datalist, config=config
    )
    run_train(debug_dataset_root, debug_datalist, batch=4, cnn=cnn, config=config)


def pipelines(
    debug_datalist="/home/nfferreira/data/dataset_site-1.json",
    cnn="resnet",
    config: dict = None,
):

    norm = ["min-max", "z-score"]
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
    pipe = []

    random.seed(42)
    qt_exec = config["hyperparameters"].get("num_pipelines", 25)

    while len(pipe) < qt_exec:
        t = (
            random.sample(range(len(norm)), k=1)[0],
            random.sample(range(len(filters)), k=1)[0],
            random.sample(range(len(sizes)), k=1)[0],
        )
        if t not in pipe:
            pipe.append(t)

    for i in pipe:
        outpath = os.path.join(PROJECT_ROOT, config["io_dirs"].get("preprocess_prefix"))

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
        )
