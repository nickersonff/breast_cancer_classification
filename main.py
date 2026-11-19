from datetime import datetime, timezone
from logging import INFO, FileHandler, Logger, StreamHandler, basicConfig, getLogger
from os import makedirs
from os.path import exists, join
from pathlib import Path
from sys import exit, stdout
from time import perf_counter
from typing import Any

from yaml import YAMLError, safe_load

from pt.orchestrator import (
    architecture_pipeline,
    pipelines,
    preprocessing,
    save_fold_metrics,
)
from pt.utils.constants import Constants

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT: str = Constants.get_absolute_project_path()


def setup_logging(timestamp: str, log_dir: str, model: str) -> tuple[str, Logger]:
    """
    Configures dual logging.
    Uses the directory specified in io_dirs.logs_dir.
    """
    if not exists(log_dir):
        makedirs(log_dir, exist_ok=True)

    log_filename: str = f"pipeline_{model}_{timestamp}.log"
    log_path: str = join(log_dir, log_filename)

    basicConfig(
        level=INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[FileHandler(log_path), StreamHandler(stdout)],
    )
    logger: Logger = getLogger(__name__)
    return log_path, logger


def load_config(config_rel_path: str = "config/config.yaml") -> dict[str, Any]:
    """Loads YAML configuration file
    using an absolute path relative to project root."""
    config_abs_path: str = join(PROJECT_ROOT, config_rel_path)
    try:
        with open(config_abs_path) as f:
            return safe_load(f)
    except YAMLError as e:
        print(f"[!] Error loading config at {config_abs_path}: {e!s}")
        exit(1)
    except OSError as e:
        print(f"[!] OS error while accessing config at {config_abs_path}: {e!s}")
        exit(1)


def main():
    # 0. Initialize Session Metadata
    session_timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. Load Configuration First (to know where to log)
    config: dict[str, Any] = load_config()
    task: str = config["hyperparameters"].get("task")

    # 2. Initialize Logging using path from config
    logs_base_dir: str = config["io_dirs"].get("logs_dir", join(PROJECT_ROOT, "logs"))
    cnn: str = (
        "au_net"
        if task in ("au_net", "au_net_test")
        else config["hyperparameters"].get("architecture", "unknown")
    )
    log_path, logger = setup_logging(session_timestamp, logs_base_dir, cnn)

    logger.info("=" * 60)
    logger.info("PREPROCESS MAMMOGRAPHY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Log file created at: {log_path}")
    start_total: float = perf_counter()

    datalist: list[str] = config["io_dirs"].get("dataset_list", [])

    if task == "preprocess":
        datasets: list[str] = config["io_dirs"].get("kfold_ddsm", [])
        fold_metrics: list[dict[str, Any]] = []
        for fold, i in enumerate(datasets, start=1):
            data: str = join(PROJECT_ROOT, i)
            metrics = preprocessing(debug_datalist=data, config=config)
            for pipeline_metrics in metrics:
                fold_metrics.append({"fold": fold, **pipeline_metrics})
        save_fold_metrics(fold_metrics, config)

    elif task == "pipelines":
        for i in datalist:
            data: str = join(PROJECT_ROOT, i)
            pipelines(debug_datalist=data, config=config)

    elif task == "architecture":
        architecture_pipeline(config=config)
    elif task == "au_net":
        from pt.learners.au_net_trainer import train_au_net_from_config

        train_au_net_from_config(config.get("au_net", {}), Path(PROJECT_ROOT))
    elif task == "au_net_test":
        from pt.learners.au_net_trainer import test_au_net_from_config

        test_au_net_from_config(config.get("au_net", {}), Path(PROJECT_ROOT))
    else:
        raise ValueError(
            f"Unknown pipeline task '{task}'. "
            "Expected 'preprocess', 'pipelines', 'architecture', 'au_net', "
            "or 'au_net_test'."
        )

    # 3. Pipeline Summary
    end_total: float = perf_counter()
    logger.info("-" * 40)
    logger.info("Pipeline execution finished successfully.")
    logger.info(f"Total time elapsed: {end_total - start_total:.2f} seconds.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
