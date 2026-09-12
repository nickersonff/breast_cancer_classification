from datetime import datetime, timezone
from logging import INFO, FileHandler, Logger, StreamHandler, basicConfig, getLogger
from os import makedirs
from os.path import exists, join
from sys import exit, stdout
from time import perf_counter
from typing import Any

from yaml import YAMLError, safe_load

from pt.orchestrator import pipelines, preprocessing
from pt.utils.constants import Constants

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT: str = Constants.get_absolute_project_path()


def setup_logging(timestamp: str, log_dir: str) -> tuple[str, Logger]:
    """
    Configures dual logging.
    Uses the directory specified in io_dirs.logs_dir.
    """
    if not exists(log_dir):
        makedirs(log_dir, exist_ok=True)

    log_filename: str = f"pipeline_{timestamp}.log"
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
        with open(config_abs_path, "r") as f:
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

    # 2. Initialize Logging using path from config
    logs_base_dir: str = config["io_dirs"].get("logs_dir", join(PROJECT_ROOT, "logs"))
    log_path, logger = setup_logging(session_timestamp, logs_base_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESS MAMMOGRAPHY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Log file created at: {log_path}")
    start_total: float = perf_counter()

    datalist: list[str] = config["io_dirs"].get("dataset_list", [])
    task: str = config["hyperparameters"].get("task")

    if task == "preprocess":
        datasets: list[str] = config["io_dirs"].get("kfold_ddsm", [])
        for i in datasets:
            data: str = join(PROJECT_ROOT, i)
            preprocessing(debug_datalist=data, config=config)

    elif task == "pipelines":
        for i in datalist:
            data: str = join(PROJECT_ROOT, i)
            pipelines(debug_datalist=data, config=config)

    # 3. Pipeline Summary
    end_total: float = perf_counter()
    logger.info("-" * 40)
    logger.info("Pipeline execution finished successfully.")
    logger.info(f"Total time elapsed: {end_total - start_total:.2f} seconds.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
