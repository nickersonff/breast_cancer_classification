import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from pt.orchestrator import pipelines, preprocessing
from pt.utils.constants import Constants

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT: str = Constants.get_absolute_project_path()


def setup_logging(timestamp: str, log_dir: str) -> tuple[str, logging.Logger]:
    """
    Configures dual logging.
    Uses the directory specified in io_dirs.logs_dir.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_filename: str = f"pipeline_{timestamp}.log"
    log_path: str = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logger: logging.Logger = logging.getLogger(__name__)
    return log_path, logger


def load_config(config_rel_path: str = "config/config.yaml") -> dict[str, Any]:
    """Loads YAML configuration file
    using an absolute path relative to project root."""
    config_abs_path: str = os.path.join(PROJECT_ROOT, config_rel_path)
    try:
        with open(config_abs_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[!] Error loading config at {config_abs_path}: {e!s}")
        sys.exit(1)
    except OSError as e:
        print(f"[!] OS error while accessing config at {config_abs_path}: {e!s}")
        sys.exit(1)


def main():
    # 0. Initialize Session Metadata
    session_timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. Load Configuration First (to know where to log)
    config: dict[str, Any] = load_config()

    # 2. Initialize Logging using path from config
    logs_base_dir: str = config["io_dirs"].get(
        "logs_dir", os.path.join(PROJECT_ROOT, "logs")
    )
    log_path, logger = setup_logging(session_timestamp, logs_base_dir)

    logger.info("=" * 60)
    logger.info("PREPROCESS MAMMOGRAPHY PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Log file created at: {log_path}")
    start_total: float = time.perf_counter()

    datalist: list[str] = config["io_dirs"].get("dataset_list", [])
    task: str = config["hyperparameters"].get("task")

    if task == "preprocess":
        datasets: list[str] = config["io_dirs"].get("kfold_ddsm", [])
        for i in datasets:
            data: str = os.path.join(PROJECT_ROOT, i)
            preprocessing(debug_datalist=data, config=config)

    elif task == "pipelines":
        for i in datalist:
            data: str = os.path.join(PROJECT_ROOT, i)
            pipelines(debug_datalist=data, config=config)

    # 3. Pipeline Summary
    end_total: float = time.perf_counter()
    logger.info("-" * 40)
    logger.info("Pipeline execution finished successfully.")
    logger.info(f"Total time elapsed: {end_total - start_total:.2f} seconds.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
