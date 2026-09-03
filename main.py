import logging
import os
import sys
import time
from datetime import datetime

import yaml

from src.pt.orchestrator import pipelines, preprocessing
from src.pt.utils.constants import Constants

# Resolve the absolute path of the script's directory (Project Root)
PROJECT_ROOT = Constants.get_absolute_project_path()


def setup_logging(timestamp: str, log_dir: str):
    """
    Configures dual logging.
    Uses the directory specified in io_dirs.logs_dir.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_filename = f"pipeline_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def load_config(config_rel_path: str = "config/config.yaml") -> dict:
    """Loads YAML configuration file using an absolute path relative to project root."""
    config_abs_path = os.path.join(PROJECT_ROOT, config_rel_path)
    try:
        with open(config_abs_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[!] Error loading config at {config_abs_path}: {str(e)}")
        sys.exit(1)


def main():

    # 0. Initialize Session Metadata
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load Configuration First (to know where to log)
    config = load_config()

    # 2. Initialize Logging using path from config
    logs_base_dir = config["io_dirs"].get(
        "logs_dir", os.path.join(PROJECT_ROOT, "logs")
    )
    log_path = setup_logging(session_timestamp, logs_base_dir)

    logging.info("=" * 60)
    logging.info("PREPROCESS MAMMOGRAPHY PIPELINE")
    logging.info("=" * 60)
    logging.info(f"Log file created at: {log_path}")
    start_total = time.perf_counter()

    datalist = config["io_dirs"].get("dataset_list", [])
    task = config["hyperparameters"].get("task")

    if task == "preprocess":
        datasets = config["io_dirs"].get("kfold_ddsm", [])
        for i in datasets:
            data = os.path.join(PROJECT_ROOT, i)
            preprocessing(debug_datalist=data, config=config)

    elif task == "pipelines":
        for i in datalist:
            data = os.path.join(PROJECT_ROOT, i)
            pipelines(debug_datalist=data, config=config)

    # 3. Pipeline Summary
    end_total = time.perf_counter()
    logging.info("-" * 40)
    logging.info(f"Pipeline execution finished successfully.")
    logging.info(f"Total time elapsed: {end_total - start_total:.2f} seconds.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
