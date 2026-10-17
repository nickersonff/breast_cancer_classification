from os.path import join
from pathlib import Path

from yaml import safe_load


class Constants:
    config = None

    @staticmethod
    def get_absolute_project_path():
        atual: Path = Path(__file__).resolve()
        return str(atual.parent.parent.parent.parent)

    @staticmethod
    def get_config():
        if Constants.config is None:
            config_abs_path: str = join(
                Constants.get_absolute_project_path(), "config/config.yaml"
            )
            with open(config_abs_path) as f:
                Constants.config = safe_load(f)
        return Constants.config
