import os
from pathlib import Path
import yaml
class Constants():

    config = None  # Class variable to hold the configuration dictionary
    
    @staticmethod
    def get_absolute_project_path():
        # Returns the absolute path of the project root directory.
        atual = Path(__file__).resolve()
        return str(atual.parent.parent.parent.parent)

    @staticmethod
    def get_config():
        # Returns the configuration dictionary loaded from the config.yaml file.
        if Constants.config is None:
            config_abs_path = os.path.join(Constants.get_absolute_project_path(), "config/config.yaml")
            with open(config_abs_path, 'r') as f:
                Constants.config = yaml.safe_load(f)
        return Constants.config