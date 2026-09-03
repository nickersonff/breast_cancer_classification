from pathlib import Path


class Constants:
    # This class contains constants used throughout the project.
    @staticmethod
    def get_absolute_project_path():
        # Returns the absolute path of the project root directory.
        atual = Path(__file__).resolve()
        return str(atual.parent.parent.parent.parent)
