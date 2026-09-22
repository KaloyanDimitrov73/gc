import datetime
import os
from pathlib import Path
import json
import re
from typing import List


def find_project_root():
    """
    Searches for the project root directory by going up the
    directory hierarchy until a file named 'pyproject.toml' is found.
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return str(current_dir)
        current_dir = current_dir.parent

    raise FileNotFoundError("Could not find project root")


class FilePathManager:
    """    Manages file paths for the experiments package.

    Provides generic path utilities and a JSON-based name-to-path lookup
    for shared assets (e.g. taxonomy files). The root is anchored to the
    experiments/ directory (where pyproject.toml lives).

    Implemented as a singleton.
    """

    ROOT_DIR = find_project_root()
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    FILE_PATHS_JSON = os.path.join(DATA_DIR, "file_paths", "paths.json")
    RESULTS_DIR = os.path.join(DATA_DIR, "evaluation_results")
    ASSETS_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "assets"))

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(FilePathManager, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.paths = self._load_paths()

    def _load_paths(self) -> dict:
        if not os.path.exists(self.FILE_PATHS_JSON):
            return {}
        try:
            with open(self.FILE_PATHS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"File paths JSON could not be parsed: {exc}") from exc

    def get_path(self, file_name: str) -> str:
        """
        Returns the absolute path for a registered file name.

        Args:
            file_name (str): Key in the paths JSON.

        Returns:
            str: Absolute path to the file.
        """
        if file_name not in self.paths:
            raise KeyError(
                f"File name '{file_name}' not found in paths configuration. "
                f"Make sure it is added to {self.FILE_PATHS_JSON}."
            )
        relative_path = self.paths[file_name]
        if "assets/" in relative_path:
            relative_path = relative_path.replace("assets/", "")
            return os.path.join(self.ASSETS_DIR, relative_path)
        return os.path.join(self.ROOT_DIR, relative_path)

    def combine_paths(self, *paths) -> str:
        return os.path.join(*paths)

    def get_parent_directory(self, path: str, levels: int = 1) -> str:
        for _ in range(levels):
            path = os.path.dirname(path)
        return path

    def ensure_dir_exists(self, path: str):
        if "." in os.path.basename(path):
            path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)

    def file_path_exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    def get_file_name_from_path(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")
        return os.path.basename(path)

    def get_file_directory(self, file_path: str) -> str:
        return os.path.dirname(file_path)

    def get_file_from_path(self, file_path: str) -> str:
        return os.path.basename(file_path)

    def to_relative_path(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, self.ROOT_DIR)
        rel_path = os.path.normpath(rel_path)
        return rel_path.replace("\\", "/")

    def get_files_in_folder(self, folder_path: str, file_type: str) -> List[str]:
        """
        Recursively finds all files with the given extension inside folder_path.

        Args:
            folder_path (str): Absolute or ROOT_DIR-relative folder path.
            file_type (str): File extension to filter by (e.g. '.csv', 'csv').

        Returns:
            List[str]: Absolute paths of matching files.
        """
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(self.ROOT_DIR, folder_path)
        if not os.path.exists(folder_path):
            raise ValueError(f"The folder path '{folder_path}' does not exist.")
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a directory.")

        matching_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(file_type.lower()):
                    matching_files.append(os.path.abspath(os.path.join(root, file)))
        return matching_files

    def create_evaluation_result_path(self, folder_path: str = None) -> str:
        """
        Generates a timestamped folder path for evaluation results.

        Args:
            folder_path (str): Base folder. Defaults to RESULTS_DIR.

        Returns:
            str: Path in the form <base>/<YYYY-MM-DD>/<HH-MM-SS>.
        """
        now = datetime.datetime.now()
        base = folder_path if folder_path else self.RESULTS_DIR
        return self.combine_paths(
            base,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H-%M-%S"),
        )

    def get_path_cleaned_name(self, name: str, length: int = 16, hash_value: str = "") -> str:
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())
        truncated_name = clean_name[:length]
        if len(clean_name) > length:
            if hash_value == "":
                hash_value = str(hash(clean_name))
            truncated_name = f"{truncated_name[:23]}_hash{hash_value}"
        return truncated_name
