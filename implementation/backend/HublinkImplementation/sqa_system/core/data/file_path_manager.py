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
    # Walk up until root
    while current_dir != current_dir.parent:
        if (current_dir / "pyproject.toml").exists():
            return str(current_dir)
        current_dir = current_dir.parent

    raise FileNotFoundError(
        "Could not find project root"
    )


class FilePathManager:
    """
    The FilePathManager is responsible for managing file paths and directories 
    across the project.

    It uses a JSON file which stores a mapping from the file name to the relative
    path of the file. This allows for easy access to files and ensures that the
    paths are consistent across the project.

    It implements the singleton design pattern to ensure only one instance is created.
    """

    # DEFAULT PATHS FOR DIRECTORIES
    ROOT_DIR = find_project_root()
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    CONFIG_DIR = os.path.join(DATA_DIR, "configs")
    FILE_PATHS_JSON = os.path.join(DATA_DIR, "file_paths", "paths.json")
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "knowledge_base")
    VECTOR_STORE_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "vector_stores")
    KNOWLEDGE_GRAPH_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "knowledge_graphs")
    RESULTS_DIR = os.path.join(DATA_DIR, "evaluation_results")
    QAGENERATION_DIR = os.path.join(DATA_DIR, "question_answering")
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
    ASSETS_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "assets"))

    def __new__(cls, *args, **kwargs):
        """Make this class a singleton"""
        if not hasattr(cls, "instance"):
            cls.instance = super(FilePathManager, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.paths = self._load_default_paths()

    def _load_default_paths(self) -> dict:
        """Load file paths from the JSON file."""
        if not os.path.exists(self.FILE_PATHS_JSON):
            raise FileNotFoundError(
                f"File paths JSON not found at {self.FILE_PATHS_JSON}")

        try:
            with open(self.FILE_PATHS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"File paths JSON could not be parsed: {exc}") from exc

    def get_path(self, file_name: str) -> str:
        """
        Returns the absolute path of a given file name.

        Args:
            file_name (str): The name of the file.

        Returns:
            str: The absolute path of the file.
        """
        if file_name not in self.paths:
            raise KeyError(
                f"""File name '{file_name}' not found in paths configuration. 
                Make sure that it is added in the paths json file.""")

        relative_path = self.paths[file_name]
        if "assets/" in relative_path:
            relative_path = relative_path.replace("assets/", "")
            return os.path.join(self.ASSETS_DIR, relative_path)
        return os.path.join(self.ROOT_DIR, relative_path)

    def add_path(self, path: str):
        """
        Adds a new file path to the paths configuration.

        Args:
            path (str): The path of the file (can be absolute or relative).
        """
        file_name = self.get_file_name_from_path(path)
        relative_path = self.to_relative_path(path)
        self.paths[file_name] = relative_path

    def remove_path(self, file_name: str):
        """
        Removes a file path from the paths configuration.

        Args:
            file_name (str): The name of the file.
        """
        if file_name in self.paths:
            del self.paths[file_name]

    def get_all_file_names(self) -> list:
        """
        Returns a list of all files that are available for the path manager.

        Returns:
            list: A list of all files.
        """
        return list(self.paths.keys())

    def get_file_name_from_path(self, path: str) -> str:
        """
        Returns the name of a file given its absolute path.

        Args:
            path (str): The absolute path of the file.

        Returns:
            str: The name of the file.

        Raises:
            FileNotFoundError: If the file is not found.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")

        return os.path.basename(path)

    def save_paths_as_default(self):
        """Save file paths to the JSON file."""
        with open(self.FILE_PATHS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.paths, f, indent=4)

    def to_relative_path(self, path: str) -> str:
        """
        Converts an absolute path to a relative path from the project root.

        Args:
            path (str): The path to convert (can be absolute or already relative).

        Returns:
            str: The relative path from the project root.
        """
        abs_path = os.path.abspath(path)
        rel_path = os.path.relpath(abs_path, self.ROOT_DIR)
        rel_path = os.path.normpath(rel_path)
        return rel_path.replace("\\", "/")

    def get_parent_directory(self, path: str, levels: int = 1) -> str:
        """
        Returns the parent directory of a given path.

        Args:
            path (str): The path to the file.
            levels (int): The number of parent directories to go up.

        Returns:
            str: The parent directory of the file.
        """
        for _ in range(levels):
            path = os.path.dirname(path)
        return path

    def ensure_dir_exists(self, path: str):
        """
        Makes sure that the directory of the given path exists.
        """
        if "." in os.path.basename(path):
            path = os.path.dirname(path)
        os.makedirs(path, exist_ok=True)

    def combine_paths(self, *paths) -> str:
        """
        Combines multiple paths into a single path.

        Args:
            *paths: The paths to combine.

        Returns:
            str: The combined path.
        """
        return os.path.join(*paths)

    def get_file_directory(self, file_path: str) -> str:
        """
        Returns the directory of a file path.

        Args:
            file_path (str): The path of the file.

        Returns:
            str: The directory of the file.
        """
        return os.path.dirname(file_path)

    def get_file_from_path(self, file_path: str) -> str:
        """
        Returns the file name from a file path.

        Args:
            file_path (str): The path of the file.

        Returns:
            str: The file name.
        """
        return os.path.basename(file_path)

    def file_path_exists(self, file_path: str) -> bool:
        """
        Checks if a file path exists.

        Args:
            file_path (str): The path of the file.

        Returns:
            bool: True if the file path exists, False otherwise.
        """
        return os.path.exists(file_path)

    def create_evaluation_result_path(self,
                                      folder_path: str = None) -> str:
        """
        Generates a folder path for the evaluation results.

        Args:
            folder_path (str): The path to the folder. If None, the default
                results directory is used.

        Returns:
            str: The path of the folder.
        """
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        date_time_folder = now.strftime("%H-%M-%S")
        if folder_path:
            results_dir = self.combine_paths(
                folder_path,
                date_folder,
                date_time_folder
            )
        else:
            results_dir = self.combine_paths(
                self.RESULTS_DIR,
                date_folder,
                date_time_folder
            )
        return results_dir

    def get_qa_generation_result_path(self, for_intermediate: bool) -> str:
        """
        Generates a folder path for the QA datasets.

        Args:
            for_intermediate (bool): Whether the dataset that should be
                saved is a final dataset or a intermediate one.

        Returns:
            str: The path of the folder.
        """
        if for_intermediate:
            now = datetime.datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            date_time_folder = now.strftime("%H-%M-%S")
            qa_dir = os.path.join(
                self.QAGENERATION_DIR,
                "intermediate",
                date_folder,
                date_time_folder)
            os.makedirs(qa_dir, exist_ok=True)
            return qa_dir

        os.makedirs(self.QAGENERATION_DIR, exist_ok=True)
        return self.QAGENERATION_DIR

    def get_cache_path(self, file_name: str) -> str:
        """
        Generates a path for a cache file.

        Args:
            file_name (str): The name of the cache file.

        Returns:
            str: The path of the cache file.
        """
        cache_dir = os.path.join(self.CACHE_DIR, file_name)
        self.ensure_dir_exists(cache_dir)
        return cache_dir

    def get_files_in_folder(self, folder_path: str, file_type: str) -> List[str]:
        """
        Searches the given folder and its subfolders for files with the specified file extension.

        Args:
            folder_path (str): The path to the folder to search. Can be absolute or relative 
                to ROOT_DIR.
            file_type (str): The file extension to filter by (e.g., '.txt', '.json').

        Returns:
            List[str]: A list of absolute paths to the matching files.
        """
        # Convert to absolute path if it's relative
        if not os.path.isabs(folder_path):
            folder_path = os.path.join(self.ROOT_DIR, folder_path)

        if not os.path.exists(folder_path):
            raise ValueError(
                f"The folder path '{folder_path}' does not exist.")
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a directory.")

        matching_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(file_type.lower()):
                    full_path = os.path.join(root, file)
                    matching_files.append(os.path.abspath(full_path))

        return matching_files

    def get_path_cleaned_name(self, name: str, length: int = 16, hash_value: str = "") -> str:
        """
        Generates a short name suitable for use in file paths.

        Args:
            name (str): The name to be cleaned and shortened.
            length (int): The maximum length of the resulting name.
            hash_value (str): An optional hash value to append if the name is truncated.

        Returns:
            str: The cleaned and shortened name.
        """
        # Remove any non-alphanumeric characters and convert to lowercase
        clean_name = re.sub(r"[^a-zA-Z0-9]", "_", name.lower())

        # Truncate to a maximum
        truncated_name = clean_name[:length]

        # If the name was truncated, append a hash to ensure uniqueness
        if len(clean_name) > length:
            if hash_value == "":
                hash_value = str(hash(clean_name))
            truncated_name = f"{truncated_name[:23]}_hash{hash_value}"

        return truncated_name
