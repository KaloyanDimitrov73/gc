from typing import List, Optional, Tuple

import yaml

from implementation.file_path_management import FilePathManager


class PromptProvider:
    """Load prompts owned by the experiments package."""

    def __init__(self, prompt_dir: Optional[str] = None):
        self.file_path_manager = FilePathManager()
        self.prompt_dir = (
            prompt_dir
            if prompt_dir is not None
            else self.file_path_manager.PROMPT_DIR
        )

    def get_prompt(
        self,
        prompt_file_name: str,
    ) -> Tuple[str, List[str], List[str]]:
        """Return a prompt template and its declared variables."""
        prompt_path = self._get_prompt_path(prompt_file_name)
        if not self.file_path_manager.file_path_exists(prompt_path):
            raise FileNotFoundError(
                f"Prompt file '{prompt_file_name}' not found at '{prompt_path}'."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt_yaml = yaml.safe_load(file)

        prompt_template = prompt_yaml.get("template", "")
        if not prompt_template:
            raise ValueError(
                f"Prompt template not found in '{prompt_file_name}'."
            )

        input_variables = prompt_yaml.get("input_variables", [])
        if not isinstance(input_variables, list):
            raise ValueError(
                f"Input variables not correctly formatted in '{prompt_file_name}'."
            )

        partial_variables = prompt_yaml.get("partial_variables", [])
        if not isinstance(partial_variables, list):
            raise ValueError(
                f"Partial variables not correctly formatted in '{prompt_file_name}'."
            )

        return prompt_template, input_variables, partial_variables

    def _get_prompt_path(self, prompt_file_name: str) -> str:
        return self.file_path_manager.combine_paths(
            self.prompt_dir,
            prompt_file_name,
        )
