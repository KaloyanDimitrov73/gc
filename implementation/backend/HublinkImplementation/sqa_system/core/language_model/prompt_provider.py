from typing import Dict, Tuple
import yaml


from sqa_system.core.data.file_path_manager import FilePathManager


class PromptProvider:
    """
    Class responsible for loading prompts and providing them to the
    rest of the system.
    """

    def __init__(self):
        self.file_path_manager = FilePathManager()
        self.prompt_dir = self.file_path_manager.PROMPT_DIR

    def get_prompt(self,
                   prompt_file_name: str) -> Tuple[str, Dict[str, type], Dict[str, type]]:
        """
        Fetches the prompt by the given file name.
        The ".yaml" extension should be added to the file name.

        Args:
            prompt_file_name: Name of the prompt YAML file

        Returns:
            Tuple containing:
            - prompt template string
            - dictionary of input variables and their types
            - dictionary of partial variables and their types
        """
        if not self._check_if_prompt_exists(prompt_file_name):
            raise FileNotFoundError(
                f"Prompt file '{prompt_file_name}' not found.")

        with open(self._get_prompt_path(prompt_file_name), "r", encoding="utf-8") as file:
            prompt_yaml = yaml.safe_load(file)

        # Get the prompt template text
        prompt_template = prompt_yaml.get("template", "")
        if not prompt_template:
            raise ValueError(
                f"Prompt template not found in '{prompt_file_name}'.")

        # Get the input variables
        input_variables = prompt_yaml.get("input_variables", [])
        if not isinstance(input_variables, list):
            raise ValueError(
                f"Input variables not correctly formated in '{prompt_file_name}'.")

        # Get the partial variables
        partial_variables = prompt_yaml.get("partial_variables", [])
        if not isinstance(partial_variables, list):
            raise ValueError(
                f"Partial variables not correctly formated in '{prompt_file_name}'.")

        return prompt_template, input_variables, partial_variables

    def _check_if_prompt_exists(self, prompt_file_name: str) -> bool:
        """
        Checks if the prompt file exists in the prompt directory.
        
        Args:
            prompt_file_name: Name of the prompt YAML file
            
        Returns:
            bool: True if the prompt file exists, False otherwise
        """
        return self.file_path_manager.file_path_exists(
            self._get_prompt_path(prompt_file_name)
        )

    def _get_prompt_path(self, prompt_file_name: str) -> str:
        """
        Returns the path of the prompt file.
        
        Args:
            prompt_file_name: Name of the prompt YAML file
            
        Returns:
            str: Path to the prompt file
        """
        return self.file_path_manager.combine_paths(self.prompt_dir, prompt_file_name)
