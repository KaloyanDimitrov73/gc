from typing import Any, Optional, Union
from copy import copy
import typer
from InquirerPy.prompts.confirm import ConfirmPrompt
from InquirerPy.prompts.list import ListPrompt
from InquirerPy.prompts.input import InputPrompt
from InquirerPy.base.control import Choice
from InquirerPy.prompts.checkbox import CheckboxPrompt

from sqa_system.core.config.models.base.config import Config
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class ExitSelection:
    """
    A simple class that is used to check wether the user has
    selected to exit the current selection.
    """


COLORS = typer.colors


class CLICommandHelper:
    """Provides a set of helper methods to handle user input."""
    @staticmethod
    def get_text_input(message: str,
                       default: str = "",
                       instruction: str = "") -> str:
        """
        Get user input as text.

        Args:
            message (str): The message to display to the user.
            default (str, optional): The default value to use if the user
            doesn't provide any input.
            instruction (str, optional): The instruction to display to the user.

        Returns:
            str: The user input as text.
        """
        return InputPrompt(message=message,
                           default=default,
                           instruction=instruction).execute()

    @staticmethod
    def get_int_input(message: str,
                      default: int = 0,
                      instruction: str = "") -> Union[int, None]:
        """
        Get user input as an integer.

        Args:
            message (str): The message to display to the user.
            default (int, optional): The default value to use if the user
            doesn't provide any input. Defaults to 0.
            instruction (str, optional): The instruction to display to the user.

        Returns:
            int: The user input as an integer.
        """
        try:
            input_value = InputPrompt(message=message,
                                      default=str(default),
                                      instruction=instruction,
                                      validate=lambda x: x.isdigit()).execute()
            return int(input_value)
        except ValueError:
            return None

    @staticmethod
    def get_float_input(message: str,
                        default: float = 0.0,
                        instruction: str = "") -> Union[float, None]:
        """
        Get user input as a float.

        Args:
            message (str): The message to display to the user.
            default (float, optional): The default value to use if the user
            doesn't provide any input. Defaults to 0.0.
            instruction (str, optional): The instruction to display to the user.
        """
        try:
            input_value = InputPrompt(message=message,
                                      default=str(default),
                                      instruction=instruction,
                                      validate=lambda x: x.replace(".", "", 1).isdigit()).execute()
            return float(input_value)
        except ValueError:
            return None

    @staticmethod
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def get_selection(message: str,
                      choices: list,
                      default: str = "",
                      include_exit: bool = True,
                      instruction: str = "",
                      exit_text: str = "Back") -> Any:
        """
        Get user selection from the given choices.

        Args:
            message (str): The message to display to the user.
            choices (list): List of choices for the user to select from.
            include_exit (bool): Whether to include an exit option.
            instruction (str, optional): The instruction to display to the user.
            default (str, optional): The default value to use if the user 
                doesn't provide any selection. Defaults to "".

        Returns:
            Any: The user's selection.
        """
        choices = copy(choices)

        if include_exit:
            choices.append(Choice(ExitSelection(), exit_text))
        if len(choices) <= 0:
            CLICommandHelper.print("There are no choices available.")
            raise ValueError("There are no choices available.")
        return ListPrompt(
            message=message,
            choices=choices,
            default=default,
            instruction=instruction
        ).execute()

    @staticmethod
    def is_exit_selection(selection: Any) -> bool:
        """
        Check if the given selection is an ExitSelection.

        Args:
            selection (Any): The selection to check.

        Returns:
            bool: True if the selection is an ExitSelection, False otherwise.
        """
        return isinstance(selection, ExitSelection)

    @staticmethod
    def get_multiple_selection(message: str, choices: list) -> list:
        """
        Get user selection from the given choices.

        Args:
            message (str): The message to display to the user.
            choices (list): List of choices for the user to select from.

        Returns:
            list: The user's selection.
        """
        return CheckboxPrompt(message=message,
                              choices=choices,
                              instruction="(Use arrow keys to move, Space to select, and Enter to confirm)"
                              ).execute()

    @staticmethod
    def get_confirmation(message: str,
                         instruction: str = "",
                         default: bool = True) -> bool:
        """
        Prompts the user with a confirmation message and returns a boolean 
        value indicating whether the user confirmed or not.

        Args:
            message (str): The confirmation message to display to the user.
            instruction (str, optional): The instruction to display to the user.
            default (bool, optional): The default value to use if the user 
                doesn't provide any input.

        Returns:
            bool: True if the user confirmed, False otherwise.
        """
        return ConfirmPrompt(
            message=message,
            instruction=instruction,
            default=default
        ).execute()

    @staticmethod
    def print(message: str, color: Optional[str] = ""):
        """
        Prints a message with the given color.

        Args:
            message (str): The message to print.
            color (str, optional): The color to use for the message.
        """
        if color or color == "":
            typer.secho(message, fg=color)
        else:
            typer.echo(message)

    @staticmethod
    def print_error(message: str):
        """
        Prints an error message.

        Args:
            message (str): The error message to print.
        """
        typer.secho(message, fg=typer.colors.RED)
        logger.error(message)

    @staticmethod
    def print_success(message: str):
        """
        Prints a success message.

        Args:
            message (str): The success message to print.
        """
        typer.secho(message, fg=typer.colors.GREEN)

    @staticmethod
    def print_config(config: Config, indent=0):
        """
        Prints a configuration object in a human-readable format.

        Args:
            config (Config): The configuration object to print.
            indent (int, optional): The indentation level. Defaults to 0.
        """
        # for each field that the config has, iterate through the fields and print the value
        for field_name, _ in config.model_fields.items():
            field_value = getattr(config, field_name)
            # if the value itself is a config object, call print_config on it
            if isinstance(field_value, Config):
                CLICommandHelper.print(" " * indent + f"{field_name}:")
                CLICommandHelper.print_config(field_value, indent + 2)
            # if the value is a list of configs, call print_config on each item in the list
            elif isinstance(field_value, list):
                CLICommandHelper.print(" " * indent + f"{field_name}:")
                for item in field_value:
                    if isinstance(item, Config):
                        CLICommandHelper.print_config(item, indent + 2)
            else:
                CLICommandHelper.print(
                    " " * indent + f"{field_name}: {field_value}")
