from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, get_origin, get_args, Type
from typing_extensions import override
import typer
from InquirerPy.base.control import Choice

from sqa_system.core.config.config_manager import ConfigManagerFactory
from sqa_system.core.config.config_manager import ConfigurationManager
from sqa_system.core.config.models.base.config import Config
from sqa_system.app.cli.handler.base.command_handler import CommandHandler
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.app.cli.cli_command_helper import CLICommandHelper

C = TypeVar('C', bound=Config)


class ConfigCommandHandler(CommandHandler, ABC, Generic[C]):
    """
    A base class for configuration command handlers. It provides a common interface
    for handling configuration commands such as listing, adding, and removing configurations.

    The intended use of the handler is to provide a way for the user to interact with the
    configurations of the system through the CLI interface.
    """

    def __init__(self, config_manager: ConfigurationManager[C], config_name: str):
        self.config_manager: ConfigurationManager[C] = config_manager
        self.config_manager.load_default_configs()
        self.config_name: str = config_name
        self.config_name_plural: str = config_name + "s"

    def get_additional_commands(self) -> List[Choice]:
        """
        Override this method in child classes to add additional commands.
        Returns a list of Choice objects for additional commands.
        """
        return []

    def handle_additional_command(self, command: str):
        """
        Override this method in child classes to handle additional commands.
        """

    @override
    def handle_command(self):
        while True:
            base_commands = [
                Choice("list", f"List {self.config_name_plural}"),
                Choice("add", f"Add a {self.config_name}"),
                Choice("remove", f"Remove a {self.config_name}"),
                Choice("back", "Back to main menu"),
            ]

            all_commands = base_commands[:-1] + \
                self.get_additional_commands() + [base_commands[-1]]

            action = CLICommandHelper.get_selection(
                f"{self.config_name} Management:",
                all_commands,
                include_exit=False
            )

            if action == "list":
                self._list_configs()
            elif action == "add":
                self.add_config()
            elif action == "remove":
                self._remove_config()
            elif action == "back":
                break
            else:
                self.handle_additional_command(action)

    def _list_configs(self):
        configs = self.config_manager.get_all_configs()
        if configs is None or len(configs) == 0:
            typer.secho("No " + self.config_name_plural + " available.",
                        fg=typer.colors.BRIGHT_MAGENTA)
            return

        typer.echo("\nAvailable " + self.config_name_plural + ":")
        i = 0
        for config in configs:
            CLICommandHelper.print("--" * 20)
            CLICommandHelper.print(f">> [{i}]: {config.name}")
            i += 1
            CLICommandHelper.print_config(config, 3)

    @abstractmethod
    def add_config(self) -> Config:
        """Adds a new configuration to the manager."""

    def _remove_config(self):
        configs = self.config_manager.get_all_configs()
        if not configs:
            typer.secho("No " + self.config_name_plural + " available to remove.",
                        fg=typer.colors.BRIGHT_MAGENTA)
            return

        config_id = CLICommandHelper.get_selection(
            "Select" + self.config_name + " to remove:",
            [Choice(config.uid, config.name) for config in configs]
        )
        if config_id is None or CLICommandHelper.is_exit_selection(config_id):
            return
        self.config_manager.remove_config(config_id)
        typer.secho("Configuration removed successfully.",
                    fg=typer.colors.BRIGHT_GREEN)

        if CLICommandHelper.get_confirmation("Remove this " + self.config_name + " from default?"):
            self.config_manager.save_configs_as_default()

    def _get_additional_config_params(self, params: List[AdditionalConfigParameter]) -> dict:

        # check if there are additional params
        if not params:
            return {}

        # for each additional parameter, ask the user for the value
        additional_config_params = {}
        for param in params:
            CLICommandHelper.print(
                f"\nParameter ({param.param_type.__name__}): {param.name}")
            if param.description:
                CLICommandHelper.print(f"Description: {param.description}")
            if param.available_values:
                CLICommandHelper.print(
                    f"Available values: {param.available_values}")

            if self._is_config_type(param.param_type) and param.default_value is not None:
                self._handle_config_type(param, additional_config_params)
            elif self._is_list_type(param.param_type):
                self._handle_list_type(param, additional_config_params)
            else:
                self._handle_primitive_types(param, additional_config_params)

        return additional_config_params

    def _handle_config_type(self,
                            param: AdditionalConfigParameter,
                            additional_config_params: dict):
        while True:
            field_value = param.default_value
            param_to_tune = param.name
            config_manager = ConfigManagerFactory.get_config_manager(
                field_value)
            existing_configs = config_manager.get_all_configs()

            if not existing_configs:
                raise ValueError(
                    f"No existing configs found for {param_to_tune}. Please add one first and then try again.")
                
            selected_config = CLICommandHelper.get_selection(
                f"Select {param_to_tune}:",
                [Choice(cfg, cfg.name) for cfg in existing_configs]
            )
            if CLICommandHelper.is_exit_selection(selected_config):
                return
            if selected_config is None:
                # The user canceled or typed an exit
                CLICommandHelper.print("No selection made. Please try again.")
                continue
            # If we get here, selected_value should be valid
            additional_config_params[param_to_tune] = selected_config
            break

    def _handle_list_type(self,
                          param: AdditionalConfigParameter,
                          additional_config_params: dict):
        while True:
            default_str = str(
                param.default_value) if param.default_value is not None else ""
            user_input = CLICommandHelper.get_text_input(
                "Enter values (comma-separated or Python list format): ",
                default=default_str
            )
            try:
                element_type = get_args(param.param_type)[0]
                # Clean the input string
                cleaned_input = self._clean_list_input(user_input)
                values = [x.strip()
                          for x in cleaned_input.split(",") if x.strip()]
                parsed_value = [self._convert_string_to_parameter_type(val, element_type)
                                for val in values]
                additional_config_params[param.name] = parsed_value
                break
            except ValueError as e:
                CLICommandHelper.print(f"Invalid input: {e}")

    def _handle_primitive_types(self,
                                param: AdditionalConfigParameter,
                                additional_config_params: dict):
        while True:
            default_str = str(
                param.default_value) if param.default_value is not None else ""
            user_input = CLICommandHelper.get_text_input(
                "Enter value: ", default=default_str)

            try:
                parsed_value = param.parse_value(user_input)
            except ValueError as e:
                CLICommandHelper.print(f"Invalid input: {e}")
                continue  # re-prompt until user enters a valid value

            # If param has available_values, check membership
            if param.available_values and parsed_value not in param.available_values:
                CLICommandHelper.print(
                    f"Value '{parsed_value}' is not in available values: {param.available_values}"
                )
                continue

            # Everything is valid
            additional_config_params[param.name] = parsed_value
            break

    def _convert_string_to_parameter_type(self, value: str, param_type):
        if param_type == int:
            return int(value)
        if param_type == float:
            return float(value)
        if param_type == bool:
            if isinstance(value, str):
                return value.strip().lower() in ['true', '1']
            if isinstance(value, int):
                return value != 0
            return bool(value)

        return value

    def _is_list_type(self, param_type: Type) -> bool:
        return get_origin(param_type) is list

    def _is_config_type(self, param_type: Type) -> bool:
        try:
            return issubclass(param_type, Config)
        except TypeError:
            return False

    def _clean_list_input(self, input_str: str) -> str:
        """Clean list-formatted input string."""
        cleaned = input_str.strip('[]')
        cleaned = cleaned.replace("'", "").replace('"', "")
        return cleaned
