from typing import no_type_check
import typer
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)

app = typer.Typer()


class CLIController:
    """
    Main controller class that allows users to interact and 
    control the application through a command line interface.
    """
    @no_type_check
    def run(self):
        """
        The main loop of the controller that shows the main menu.
        """
        CLICommandHelper.print_success(
            "Welcome to the Question Answering System!")
        while True:
            action = CLICommandHelper.get_selection(
                "What would you like to do?",
                [
                    Choice("run_pipeline", "Run a pipeline"),
                    Choice("evaluation", "Run an experiment"),
                    Choice("secret_manager", "Manage secrets"),
                    Choice("configurations", "Manage configurations"),
                    Choice("exit", "Exit"),
                ],
                include_exit=False
            )

            if action == "run_pipeline":
                handler = self._import_handler("pipeline")
                handler.handle_command()
            elif action == "evaluation":
                handler = self._import_handler("experiment")
                handler.handle_command()
            elif action == "secret_manager":
                handler = self._import_handler("secret")
                handler.handle_command()
            elif action == "configurations":
                menu = self._import_handler("config")
                menu.run()
            elif action == "qa_generation":
                menu = self._import_handler("qa")
                menu.run()
            else:
                break

    def _import_handler(self, handler_name):
        """
        Loads the handler class based on the given handler name.
        """
        # We implemented this method because the CLI controller initially
        # took a long time to load. By importing the handler only when it is
        # needed, we can reduce the time it takes to load the CLI controller.
        # Doing this massively reduced the loading time of the application.

        if handler_name == "pipeline":
            CLICommandHelper.print("Preparing pipeline runner...")
            # pylint: disable=import-outside-toplevel
            from sqa_system.app.cli.handler.implementations.pipeline_run_command_handler \
                import PipelineRunCommandHandler
            return PipelineRunCommandHandler()
        if handler_name == "secret":
            CLICommandHelper.print("Preparing secret manager...")
            # pylint: disable=import-outside-toplevel
            from sqa_system.core.data.secret_manager import SecretManager
            from sqa_system.app.cli.handler.implementations.secret_manager_command_handler \
                import SecretManagerCommandHandler
            secret_manager = SecretManager()
            return SecretManagerCommandHandler(secret_manager)
        if handler_name == "experiment":
            CLICommandHelper.print("Preparing experiment runner...")
            # pylint: disable=import-outside-toplevel
            from sqa_system.app.cli.handler.implementations.experiment_run_command_handler \
                import ExperimentRunCommandHandler
            return ExperimentRunCommandHandler()
        if handler_name == "config":
            CLICommandHelper.print("Preparing configuration menu...")
            # pylint: disable=import-outside-toplevel
            from sqa_system.app.cli.menu.implementations.configuration_menu import ConfigurationMenu
            return ConfigurationMenu()

        raise ValueError(f"Handler {handler_name} not found.")


@app.command()
def main():
    """
    The main entry point of the application.

    Initializes a CLIController instance and starts the application loop.
    """
    controller = CLIController()
    controller.run()


if __name__ == "__main__":
    app()
