from InquirerPy.base.control import Choice
from sqa_system.core.data.secret_manager import SecretManager, SecretType
from sqa_system.app.cli.cli_command_helper import CLICommandHelper
from sqa_system.app.cli.handler.base.command_handler import CommandHandler
from sqa_system.core.language_model.enums.llm_enums import EndpointType


class SecretManagerCommandHandler(CommandHandler):
    """
    Command handler to manage the SecretManager.

    It allows the user to add, delete and list secrets like API keys,
    passwords, etc.
    """

    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager

    def handle_command(self):
        while True:
            action = CLICommandHelper.get_selection(
                "Manage Secrets",
                [
                    Choice("add", "Add a secret"),
                    Choice("delete", "Delete a secret"),
                    Choice("list", "List secrets"),
                    Choice("back", "Back to main menu"),
                ],
                include_exit=False
            )

            if action == "add":
                self._add_secret()
            elif action == "delete":
                self._delete_secret()
            elif action == "list":
                self._list_secrets()
            elif action == "back":
                break

    def _add_secret(self):
        secret_type = CLICommandHelper.get_selection(
            "Select secret type",
            [
                Choice(secret_type.value, secret_type.value)
                for secret_type in SecretType
            ]
        )
        if CLICommandHelper.is_exit_selection(secret_type):
            return
        if secret_type == SecretType.API_KEY.value:
            secret_identifier = CLICommandHelper.get_selection(
                "Select the endpoint for which you want to add an API key",
                [
                    Choice(endpoint.value, endpoint.name)
                    for endpoint in EndpointType
                ]
            )
        else:
            secret_identifier = CLICommandHelper.get_text_input(
                "Enter secret identifier")
        secret = CLICommandHelper.get_text_input("Enter secret")
        self.secret_manager.save_secret(SecretType(
            secret_type), secret_identifier, secret)
        CLICommandHelper.print("Secret added successfully")

    def _delete_secret(self):
        secrets = self.secret_manager.get_all_secret_ids()
        secret_identifier = CLICommandHelper.get_selection(
            "Select secret to delete",
            [Choice(entry, f"{entry[0]} ({entry[1]})") for entry in secrets]
        )
        if CLICommandHelper.is_exit_selection(secret_identifier):
            return
        secret_type = SecretType(secret_identifier[1])
        successfull = self.secret_manager.delete_secret(
            secret_type, secret_identifier[0])
        if successfull:
            CLICommandHelper.print("Secret deleted successfully")
        else:
            CLICommandHelper.print("Secret not found")

    def _list_secrets(self):
        try:
            secret_ids = self.secret_manager.get_all_secret_ids()
            if secret_ids is not None:
                for entry in secret_ids:
                    if entry is not None and entry[0] is not None and entry[1] is not None:
                        CLICommandHelper.print(f"{entry[0]} ({entry[1]})")
            else:
                CLICommandHelper.print("No secrets found")
        except Exception as e:
            CLICommandHelper.print(f"An error occurred: {e}")
