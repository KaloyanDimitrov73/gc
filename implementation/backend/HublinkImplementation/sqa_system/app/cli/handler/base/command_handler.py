from abc import ABC, abstractmethod


class CommandHandler(ABC):
    """
    Base class for command handlers. It provides a common interface for handling user commands.

    The intended use of this class is to provide a way for the user to interact 
    with the system through the CLI interface.
    """
    @abstractmethod
    def handle_command(self):
        """Handles a user command."""
