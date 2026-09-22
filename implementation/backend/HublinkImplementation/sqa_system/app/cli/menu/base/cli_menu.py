from abc import ABC, abstractmethod


class CLIMenu(ABC):
    """A user interaction menu for the CLI."""

    def run(self):
        """
        Starts the menu loop which continuously prompts the user 
        to input actions as defined by the menu.
        """
        while True:
            action = self.get_action()
            if action == "back":
                break

            self.handle_action(action)

    @abstractmethod
    def get_action(self):
        """
        Returns the action the user wants to perform.
        """

    @abstractmethod
    def handle_action(self, action):
        """
        Handles the action the user wants to perform.
        """
