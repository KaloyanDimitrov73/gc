import weave
from typing_extensions import override
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.config.config_manager.implementations.llm_config_manager import LLMConfigManager
from sqa_system.core.language_model.llm_provider import EndpointType, LLMProvider, ValidationResult


class LLMConfigCommandHandler(ConfigCommandHandler[LLMConfig]):
    """Class responsible for handling LLM configurations"""

    def __init__(self):
        super().__init__(LLMConfigManager(), "LLM")
        self.llm_provider = LLMProvider()
        self.config_manager.load_default_configs()

    @override
    def get_additional_commands(self):
        return [Choice("test_llm", "Test LLM")]

    @override
    def handle_additional_command(self, command: str):
        if command == "test_llm":
            self.test_llm()

    @override
    def add_config(self):
        endpoint = CLICommandHelper.get_selection(
            "Select endpoint:",
            [Choice(endpoint.value, endpoint.value)
             for endpoint in EndpointType],
        )
        if CLICommandHelper.is_exit_selection(endpoint):
            return None

        endpoint_validation_result = self.llm_provider.validate_endpoint(
            EndpointType(endpoint))
        if endpoint_validation_result == ValidationResult.MISSING_API_KEY:
            api_key = CLICommandHelper.get_text_input(
                "Enter API key for the endpoint:")
            self.llm_provider.prepare_endpoint(EndpointType(endpoint), api_key)

        name_model = CLICommandHelper.get_text_input(
            "Enter the name of the model:")
        temperature = float(CLICommandHelper.get_text_input(
            "Enter the temperature:", default="0.0"))
        max_tokens = int(CLICommandHelper.get_text_input(
            "Enter the max tokens (-1 for infinite):", default="-1"))
        llm_config = ConfigFactory.create_llm_config(
            endpoint=endpoint,
            name_model=name_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm = self.llm_provider.get_llm_adapter(llm_config)
        if llm is None:
            CLICommandHelper.print(f"The configuration {llm_config.model_dump} \
                                   is not valid. LLM could not be added.")
            return None

        # Make a test call to the LLM
        try:
            weave.init("llm_config_test")
            answer = llm.generate("Hello language model!")
            CLICommandHelper.print("The LLM connection has been established. " +
                                    f"The LLM says: {answer.content}")
        except Exception as e:
            CLICommandHelper.print(f"The configuration: {llm_config.model_dump} \
                                    is not valid. LLM could not be added: {str(e)}")
            return None

        self.config_manager.add_config(llm_config)

        if CLICommandHelper.get_confirmation("Add this LLM as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(
            f"LLM configuration {self.config_name} added successfully.")

        return llm_config

    def test_llm(self):
        """Allows to select one of the available LLM configurations and test it"""
        weave.init("llm_config_test")
        llm_id = CLICommandHelper.get_selection(
            "Select LLM to test:",
            [Choice(llm_id, llm_id)
             for llm_id in self.config_manager.get_all_ids()],
        )
        if CLICommandHelper.is_exit_selection(llm_id):
            return
        llm = self.config_manager.get_config(llm_id)
        llm = self.llm_provider.get_llm_adapter(llm_config=llm)
        text_input = CLICommandHelper.get_text_input("Enter a prompt:")
        answer = llm.generate(text_input)
        CLICommandHelper.print(f"The LLM says: {answer.content}")
