from pathlib import Path

from implementation.file_path_management import FilePathManager
from implementation.language_model import PromptProvider


def test_default_prompt_directory_belongs_to_experiments():
    provider = PromptProvider()

    assert Path(provider.prompt_dir) == Path(FilePathManager.PROMPT_DIR)
    assert Path(provider.prompt_dir).parent == Path(FilePathManager.DATA_DIR)


def test_loads_pipeline_prompt_from_experiment_data():
    provider = PromptProvider()

    template, input_variables, partial_variables = provider.get_prompt(
        "pipes/answer_generation_prompt.yaml"
    )

    assert template
    assert input_variables == ["context_explanation", "context", "question"]
    assert partial_variables == []
