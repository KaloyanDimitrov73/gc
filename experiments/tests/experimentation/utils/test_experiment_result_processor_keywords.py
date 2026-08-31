from types import SimpleNamespace

from implementation.experimentation.utils.experiment_result_processor import (
    ExperimentResultProcessor,
)
from implementation.shared_models.pipe_io_data import PipeIOData


def test_result_row_contains_extracted_keywords():
    pipe_io_data = PipeIOData(
        initial_question="Who published in 2017?",
        question_id="question-1",
        extracted_keywords=["2017"],
    )
    emissions_data = SimpleNamespace(
        cpu_count=1,
        cpu_energy_consumption=0.0,
        cpu_model="cpu",
        gpu_count=0,
        gpu_energy_consumption=0.0,
        gpu_model=None,
        os="Windows",
        ram_energy_consumption=0.0,
        total_energy_consumption=0.0,
        timestamp="timestamp",
        tracking_duration=0.0,
        emissions=0.0,
    )
    pipeline_data = SimpleNamespace(
        pipe_io_data=pipe_io_data,
        runtime=0.1,
        llm_stats=SimpleNamespace(cost=0.0, total_tokens=0),
        emissions_data=emissions_data,
        weave_url="",
    )
    config = SimpleNamespace(config_hash="config-hash", pipes=[])

    rows = ExperimentResultProcessor().process_result_rows(
        pipeline_datas=[pipeline_data],
        prepared_dataset=[{
            "uid": "question-1",
            "question": "Who published in 2017?",
        }],
        config=config,
    )

    assert rows[0]["extracted_keywords"] == ["2017"]
