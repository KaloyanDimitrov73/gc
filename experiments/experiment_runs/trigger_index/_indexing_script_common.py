import json
from pathlib import Path

from implementation.config.config_models import (
    KGRetrievalConfig,
    PipelineConfig,
)


def load_hublink_config(
        config_path: str | Path) -> tuple[KGRetrievalConfig, Path]:
    """Load the single HubLink retrieval config from a pipeline JSON file."""
    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise ValueError(f"Config file does not exist: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                "Indexing requires one pipeline config, but the file contains "
                f"{len(payload)} entries."
            )
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("The config file must contain a JSON object.")

    # Accept both the pipeline JSON files used by experiment runs and a full
    # ExperimentConfig JSON containing its pipeline under this field.
    pipeline_payload = payload.get("base_pipeline_config", payload)
    pipeline_config = PipelineConfig.model_validate(pipeline_payload)
    hublink_configs = [
        pipe
        for pipe in pipeline_config.pipes
        if (
            isinstance(pipe, KGRetrievalConfig)
            and pipe.retriever_type.lower() == "hublink"
        )
    ]

    if not hublink_configs:
        raise ValueError(
            f"No HubLink KG retrieval config found in {resolved_path}."
        )
    if len(hublink_configs) > 1:
        raise ValueError(
            "Indexing requires exactly one HubLink KG retrieval config, but "
            f"{len(hublink_configs)} were found in {resolved_path}."
        )

    return hublink_configs[0], resolved_path
