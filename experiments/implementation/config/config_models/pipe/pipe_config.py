from hublink_shared import Config


class PipeConfig(Config):
    """Configuration for a pipe"""
    # Discriminator field for pydantic models for proper serialization
    type: str
