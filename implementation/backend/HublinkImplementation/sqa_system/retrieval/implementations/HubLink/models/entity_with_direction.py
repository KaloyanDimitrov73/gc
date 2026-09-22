from typing import List
from pydantic import BaseModel, Field

from sqa_system.core.data.models.knowledge import Knowledge
from sqa_system.core.data.models.triple import Triple


class EntityWithDirection(BaseModel):
    """
    Represents a knowledge entity including the path from the
    topic entity to the entity. It also notes the direction that
    was taken from the topic entity to the entity and the whole
    path from the topic entity to the entity.
    """
    entity: Knowledge = Field(
        ..., description="The entity from the knowledge graph.")
    left: bool = Field(
        ..., description="True if the direction for further graph traversal is left.")
    path_from_topic: List[Triple] = Field(
        default_factory=list,
        description="The path from the topic entity to the entity."
    )
