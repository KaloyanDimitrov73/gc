from typing import Optional, List
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field


class Knowledge(BaseModel):
    """
    Represents a Knowledge entity from a knowledge graph.
    Essentially, each Node in the knowledge graph is internally in the SQA system treated as
    a Knowledge object.

    Args:
        uid (str): Unique identifier for the knowledge. Generated automatically if the Graph
            does not support unique identifiers.
        text (str): The value of the knowledge which is essentially the text of the node.
        knowledge_types (List[str]): The type of the knowledge. This is a list of strings
            representing the types of the knowledge. For example, a knowledge can be a
            'Person', 'Organization', etc. This is also not supported by any graph but if
            it is, the types can be set here.
    """
    uid: str = Field(default_factory=lambda: str(uuid4()))
    text: Optional[str] = Field(
        default=None, description="The value of the knowledge.")
    knowledge_types: Optional[List[str]] = Field(
        default_factory=list, description="The types of the knowledge.")

    def __lt__(self, other: "Knowledge") -> bool:
        """
        Implements a less than comparison.

        Args:
            other (Knowledge): The other Knowledge object to compare with.

        Returns:
            bool: True if this Knowledge object is less than the other, False otherwise.
        """
        if not isinstance(other, Knowledge):
            return NotImplemented
        return self.uid < other.uid

    def __eq__(self, other: object) -> bool:
        """
        Implements equality comparison.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: True if this Knowledge object is equal to the other, False otherwise.
        """
        if not isinstance(other, Knowledge):
            return NotImplemented
        return self.uid == other.uid

    def __hash__(self) -> int:
        """
        Implements hash function for Knowledge. This hashes the uid of the Knowledge object.

        Returns:
            int: The hash value of the uid.
        """
        return hash(self.uid)

    def to_series(self) -> pd.Series:
        """
        Returns the Knowledge object as a pandas Series with proper index names.

        Returns:
            pd.Series: A pandas Series representation of the Knowledge object.
        """
        return pd.Series({
            "uid": self.uid,
            "text": self.text,
        })
