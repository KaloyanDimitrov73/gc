from typing import Union
import pandas as pd
import re
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from .knowledge import Knowledge


class Triple(BaseModel):
    """
    Represents a relation between two Knowledge entities of a graph conforming to
    the RDF standard format of (subject, predicate, object). The subject and object are 
    Knowledge entities and the predicate is a string that describes the relation between
    the two entities.    
    """
    entity_subject: Knowledge = Field(
        ..., description="The head entity of the relation")
    predicate: str = Field(
        ..., description="The description of the relation between the two entities")
    entity_object: Knowledge = Field(
        ..., description="The tail entity of the relation.")

    def __lt__(self, other: "Triple") -> bool:
        """
        Overrides the less than comparison.

        Args:
            other (Triple): The other Triple object to compare with.

        Returns:
            bool: True if this Triple object is less than the other, False otherwise.
        """
        if not isinstance(other, Triple):
            return NotImplemented
        return (self.entity_subject, self.entity_object) < (other.entity_subject, other.entity_object)

    def __eq__(self, other: "Triple") -> bool:
        """
        Overrides the equality comparison.

        Args:
            other (Triple): The other Triple object to compare with.

        Returns:
            bool: True if this Triple object is equal to the other, False otherwise.
        """
        if not isinstance(other, Triple):
            return NotImplemented
        return (self.entity_subject == other.entity_subject and
                self.entity_object == other.entity_object and
                self.predicate == other.predicate)

    def __hash__(self) -> int:
        """
        Overrides the hash function.

        Returns:
            int: The hash value of the Triple object.
        """
        return hash((self.entity_subject, self.entity_object, self.predicate))

    def __str__(self) -> str:
        """
        Convert the Relation object to a string representation.

        Returns:
            str: The string representation of the Relation object.
        """
        source_text = getattr(self.entity_subject, "text", "") or ""
        source_id = getattr(self.entity_subject, "uid", "") or ""
        final_source_text = ""
        if source_text == source_id:
            final_source_text = f"{source_text}"
        elif source_text == "":
            final_source_text = f"{source_id}"
        elif source_id == "":
            final_source_text = f"{source_text}"
        else:
            final_source_text = f"{source_id}:{source_text}"
        target_text = getattr(self.entity_object, "text", "") or ""
        target_id = getattr(self.entity_object, "uid", "") or ""
        final_target_text = ""
        if target_text == target_id:
            final_target_text = f"{target_text}"
        elif target_text == "":
            final_target_text = f"{target_id}"
        elif target_id == "":
            final_target_text = f"{target_text}"
        else:
            final_target_text = f"{target_id}:{target_text}"
        description = self.predicate or ""
        return f"({final_source_text}, {description}, {final_target_text})".strip()

    @classmethod
    def from_string(cls, text: str) -> Union["Triple", None]:
        """
        Converts a text object to a Triple object.

        Args:
            text: The text to convert to a Triple object.

        Returns:
            Triple: The converted Triple object or None 
                if the conversion failed.         
        """

        triple_parts = get_triple_parts(text)
        if triple_parts is None:
            return None
        subject, predicate, obj = triple_parts

        return Triple(
            entity_subject=Knowledge(text=subject.strip()),
            predicate=predicate.strip(),
            entity_object=Knowledge(text=obj.strip())
        )

    @classmethod
    def convert_relations_to_dataframe(cls, relations: list["Triple"]) -> pd.DataFrame:
        """
        Converts a list of Relation objects to a pandas DataFrame.

        Args:
            relations: The list of Relation objects to convert.

        Returns:
            pd.DataFrame: The DataFrame containing the converted data.
        """
        data = []
        for relation in relations:
            source_series = relation.entity_subject.to_series()
            target_series = relation.entity_object.to_series()

            flat_dict = {
                f"source.{k}": v for k, v in source_series.items()
            }
            flat_dict.update({
                f"target.{k}": v for k, v in target_series.items()
            })
            flat_dict["description"] = relation.predicate

            data.append(flat_dict)

        return pd.DataFrame(data)

    @classmethod
    def convert_list_to_string(cls, relations: list["Triple"]) -> list[str]:
        """
        Converts a list of Relation objects to a string representation.

        Args:
            relations: The list of Relation objects to convert.

        Returns:
            list[str]: The list of string representations.
        """
        return [str(relation) for relation in relations]


def get_triple_parts(triple_string: str) -> Optional[Tuple[str, str, str]]:
    """
    Extracts the triple parts from a string representation of a triple.

    Args:
        triple_string (str): The string representation of the triple.

    Returns:
        Optional[Tuple[str, str, str]]: A tuple containing the subject, predicate, and object
            parts of the triple. Returns None if the string is not in a valid format.
    """
    triple_string = triple_string.strip()
    if not (triple_string.startswith("(") and triple_string.endswith(")")):
        return None

    inner = triple_string[1:-1].strip()

    # We first try the ORKG based triple format
    pattern = re.compile(r",\s*(?=[RLP]\d+:)")
    matches = list(pattern.finditer(inner))
    if matches:
        # The last match is the boundary between the predicate and the object
        last_match = matches[-1]
        boundary_index = last_match.start()
        # Everything after this comma (after the whitespace) is the object
        object_part = inner[last_match.end():].strip()

        # The portion before the object boundary contains the subject and predicate
        subject_predicate = inner[:boundary_index].rstrip()
        if "," not in subject_predicate:
            return None
        subject_part, predicate_part = subject_predicate.rsplit(",", 1)
    else:
        # This is the fallback to split on the comma
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != 3:
            return None
        subject_part, predicate_part, object_part = parts

    # Clean up
    subject_part = subject_part.strip().replace("'", "")
    predicate_part = predicate_part.strip().replace("'", "")
    object_part = object_part.strip().replace("'", "")

    return subject_part, predicate_part, object_part
