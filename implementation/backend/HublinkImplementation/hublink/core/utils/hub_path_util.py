import hashlib
from typing import List, Dict
from core.data.models.triple import Triple
from hublink.core.models.hub_path import HubPath

PATH_SEPARATOR = "$$$||$$$"


def path_to_hash(path: List[Triple]) -> str:
    """
    Generates a hash for a given path.

    Args:
        path (List[Triple]): The path to generate a hash for.

    Returns:
        str: The hash of the path.
    """
    path_str = ''.join([str(triple) for triple in path])
    return hashlib.md5(path_str.encode()).hexdigest()

def serialize_path(path: List[Triple]) -> str:
    return PATH_SEPARATOR.join(triple.model_dump_json() for triple in path)


def deserialize_path(path_as_string: str) -> List[Triple]:
    path_as_list = path_as_string.split(PATH_SEPARATOR)
    return [Triple.model_validate_json(t) for t in path_as_list]

def parse_hub_path(path_hash: str, path_as_string: str, path_text: str) -> HubPath:
    """
    Creates a HubPath object from its serialized components.

    Args:
        path_hash (str): The unique identifier for the path.
        path_as_string (str): The path serialized as JSON triples joined by PATH_SEPARATOR.
        path_text (str): The LLM-generated text representation of the path.

    Returns:
        HubPath: The HubPath object.
    """
    return HubPath(
        path_hash=path_hash,
        path=deserialize_path(path_as_string),
        path_text=path_text,
    )