"""
test_hub_path_util.py
"""
import hashlib

import pytest

from core.data.models import Knowledge, Triple
from hublink.core.utils.hub_path_util import (
    PATH_SEPARATOR,
    deserialize_path,
    parse_hub_path,
    path_to_hash,
    serialize_path,
)


@pytest.fixture
def paper_entity():
    return Knowledge(
        uid="R868364",
        text="A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems",
        knowledge_types=["Paper"],
    )


@pytest.fixture
def research_field_entity():
    return Knowledge(
        uid="R659055",
        text="Software Architecture and Design",
        knowledge_types=["ResearchField"],
    )


@pytest.fixture
def venue_entity():
    return Knowledge(
        uid="R820814",
        text="International Conference on Software Architecture (ICSA)",
        knowledge_types=["Venue"],
    )


@pytest.fixture
def research_field_path_triple(paper_entity, research_field_entity):
    return Triple(
        entity_subject=paper_entity,
        predicate="research field",
        entity_object=research_field_entity,
    )


@pytest.fixture
def venue_path_triple(paper_entity, venue_entity):
    return Triple(
        entity_subject=paper_entity,
        predicate="venue",
        entity_object=venue_entity,
    )


# ---------------------------------------------------------------------------
# path_to_hash
# ---------------------------------------------------------------------------

def test_path_to_hash_matches_md5_contract(research_field_path_triple, venue_path_triple):
    path = [research_field_path_triple, venue_path_triple]

    expected = hashlib.md5(
        "".join(str(triple) for triple in path).encode()
    ).hexdigest()

    assert path_to_hash(path) == expected


def test_path_to_hash_empty_path_is_md5_of_empty_string():
    assert path_to_hash([]) == "d41d8cd98f00b204e9800998ecf8427e"

def test_path_to_hash_research_field_path_triple_hash(research_field_path_triple):
    assert path_to_hash([research_field_path_triple]) == "5c8cc6738ab64916d5cda4d0098623ee"

def test_path_to_hash_is_sensitive_to_triple_order(research_field_path_triple, venue_path_triple):
    hash_ab = path_to_hash([research_field_path_triple, venue_path_triple])
    hash_ba = path_to_hash([venue_path_triple, research_field_path_triple])

    assert hash_ab != hash_ba


def test_path_to_hash_multi_triple_equals_single_hashes_concatenated(
        research_field_path_triple, venue_path_triple,
):
    combined = path_to_hash([research_field_path_triple, venue_path_triple])
    manual = hashlib.md5(
        (str(research_field_path_triple) + str(venue_path_triple)).encode()
    ).hexdigest()

    assert combined == manual


# ---------------------------------------------------------------------------
# serialize_path / deserialize_path
# ---------------------------------------------------------------------------

def test_serialize_path_single_triple_contains_no_separator(research_field_path_triple):
    result = serialize_path([research_field_path_triple])

    assert PATH_SEPARATOR not in result
    assert result == research_field_path_triple.model_dump_json()


def test_serialize_path_multiple_triples_joined_by_separator(
        research_field_path_triple, venue_path_triple,
):
    path = [research_field_path_triple, venue_path_triple]
    result = serialize_path(path)

    assert result.count(PATH_SEPARATOR) == 1
    parts = result.split(PATH_SEPARATOR)
    assert len(parts) == 2
    assert parts[0] == research_field_path_triple.model_dump_json()
    assert parts[1] == venue_path_triple.model_dump_json()


def test_serialize_path_empty_list_returns_empty_string():
    assert serialize_path([]) == ""


def test_deserialize_serialized_path(research_field_path_triple):
    serialized = serialize_path([research_field_path_triple])
    result = deserialize_path(serialized)

    assert len(result) == 1
    assert result[0] == research_field_path_triple


def test_deserialize_serialized_path_of_multiple_triples(
        research_field_path_triple, venue_path_triple,
):
    path = [research_field_path_triple, venue_path_triple]
    serialized = serialize_path(path)

    result = deserialize_path(serialized)

    assert result == path


def test_deserialize_path_preserves_triple_field_values(research_field_path_triple):
    serialized = serialize_path([research_field_path_triple])
    result = deserialize_path(serialized)

    assert result[0].entity_subject.uid == research_field_path_triple.entity_subject.uid
    assert result[0].entity_subject.text == research_field_path_triple.entity_subject.text
    assert result[0].predicate == research_field_path_triple.predicate
    assert result[0].entity_object.uid == research_field_path_triple.entity_object.uid
    assert result[0].entity_object.text == research_field_path_triple.entity_object.text


# ---------------------------------------------------------------------------
# parse_hub_path
# ---------------------------------------------------------------------------

def test_parse_hub_path_builds_correct_hub_path(research_field_path_triple):
    path = [research_field_path_triple]
    path_hash = path_to_hash(path)
    path_as_string = serialize_path(path)
    path_text = "A Goal-Driven Approach for Deploying Self-Adaptive IoT Systems is associated with the research field of Software Architecture and Design."

    hub_path = parse_hub_path(path_hash=path_hash, path_as_string=path_as_string, path_text=path_text)

    assert hub_path.path_hash == path_hash
    assert hub_path.path_text == path_text
    assert hub_path.path == path


def test_parse_hub_path_with_multiple_triples(
        research_field_path_triple, venue_path_triple,
):
    path = [research_field_path_triple, venue_path_triple]
    path_hash = path_to_hash(path)
    path_as_string = serialize_path(path)

    hub_path = parse_hub_path(path_hash=path_hash, path_as_string=path_as_string, path_text="Information about the paper.")

    assert len(hub_path.path) == 2
    assert hub_path.path[0] == research_field_path_triple
    assert hub_path.path[1] == venue_path_triple