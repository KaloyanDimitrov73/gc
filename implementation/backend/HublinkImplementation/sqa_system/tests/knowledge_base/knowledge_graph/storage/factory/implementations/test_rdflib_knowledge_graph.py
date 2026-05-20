import pytest
from rdflib import RDF, Literal
from rdflib.namespace import RDFS
from rdflib import Namespace

from sqa_system.knowledge_base.knowledge_graph.storage.factory.implementations.local_knowledge_graph.local_knowledge_graph_factory import (
    LocalKnowledgeGraphFactory
)
from sqa_system.core.data.models.knowledge import Knowledge


# Mock Classes
class MockKnowledgeGraphConfig:
    additional_params = {
        "building_blocks": [
            "metadata",
            "authors",
            "publisher",
            "venue",
            "research_field",
            "additional_fields",
            "annotations"
        ]
    }


class MockPublication:
    def __init__(
        self,
        doi=None,
        title=None,
        authors=None,
        url=None,
        year=None,
        month=None,
        venue=None,
        abstract=None,
        research_field=None,
        keywords=None,
        additional_fields=None,
        publisher=None,
    ):
        self.doi = doi
        self.title = title
        self.authors = authors or []
        self.url = url
        self.year = year
        self.month = month
        self.venue = venue
        self.abstract = abstract
        self.research_field = research_field
        self.keywords = keywords or []
        self.additional_fields = additional_fields or {}
        self.publisher = publisher


class MockPublicationDataset:
    def __init__(self, publications=None):
        self.publications = publications or []

    def get_all_entries(self):
        return self.publications


# Fixtures
@pytest.fixture
def config():
    mock_config = MockKnowledgeGraphConfig()
    mock_config.config_hash = "test_config_hash"
    return mock_config


@pytest.fixture
def single_publication_dataset():
    publication = MockPublication(
        doi="10.1234/example.doi",
        title="Sample Publication",
        authors=["Alice Smith", "Bob Jones"],
        url="http://ressource.org/publication",
        year=2023,
        month=7,
        venue="International Conference",
        abstract="This is a sample abstract.",
        research_field="Computer Science",
        keywords=["pytest", "testing", "rdf"],
        additional_fields={"customField1": "Custom Value 1", "customField2": "Custom Value 2"},
    )
    return MockPublicationDataset([publication])


@pytest.fixture
def multiple_publications_dataset():
    pub1 = MockPublication(
        doi="10.1234/pub1",
        title="First Publication",
        authors=["Charlie Brown"],
        year=2021,
        publisher="Test publisher"
    )
    pub2 = MockPublication(
        doi="10.5678/pub2",
        title="Second Publication",
        authors=["Dana White", "Eve Black"],
        url="http://ressource.org/pub2",
        year=2022,
        abstract="Abstract for second publication.",
        keywords=["data", "science"],
        publisher="Test publisher"
    )
    return MockPublicationDataset([pub1, pub2])


@pytest.fixture
def knowledge_graph(config, single_publication_dataset):
    factory = LocalKnowledgeGraphFactory()
    graph = factory.create(config, single_publication_dataset)
    
    return graph

def test_run_sparql_query(knowledge_graph):
    """Test running a SPARQL query on the knowledge graph."""
    query = f"""
        SELECT ?label
        WHERE {{
            ?x rdfs:label ?label .
        }}
    """
    df = knowledge_graph.run_sparql_query(query)
    assert not df.empty, "SPARQL query returned empty result"
    assert "label" in df.columns, "SPARQL query should return 'label' column"

def test_get_random_entity(knowledge_graph):
    """Test getting a random entity from the knowledge graph."""
    entity = knowledge_graph.get_random_publication()
    assert entity is not None, "Random entity should not be None"

def test_get_name_from_entity_id(knowledge_graph):
    """Test getting the name from an entity ID."""
    pub_uri = "http://ressource.org/publication/10.1234/example.doi"
    name = knowledge_graph.get_entity_by_id(pub_uri)
    assert name.text == "Sample Publication", "Incorrect name retrieved from entity ID"

    non_existent_uri = "http://ressource.org/publication/non_existent"
    name = knowledge_graph.get_entity_by_id(non_existent_uri)
    assert name.text == non_existent_uri, "Should return the entity ID if name not found"


def test_is_intermediate_id(knowledge_graph):
    """Test checking if an entity ID is valid."""
    valid_uri = "http://ressource.org/publication/10.1234/example.doi"
    assert knowledge_graph.is_intermediate_id(valid_uri), "Valid entity ID should return True"

    invalid_uri = "invalid_doi"
    assert not knowledge_graph.is_intermediate_id(invalid_uri), "Invalid entity ID should return False"


def test_get_relations_of_head_entity(knowledge_graph):
    """Test getting relations where the entity is the head."""
    knowledge = Knowledge(text="Sample Publication", uid="http://ressource.org/publication/10.1234/example.doi")
    relations = knowledge_graph.get_relations_of_head_entity(knowledge)

    # The format is key = relation description and value = tail id
    expected_relations = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "http://schema.org/ScholarlyArticle",
        "http://www.w3.org/2000/01/rdf-schema#label": "Sample Publication",
        "http://predicate.org/doi": "10.1234/example.doi",
        "http://schema.org/datePublished": "2023-7",
        "http://schema.org/keywords": "pytest",
    }
    
    assert len(relations) >= len(expected_relations), "Unexpected number of relations"

    for key, value in expected_relations.items():
        is_contained = False
        for relation in relations:
            if relation.predicate == key and relation.entity_object.uid == value:
                is_contained = True
                break
        assert is_contained, "Not all expected relations were found"

def test_get_relations_of_tail_entity(knowledge_graph):
    """Test getting relations where the entity is the tail."""
    knowledge = Knowledge(text="", uid="http://ressource.org/author_0")
    relations = knowledge_graph.get_relations_of_tail_entity(knowledge)

    # The format is: key = relation description and value = head id
    expected_relations = {
        "entry": "http://ressource.org/authors_0"
    }

    assert len(relations) >= len(expected_relations), "Unexpected number of relations"

    for key, value in expected_relations.items():
        is_contained = False
        for relation in relations:
            if key in relation.predicate.lower() and relation.entity_subject.uid == value:
                is_contained = True
                break
        assert is_contained, "Not all expected relations were found"
   

def test_check_connection(knowledge_graph):
    """Test the check_connection method."""
    assert knowledge_graph.validate_graph_connection(), "check_connection should return True"

def test_get_name_from_entity_id_with_labels(knowledge_graph):
    """Test getting names from entity IDs with and without labels."""
    
    RE = Namespace("http://ressource.org/")
    # Add an entity with rdfs:label
    labeled_entity = RE["labeled_entity"]
    labeled_name = "Labeled Entity"
    knowledge_graph.graph.add((labeled_entity, RDFS.label, Literal(labeled_name)))

    name = knowledge_graph.get_entity_by_id(str(labeled_entity))
    assert name.text == labeled_name, "Should retrieve the label of the entity"

    # Entity without label should return its UID
    unlabeled_entity = RE["unlabeled_entity"]
    knowledge_graph.graph.add((unlabeled_entity, RDF.type, RE.SomeType))
    name = knowledge_graph.get_entity_by_id(str(unlabeled_entity))
    assert name.text == str(unlabeled_entity), "Should return the UID if label is not found"


if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
