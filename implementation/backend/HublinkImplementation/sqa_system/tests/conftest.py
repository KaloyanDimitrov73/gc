
"""
This file includes configurations for testing purposes.
"""
import pytest
from knowledge_base.config.knowledge_graph_config import KnowledgeGraphConfig
from core import DatasetConfig
from language_model.config.llm_config import LLMConfig

@pytest.fixture
def dataset_config():
    return DatasetConfig(
        name="test_dataset",
        additional_params={},
        file_name="ECSA-ICSA-Proceedings.bib",
        loader="BibtextPublicationLoader",
        loader_limit=-1
    )

@pytest.fixture
def knowledge_graph_config(dataset_config):
    return KnowledgeGraphConfig(
        name="test_graph",
        additional_params={},
        graph_type="local_rdflib",
        dataset_config=dataset_config
    )

@pytest.fixture
def llm_config():
    return LLMConfig(
        name="test_llm",
        additional_params={},
        endpoint="OpenAI",
        name_model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=-1,
    )
