import pytest
import shutil
import os
from sqa_system.knowledge_base.vector_store.storage.factory.implementations.chroma_vector_store_factory import ChromaVectorStoreFactory
from sqa_system.core.data.models.dataset.implementations.publication_dataset import PublicationDataset
from sqa_system.core.data.models.publication import Publication
from sqa_system.knowledge_base.vector_store.chunking.chunker import Chunker
from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.core.config.models import EmbeddingConfig


@pytest.fixture
def embedding_test_config():
    return EmbeddingConfig(
        name="test_embedding",
        additional_params={},
        endpoint="OpenAI",
        name_model="text-embedding-3-small",
    )

@pytest.fixture
def test_publications() -> PublicationDataset:
    publications = {
        "10.1234/test1": Publication(
            title="Test Publication 1",
            doi="10.1234/test1",
            abstract="This is a test abstract about software architecture.",
            authors=["Author1", "Author2"],
            year="2023",
            venue="TestConf",
            url="http://test1.com",
            publisher="TestPublisher1"
        ),
        "10.1234/test2": Publication(
            title="Test Publication 2",
            doi="10.1234/test2",
            abstract="This is another test abstract about testing methodology.",
            authors=["Author3"],
            year="2023",
            venue="TestConf",
            url="http://test2.com",
            publisher="TestPublisher2"
        )
    }
    return PublicationDataset("", publications)

@pytest.fixture
def chunking_config():
    return ChunkingStrategyConfig(
        name="test_chunking",
        additional_params={},
        chunking_strategy_type="RecursiveCharacterChunkingStrategy",
        chunk_size=100,
        chunk_overlap=20
    )

@pytest.fixture
def vector_store_config(chunking_config, embedding_test_config, dataset_config):
    return VectorStoreConfig(
        vector_store_type="chroma",
        chunking_strategy_config=chunking_config,
        embedding_config=embedding_test_config,
        dataset_config=dataset_config,
        additional_params={"distance_metric": "cosine"}
    )

@pytest.fixture
def chroma_adapter(test_publications, vector_store_config, tmp_path):

    # Set up temporary directory for Chroma    
    factory = ChromaVectorStoreFactory()
    path = factory._prepare_storage_path(vector_store_config)
    # delete the directory if it already exists
    if os.path.exists(path):
        shutil.rmtree(path)
    chunker = Chunker(vector_store_config.chunking_strategy_config)
    
    adapter = factory._create_vector_store(
        publications=test_publications,
        chunker=chunker,
        config=vector_store_config
    )
    
    yield adapter

def test_query(chroma_adapter):
    """Test querying the Chroma vector store."""
    results = chroma_adapter.query("software architecture", 2)
    
    assert len(results) > 0
    assert "software architecture" in results[0].text.lower()
    
    results = chroma_adapter.query_with_metadata_filter(
        query_text="test",
        n_results=2,
        metadata_filter={"publisher": "TestPublisher1"}
    )
    
    assert len(results) > 0
    assert all(result.metadata.get("publisher") == "TestPublisher1" for result in results)

if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
