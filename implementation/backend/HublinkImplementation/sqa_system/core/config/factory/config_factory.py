from enum import Enum
from typing import List, Optional

from sqa_system.retrieval import KnowledgeGraphRetrieverFactory
from sqa_system.retrieval import DocumentRetrieverFactory
from sqa_system.core.logging.logging import get_logger
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraphFactoryRegistry
from sqa_system.knowledge_base.vector_store.storage import VectorStoreFactoryRegistry
from ..models import KnowledgeGraphConfig
from ..models import KGRetrievalConfig
from ..models import DocumentRetrievalConfig
from ..models import ChunkingStrategyConfig
from ..models import DatasetConfig
from ..models import EmbeddingConfig
from ..models import VectorStoreConfig
from ..models import PipelineConfig
from ..models import PipeConfig
from ..models import GenerationConfig
from ..models import LLMConfig
from ..models import PostRetrievalConfig
from ..models import PreRetrievalConfig

logger = get_logger(__name__)


class PipeConfigs(Enum):
    """Enum class for all possible pipe types."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


class ConfigFactory:
    """Factory class that creates config objects."""
    
    @staticmethod
    def create_post_restrieval_config(technique: str,
                                      name: Optional[str] = None,
                                      llm_config: Optional[LLMConfig] = None,
                                      **kwargs) -> PostRetrievalConfig:
        """
        Creates a PostRetrievalProcessingConfig object with the provided parameters.

        Args:
            technique (str): The post retrieval processing technique.
            name (Optional[str], optional): The name of the configuration. Defaults to None.
            llm_config (Optional[LLMConfig], optional): The LLMConfig object. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            PostRetrievalProcessingConfig: A PostRetrievalProcessingConfig object with the 
                provided parameters.
        """
        if name is not None and name != "":
            return PostRetrievalConfig(post_technique=technique,
                                       name=name,
                                       llm_config=llm_config,
                                       **kwargs)
        return PostRetrievalConfig(post_technique=technique,
                                   llm_config=llm_config,
                                   **kwargs)
    
    @staticmethod
    def create_pre_retrieval_config(technique: str,
                                    name: Optional[str] = None,
                                    llm_config: Optional[LLMConfig] = None,
                                    **kwargs) -> PreRetrievalConfig:
        """
        Creates a PreRetrievalProcessingConfig object with the provided parameters.

        Args:
            technique (str): The pre retrieval processing technique.
            name (Optional[str], optional): The name of the configuration. Defaults to None.
            llm_config (Optional[LLMConfig], optional): The LLMConfig object. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            PreRetrievalProcessingConfig: A PreRetrievalProcessingConfig object with the 
                provided parameters.
        """
        if name is not None and name != "":
            return PreRetrievalConfig(pre_technique=technique,
                                      name=name,
                                      llm_config=llm_config,
                                      **kwargs)
        return PreRetrievalConfig(pre_technique=technique,
                                    llm_config=llm_config,
                                    **kwargs)

    @staticmethod
    def create_dataset_config(file_name: str,
                              loader: str,
                              loader_limit: Optional[int] = None) -> DatasetConfig:
        """
        Creates a DatasetConfig object with the provided parameters.

        Args:
            file_name (str): The file name of the dataset.
            loader (str): The loader type for the dataset.
            loader_limit (Optional[int], optional): The limit for the loader. Defaults to None.

        Returns:
            DatasetConfig: A DatasetConfig object with the provided parameters.
        """
        return DatasetConfig(file_name=file_name, loader=loader, loader_limit=loader_limit)

    @staticmethod
    def create_pipeline_config(pipes: List[PipeConfig],
                               name: Optional[str] = None) -> PipelineConfig:
        """
        Creates a pipeline configuration object with the given list of pipe configurations.

        Args:
            pipes (List[PipeConfig]): A list of pipe configurations.
            name (Optional[str], optional): The name of the pipeline.
                If not provided, a name will be automatically 
                generated.

        Returns:
            PipelineConfig: The created pipeline configuration object.
        """
        specific_pipes = []
        for pipe in pipes:
            if isinstance(pipe, (GenerationConfig, DocumentRetrievalConfig, KGRetrievalConfig)):
                specific_pipes.append(pipe)
            elif isinstance(pipe, PipeConfig):
                if pipe.type == "generation":
                    specific_pipes.append(
                        GenerationConfig(**pipe.model_dump()))
                elif pipe.type == "kg_retrieval":
                    specific_pipes.append(
                        KGRetrievalConfig(**pipe.model_dump()))
                elif pipe.type == "document_retrieval":
                    specific_pipes.append(
                        DocumentRetrievalConfig(**pipe.model_dump()))
                elif pipe.type == "post_retrieval_processing":
                    specific_pipes.append(
                        PostRetrievalConfig(**pipe.model_dump()))
                elif pipe.type == "pre_retrieval_processing":
                    specific_pipes.append(
                        PreRetrievalConfig(**pipe.model_dump()))
                else:
                    raise ValueError(f"Unknown pipe type: {pipe.type}")
            else:
                raise ValueError(f"Invalid pipe configuration: {pipe}")
        if name is not None and name != "":
            name = PipelineConfig.prepare_name_for_config(name)
            return PipelineConfig(pipes=specific_pipes,
                                  name=name)
        return PipelineConfig(pipes=specific_pipes)

    @staticmethod
    def create_generation_config(llm_config: LLMConfig) -> GenerationConfig:
        """
        Creates a GenerationConfig object with the given llm_config.

        Args:
            llm_config (LLMConfig): The LLMConfig object to be used by the GenerationConfig.

        Returns:
            GenerationConfig: The newly created GenerationConfig object.
        """
        return GenerationConfig(llm_config=llm_config)

    @staticmethod
    def create_llm_config(endpoint: str,
                          name_model: str,
                          temperature: float,
                          max_tokens: int) -> LLMConfig:
        """
        Creates an LLMConfig object with the provided parameters.

        Args:
            endpoint (str): The endpoint of the LLMConfig object.
            name_model (str): The name model of the LLMConfig object.
            temperature (float): The temperature of the LLMConfig object.
            max_tokens (int): The maximum tokens of the LLMConfig object.

        Returns:
            LLMConfig: The created LLMConfig object.
        """
        return LLMConfig(endpoint=endpoint,
                         name_model=name_model,
                         temperature=temperature,
                         max_tokens=max_tokens)

    @staticmethod
    def create_vector_store_config(vector_store_type: str,
                                   dataset_config: DatasetConfig,
                                   chunking_strategy_config: ChunkingStrategyConfig,
                                   embedding_config: EmbeddingConfig,
                                   **kwargs) -> VectorStoreConfig:
        """
        Creates a VectorStoreConfig object with the specified parameters.

        Args:
            vector_store_type (str): The type of the vector store.
            dataset_config (DatasetConfig): The dataset configuration.
            chunking_strategy_config (ChunkingStrategyConfig): The chunking strategy configuration.
            embedding_config (EmbeddingConfig): The embedding configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStoreConfig: The created VectorStoreConfig object.
        """
        store_class = VectorStoreFactoryRegistry().get_factory_class(vector_store_type)
        return store_class.create_config(vector_store_type=vector_store_type,
                                         chunking_strategy=chunking_strategy_config,
                                         embedding_config=embedding_config,
                                         dataset_config=dataset_config,
                                         **kwargs)

    @staticmethod
    def create_chunking_strategy_config(chunking_strategy_type: str,
                                        chunk_size: int,
                                        chunk_overlap: int) -> ChunkingStrategyConfig:
        """
        Creates a ChunkingStrategyConfig object with the specified parameters.

        Args:
            chunking_strategy_type (str): The type of the chunking strategy.
            chunk_size (int): The size of the chunks.
            chunk_overlap (int): The overlap of the chunks.

        Returns:
            ChunkingStrategyConfig: The created ChunkingStrategyConfig object.
        """
        return ChunkingStrategyConfig(chunking_strategy_type=chunking_strategy_type,
                                      chunk_size=chunk_size,
                                      chunk_overlap=chunk_overlap)

    @staticmethod
    def create_embedding_config(endpoint: str,
                                name_model: str) -> EmbeddingConfig:
        """
        Creates an EmbeddingConfig object with the specified parameters.

        Args:
            endpoint (str): The endpoint of the embedding model.
            name_model (str): The name of the embedding model.

        Returns:
            EmbeddingConfig: The created EmbeddingConfig object.
        """
        return EmbeddingConfig(endpoint=endpoint, name_model=name_model)

    @staticmethod
    def create_knowledge_graph_config(
            graph_type: str,
            dataset_config: DatasetConfig, 
            extraction_llm_config: LLMConfig,
            **kwargs) -> KnowledgeGraphConfig:
        """
        Creates a KnowledgeGraphConfig object with the specified parameters.

        Args:
            graph_type (str): The type of the knowledge graph.
            dataset_config (DatasetConfig): The dataset configuration used for
                creating or populating the graph.
            extraction_llm_config (LLMConfig): The LLMConfig object used for
                extracting information from the dataset to populate the graph.
            **kwargs: Additional keyword arguments.

        Returns:
            KnowledgeGraphConfig: The created KnowledgeGraphConfig object.
        """
        kg_factory_class = KnowledgeGraphFactoryRegistry().get_factory_class(graph_type)
        if kg_factory_class is None:
            raise ValueError(
                f"Knowledge graph type {graph_type} is not supported.")
        return kg_factory_class.create_config(graph_type=graph_type,
                                              dataset_config=dataset_config,
                                                extraction_llm_config=extraction_llm_config,
                                              **kwargs)

    @staticmethod
    def create_kg_retrieval_config(retriever_type: str,
                                   name: Optional[str] = None,
                                   **kwargs) -> KGRetrievalConfig:
        """
        Creates a KGRetrievalConfig object with the specified parameters.

        Args:
            retriever_type (str): The type of the knowledge graph retriever.
            name (Optional[str], optional): The name of the configuration. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            KGRetrievalConfig: The created KGRetrievalConfig object.
        """
        retriever_class = KnowledgeGraphRetrieverFactory.get_retriever_class(
            retriever_type)
        if retriever_class is None:
            raise ValueError(
                f"Retriever type {retriever_type} is not supported.")
        if name is not None and name != "":
            return retriever_class.create_config(retriever_type=retriever_type,
                                                 name=name,
                                                 **kwargs)
        return retriever_class.create_config(retriever_type=retriever_type,
                                             **kwargs)

    @staticmethod
    def create_doc_retrieval_config(retriever_type: str,
                                    name: Optional[str] = None,
                                    **kwargs) -> DocumentRetrievalConfig:
        """
        Creates a DocumentRetrievalConfig object with the specified parameters.

        Args:
            retriever_type (str): The type of the knowledge graph retriever.
            name (Optional[str], optional): The name of the configuration. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            DocumentRetrievalConfig: The created DocumentRetrievalConfig object.
        """
        retriever_class = DocumentRetrieverFactory.get_retriever_class(
            retriever_type)
        if retriever_class is None:
            raise ValueError(
                f"Retriever type {retriever_type} is not supported.")
        if name is not None and name != "":
            return retriever_class.create_config(retriever_type=retriever_type,
                                                 name=name,
                                                 **kwargs)
        return retriever_class.create_config(retriever_type=retriever_type,
                                             **kwargs)

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Replaces spaces with underscores which is helpful when using the name in file
        paths.
        
        Args:
            name (str): The name to be normalized.
            
        Returns:
            str: The normalized name with spaces replaced by underscores.
        """
        name = name.replace(" ", "_")
        return name
