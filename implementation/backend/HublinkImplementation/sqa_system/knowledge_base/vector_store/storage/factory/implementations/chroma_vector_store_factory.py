from typing import List
import json
from typing_extensions import override
import chromadb
from langchain_chroma import Chroma


from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.core.data.models import PublicationDataset
from sqa_system.core.data.models.context import Context
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.knowledge_base.vector_store.chunking.chunker import Chunker
from sqa_system.core.logging.logging import get_logger

from ...factory.base.vector_store_factory import VectorStoreFactory
from ...implementations.langchain_adapter import LangchainVectorStoreAdapter
logger = get_logger(__name__)


class ChromaVectorStoreFactory(VectorStoreFactory):
    """
    This is a implementation of the VectorStoreFactory that uses Chroma as the vector store.
    It is responsible for creating and managing the Chroma vector store.
    """

    def __init__(self):
        super().__init__()
        self.progress_handler = ProgressHandler()

    @classmethod
    @override
    def get_vector_store_class(cls) -> type[LangchainVectorStoreAdapter]:
        return LangchainVectorStoreAdapter

    @override
    def _create_vector_store(self,
                             publications: PublicationDataset,
                             chunker: Chunker,
                             config: VectorStoreConfig) -> LangchainVectorStoreAdapter:
        """
        Creates a Chroma vector store based on the provided publications and configuration.
        This method handles the initialization of the vector store and the embedding process.

        Args:
            publications (PublicationDataset): The dataset of publications to be used.
            chunker (Chunker): The chunker to be used for processing the publications.
            config (VectorStoreConfig): The configuration for the vector store.
        Returns:
            LangchainVectorStoreAdapter: The created vector store adapter.
        Raises:
            ValueError: If the vector store could not be created or if the collection is None.
        """
        storage_path = self._prepare_storage_path(config)
        client = self._initialize_vector_store(config)
        collection = self._initialize_collection(client, config)

        if collection is None:
            raise ValueError("Vector store could not be created")

        logger.debug("Getting embeddings adapter...")
        embeddings_adapter = LLMProvider().get_embeddings(config.embedding_config)

        vector_store = Chroma(
            client=client,
            collection_name="master_thesis",
            embedding_function=embeddings_adapter.get_embeddings(),
            collection_metadata={
                "hnsw:space": config.additional_params["distance_metric"]}
        )
        if config.force_index_rebuild:
            logger.info("Rebuilding index...")
            client.delete_collection("master_thesis")
            collection = self._initialize_collection(client, config)
            vector_store = Chroma(
                client=client,
                collection_name="master_thesis",
                embedding_function=embeddings_adapter.get_embeddings(),
                collection_metadata={
                    "hnsw:space": config.additional_params["distance_metric"]}
            )
        if collection.count() > 0:
            logger.info(
                "Vector store already contains entries. Skipping embedding process.")
            return LangchainVectorStoreAdapter(vector_store=vector_store)

        logger.debug("Starting chunking process...")
        # prepare the documents used to create the vector store
        chunker.run_chunking(publications.get_all_entries())
        chunks = chunker.get_chunks()
        self._add_documents(vector_store,
                            collection,
                            chunks,
                            storage_path)

        logger.info("Finished creating Chroma vector store")
        self._add_vector_store_information(config, storage_path)
        return LangchainVectorStoreAdapter(vector_store=vector_store)

    def _initialize_vector_store(self, config: VectorStoreConfig):
        """
        Initializes the storage path for the Chroma vector store. If at the path
        there already exists a vector store, it will be used.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.
        Returns:
            chromadb.PersistentClient: The initialized Chroma client.
        """
        storage_path = self._prepare_storage_path(config)
        logger.info("Preparing Chroma vector store at path: %s", storage_path)

        client = chromadb.PersistentClient(path=storage_path)
        return client

    def _initialize_collection(self,
                               client: chromadb.PersistentClient,
                               config: VectorStoreConfig) -> chromadb.Collection:
        """
        Initializes the collection for the Chroma vector store.
        This method creates a new collection or retrieves an existing one based on the provided configuration.

        Args:
            client (chromadb.PersistentClient): The Chroma client.
            config (VectorStoreConfig): The configuration for the vector store.
        Returns:
            chromadb.Collection: The initialized Chroma collection.
        """
        if client is None:
            raise ValueError("Chroma client is not initialized.")
        collection = client.get_or_create_collection(
            name="master_thesis",
            metadata={
                "hnsw:space": config.additional_params["distance_metric"]}
        )
        logger.info("Initialized collection: master_thesis")
        logger.info(
            f"Amount of embeddings in collection: {collection.count()}")
        return collection

    def _prepare_storage_path(self, config: VectorStoreConfig) -> str:
        """
        Prepares the storage path for the Chroma vector store.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            str: The prepared storage path for the vector store.
        """
        file_path_manager = FilePathManager()
        knowledge_base_dir_path = file_path_manager.VECTOR_STORE_DIR
        storage_path = file_path_manager.combine_paths(
            knowledge_base_dir_path, config.config_hash)
        file_path_manager.ensure_dir_exists(storage_path)
        return storage_path

    def _add_documents(self,
                       vector_store: Chroma,
                       collection,
                       contexts: List[Context],
                       storage_path: str):
        """
        Adds the documents to the vector store. This method checks if the documents
        already exist in the vector store and only adds the new ones. It also
        handles the embedding process for the new documents.

        Args:
            vector_store (Chroma): The Chroma vector store.
            collection (chromadb.Collection): The Chroma collection.
            contexts (List[Context]): The list of contexts to be added.
            storage_path (str): The storage path for the vector store.
        Raises:
            ValueError: If the vector store has not been created successfully.
        """
        logger.info("Starting to add context to vector store ...")
        logger.info("Storage path: %s", storage_path)
        logger.info("Amount of contexts: %s", len(contexts))
        logger.info("This could take a while..")

        # First we need to determine which context needs to embedded
        documents_to_embed, amount_of_duplicates = self._get_documents_to_embed(
            vector_store, contexts)
        self._fill_vector_store(vector_store, documents_to_embed)

        entries = collection.count()
        if entries < len(contexts) - amount_of_duplicates:
            logger.error(
                f"Vector store has {entries} entries but should have more than {len(contexts)}")
            raise ValueError("Vector store has not been created successfully")
        if entries != len(contexts) - amount_of_duplicates:
            logger.warning(
                f"Vector store has more entries than expected {entries}")

    def _get_documents_to_embed(self,
                                vector_store: Chroma,
                                contexts: List[Context]) -> tuple[dict, int]:
        """
        Checks if the documents already exist in the vector store. If not, it adds them
        to the list of documents to be embedded. It also checks for duplicates in the
        contexts and counts them.

        Args:
            vector_store (Chroma): The Chroma vector store.
            contexts (List[Context]): The list of contexts to be checked.
        Returns:
            Tuple[dict, int]: A tuple containing a dictionary of documents to be embedded
                and the count of duplicates found.
        """
        task_id = self.progress_handler.add_task(
            string_id="checking_vector_store",
            description="Finding contexts that need to be added..",
            total=len(contexts)
        )
        documents_to_embed = {}
        context_ids = {}
        amount_of_duplicates = 0
        for context in contexts:
            document = context.to_document()
            hash_id = document.id
            results = vector_store.get(
                ids=[hash_id],
                include=["embeddings", "metadatas"]
            )
            if not results["ids"]:
                documents_to_embed[hash_id] = document
            if hash_id in context_ids:
                logger.debug("Duplicate context found: %s", context.text)
                amount_of_duplicates += 1
            context_ids[hash_id] = ""
            self.progress_handler.update_task_by_string_id(task_id, advance=1)
        self.progress_handler.finish_by_string_id(task_id)
        return documents_to_embed, amount_of_duplicates

    def _fill_vector_store(self,
                           vector_store: Chroma,
                           documents_to_embed: dict):
        """
        Fills the vector store with the documents that need to be embedded.
        This method handles the embedding process for the new documents.

        Args:
            vector_store (Chroma): The Chroma vector store.
            documents_to_embed (dict): The dictionary of documents to be embedded.
        """
        task_id = self.progress_handler.add_task(
            string_id="filling_vector_store",
            description="Filling vector store",
            total=len(documents_to_embed)
        )
        batch_size = 100
        for i in range(0, len(documents_to_embed), batch_size):
            documents = list(documents_to_embed.values())[i:i+batch_size]
            vector_store.add_documents(
                documents=documents
            )
            self.progress_handler.update_task_by_string_id(
                task_id, advance=len(documents))
        self.progress_handler.finish_by_string_id(task_id)
        logger.info("Finished adding documents to vector store")

    def _add_vector_store_information(self, config: VectorStoreConfig, storage_path: str):
        """
        Adds a json file to the location of the vector store containing information about
        it.

        Args:
            config (VectorStoreConfig): The configuration for the vector store.
            storage_path (str): The storage path for the vector store.
        """
        data = config.model_dump_json()
        file_path_manager = FilePathManager()
        file_path_manager.ensure_dir_exists(storage_path)
        file_path = file_path_manager.combine_paths(
            storage_path, "vector_store_information.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("Added vector store information to %s", file_path)
