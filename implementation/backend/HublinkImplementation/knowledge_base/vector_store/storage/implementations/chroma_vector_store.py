"""
storage/chroma/chroma_vector_store.py

Native Chroma adapter implementing the VectorStore interface.
Contains only generic vector storage logic — no Hub/Triple domain knowledge.
"""

from threading import RLock
from typing import override, List, Optional, Dict

import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.data.file_path_manager import FilePathManager
from core.logging.logging import get_logger
from knowledge_base.vector_store.storage.vector_store import VectorStore, VectorScoreResults

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """Vector store backed directly by ChromaDB (no LangChain)."""

    # ChromaDB's default max batch size is 41 666 items.
    _CHROMA_MAX_BATCH_SIZE = 40_000

    def __init__(self,
                 store_name: str,
                 collection_name: str = "novel_retriever",
                 distance_metric: str = "cosine",
                 store_path: Optional[str] = None):
        self.store_name = store_name
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._lock = RLock()
        self.client = None
        self.collection = None
        self.store_path = store_path or self._default_store_path()
        self._initialize()

    def _default_store_path(self) -> str:
        file_path_manager = FilePathManager()
        return file_path_manager.combine_paths(
            file_path_manager.CACHE_DIR, "hublink_retriever", self.store_name
        )

    @override
    def _initialize(self) -> None:
        with self._lock:
            if self.client is None:
                self.client = self._initialize_client()
            if self.collection is None:
                self.collection = self._initialize_collection()

    def _initialize_client(self) -> chromadb.ClientAPI:
        """
        Initializes the Chroma client with the specified persist directory.

        Returns:
            chromadb.Client: Configured Chroma client instance.
        """
        file_path_manager = FilePathManager()
        file_path_manager.ensure_dir_exists(self.store_path)
        logger.info("Loading Chroma store from: %s", self.store_path)
        return chromadb.PersistentClient(path=self.store_path)

    def _initialize_collection(self) -> chromadb.Collection:
        """
        Retrieves or creates the specified Chroma collection.

        Returns:
            chromadb.Collection: The Chroma collection instance.
        """
        if self.client is None:
            raise ValueError("Chroma client is not initialized.")

        # We encountered an error with chromadb that only happend after we scaled the dataset to its fullest.
        # The error that semingly randomly appeared was 'Cannot return the results in a contigious 2D array.
        # Probably ef or M is too small'. It appeared only at query time, indexing was fine.
        # The error is badly documented but the ressources suggest, that the issue appears, when the query
        # cant return the amount of results that are asked for. Therefore, we increase the parameters
        # here to reduce the chances of it happening. https://docs.trychroma.com/docs/collections/configure
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": self.distance_metric,
                "hnsw:search_ef": 200,
                "hnsw:construction_ef": 200,
                "hnsw:M": 32
            }
        )
        logger.info(f"Initialized collection: {self.collection_name}")
        logger.info(
            f"Amount of embeddings in collection: {collection.count()}")

        return collection

    @override
    def store_data(self, record_id: str, embedding: List[float], metadata: Optional[Dict] = None):
        """
        Stores an embedding in the Chroma collection with the given hash_key as its ID.

        Args:
            record_id (str): Unique identifier for the embedding.
            embedding (List[float]): The embedding vector.
            metadata (dict, optional): Additional metadata associated with the embedding.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                self.collection.upsert(
                    embeddings=[embedding],
                    metadatas=[metadata] if metadata else None,
                    ids=[record_id]
                )
                logger.debug(f"Upserted embedding with ID: {record_id}")
            except Exception as e:
                logger.error(
                    f"Failed to upsert embedding with ID {record_id}: {e}")
                raise

    @override
    def store_data_batch(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict]) -> None:
        """
        Stores multiple embeddings in the Chroma collection in chunked upsert calls.

        ChromaDB has an internal max batch size (~41 666). Very large hubs can exceed
        this limit, so we split the data into safe-sized chunks before upserting.

        Args:
            ids (List[str]): Unique identifiers for each embedding.
            embeddings (List[List[float]]): The embedding vectors.
            metadatas (List[Dict]): Metadata dicts for each embedding.
        """
        if not ids:
            return
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        chunk_size = self._CHROMA_MAX_BATCH_SIZE
        with self._lock:
            try:
                for start in range(0, len(ids), chunk_size):
                    end = start + chunk_size
                    self.collection.upsert(
                        embeddings=embeddings[start:end],
                        metadatas=metadatas[start:end],
                        ids=ids[start:end]
                    )
                logger.debug("Batch upserted %d embeddings.", len(ids))
            except Exception as e:
                logger.error(
                    "Failed to batch upsert %d embeddings: %s", len(ids), e)
                raise

    def delete_records_with_filter(self, where_filter: Dict) -> None:
        """
        Deletes all embeddings that meet a specific filter expression.

        Args:
            where_filter (str): The filtering expression.
        """
        with self._lock:
            try:
                self.collection.delete(where=where_filter)
                logger.debug(
                    f"Deleted embeddings for filter: {where_filter}")
            except Exception as e:
                logger.error(
                    f"Failed to delete embeddings for filter: {where_filter}: {e}")
                raise

    def get_records_with_metadata_by_filter(self, where_filter: Dict, limit: Optional[int] = 1) -> Optional[VectorScoreResults]:
        """
        Returns Records of the Vector store that meet the given filter expression.

        Args:
            where_filter (str): The filtering expression.
            limit: number of retrieval results

        Returns:
            VectorScoreResults: The matching entries, or None if nothing matched.
        """

        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                results = self.collection.get(
                    where=where_filter,
                    include=["embeddings", "metadatas"],
                    limit=limit,
                )
            except Exception as e:
                logger.error("Failed to get records with filter %s: %s", where_filter, e)
                raise

        if not results["ids"]:
            return None

        return VectorScoreResults(
            ids=results["ids"],
            metadata=results.get("metadatas"),
            embeddings=results.get("embeddings"),
        )

    def get_records_with_metadata_by_ids(self, ids: List[str]) -> Optional[VectorScoreResults]:
        """
        Retrieves a single entry by its unique id.

        Args:
            ids (str): The unique identifier of the entry to retrieve.

        Returns:
            VectorScoreResults: The matching entry, or None if no entry with this id exists.
        """

        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                result = self.collection.get(
                    ids=ids, include=["embeddings", "metadatas"]
                )
            except Exception as e:
                logger.error("Failed to get record by id %s: %s", id, e)
                raise

        if not result["ids"]:
            return None


        return VectorScoreResults(
            ids=result["ids"],
            metadata=result.get("metadatas"),
            embeddings=result.get("embeddings")
        )

    def vector_similarity_search(self, query_embeddings: List[List[float]],
                                  where_filter: Optional[Dict] = None,
                                  n_results: int = 10) -> List[VectorScoreResults]:

        """
                Performs a similarity search, returning the entries most similar to the
                given query embedding(s), optionally restricted by a metadata filter.

                Args:
                    query_embeddings (List[List[float]]): One or more query embedding vectors.
                    where_filter (Dict, optional): Metadata filter condition to restrict the search.
                    n_results (int): Maximum number of similar entries to return.

                Returns:
                    VectorScoreResults: The most similar entries, including their similarity
                        distances, ordered by similarity.
                """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                result = self.collection.query(
                    query_embeddings=query_embeddings,
                    where=where_filter,
                    n_results=n_results,
                    include=["embeddings", "metadatas", "distances"],
                )
                # Flatten the per-query nesting; assumes single-query use for now.

                results = [
                    VectorScoreResults(
                        ids=result["ids"][k],
                        embeddings=result.get("embeddings"),
                        metadata=result["metadatas"][k] if result.get("metadatas") else [],
                        distances=result["distances"][k] if result.get("distances") else None
                    )
                    for k in range(len(result["ids"]))
                ]

                return results

            except Exception as e:
                if "cannot return the results" in str(e).lower():
                    logger.warning(
                        "Nearest Neighbor Query Failed trying manually..")
                    logger.debug(
                        f"Parameters: {where_filter}, {n_results}")
                    # This happens if the n_results is too large for the
                    # number of embeddings in the hub
                    return self._manually_calculate_similarity_score(
                        where_filter=where_filter,
                        query_embeddings=query_embeddings,
                        n_results=n_results
                    )
                logger.error(
                    f"Failed to perform similarity search for {where_filter}: {e}")
                raise

    def count(self) -> int:
        return self.collection.count()

    def _manually_calculate_similarity_score(self,
                                             query_embeddings: List[List[float]],
                                             where_filter: Optional[Dict] = None,
                                             n_results: int = 10) -> List[VectorScoreResults]:
        """
        We encountered an error with chromadb:
        "Cannot return the results in a contigious 2D array. Probably ef or M is too small"
        The error should no longer appear as whe increased the parameters, but in case it
        still does, this is a fallback method.

        This method is a workaround to manually calculate the similarity score
        between the query embeddings and the embeddings in the collection.

        Args:
            filter (str): The filter expression.
            query_embeddings (List[List[float]]): The embedding vector for which to find similar
                embeddings.
            n_results (int): The number of top similar embeddings to retrieve.

        Returns:
            List[VectorScoreResults]: A list of VectorScoreResults objects sorted by their similarity score.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")
        with self._lock:
            try:
                records = self.collection.get(
                    where=where_filter,
                    include=["embeddings", "metadatas"]
                )
                ids = records.get("ids", [])
                if not ids:
                    logger.debug(f"No records for {where_filter}")
                    return []

                logger.debug(
                    f"Found {len(ids)} records for {where_filter}"
                )
                embeddings_array = np.array(records["embeddings"])  # shape (N, D)

            except Exception as e:
                logger.error(
                    f"Failed to fetch embeddings for {where_filter}: {e}"
                )
                raise

        if not query_embeddings:
            logger.debug("No query embeddings provided; returning empty result list.")
            return []

        max_similarities = self._calculate_max_similarities(
            query_embeddings=query_embeddings,
            embeddings_array=embeddings_array
        )

        results: List[VectorScoreResults] = []

        for query_sims in max_similarities:
            # pair each id/metadata with its similarity score for this query
            scored = list(zip(ids, records["metadatas"], query_sims))
            scored.sort(key=lambda entry: entry[2], reverse=True)
            scored = scored[:n_results]

            results.append(VectorScoreResults(
                ids=[entry[0] for entry in scored],
                metadata=[entry[1] for entry in scored],
                distances=[1 - float(entry[2]) for entry in scored],
            ))

        return results


    def _calculate_max_similarities(self, query_embeddings: List[List[float]],
                                    embeddings_array: np.ndarray) -> np.ndarray:
        """
        Helper function to calculate the maximum similarities between query embeddings
        and the embeddings in the collection.

        Args:
            query_embeddings (List[List[float]]): The embedding vector for which to find similar
                embeddings.
            embeddings_array (np.ndarray): The array of embeddings in the collection.

        Returns:
            np.ndarray: An array of maximum similarities for each query embedding.
        """
        queries = np.asarray(query_embeddings, dtype=np.float32)
        emb     = embeddings_array.astype(np.float32, copy=False)
        return cosine_similarity(queries, emb, dense_output=True)
