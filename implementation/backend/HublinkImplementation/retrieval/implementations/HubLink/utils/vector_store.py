from typing import List, Optional, Dict, Tuple
from threading import RLock
import chromadb
from chromadb import QueryResult
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.data.file_path_manager import FilePathManager
from core.logging.logging import get_logger
from core.data.models.triple import Triple

from ..models.hub_path import HubPath

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Vector store for caching and retrieving HubPaths using ChromaDB.

    This class provides methods to store, delete, and search for hub paths and their embeddings.

    Args:
        store_name (str): Name of the Chroma store (used for persistence).
        diversity_penalty (float): Penalty for repeated subjects in diversity ranking.
        collection_name (str): Name of the collection in the store.
        distance_metric (str): Distance metric for similarity search (default: "cosine").
    """

    def __init__(self,
                 store_name: str,
                 diversity_penalty: float,
                 collection_name: str = "novel_retriever",
                 distance_metric: str = "cosine"):
        """
        Initializes the ChromaVectorStore by setting up the client and collection.

        Args:
            store_name (str): Name of the Chroma store which is also used for the
                persist directory. If it already exists as a file, the vector store
                will be loaded from the file
            collection_name (str): Name of the Chroma collection inside of the vector
                store
            distance_metric (str): The distance metric to use for the collection.
            diversity_penalty (float): The penalty value to use for the diversity ranker

        """
        self.store_name = store_name
        self.diversity_penalty = diversity_penalty
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._lock = RLock()
        self.client = None
        self.collection = None
        self.store_path = None
        self._initialize()

    def _initialize(self):
        with self._lock:
            if self.client is None:
                self.client = self._initialize_client()
            if self.collection is None:
                self.collection = self._initialize_collection()

    def _initialize_client(self):
        """
        Initializes the Chroma client with the specified persist directory.

        Returns:
            chromadb.Client: Configured Chroma client instance.
        """
        file_path_manager = FilePathManager()
        novel_retriever_cache_dir = file_path_manager.combine_paths(
            file_path_manager.CACHE_DIR, "hublink_retriever", self.store_name
        )

        # Ensure the persist directory exists
        file_path_manager.ensure_dir_exists(novel_retriever_cache_dir)
        logger.info(
            f"Loading chroma vector store from: {novel_retriever_cache_dir}")

        client = chromadb.PersistentClient(path=novel_retriever_cache_dir)
        self.store_path = novel_retriever_cache_dir
        return client

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

    def store_data(self,
                   hash_key: str,
                   embedding: List[float],
                   metadata: Optional[Dict] = None):
        """
        Stores an embedding in the Chroma collection with the given hash_key as its ID.

        Args:
            hash_key (str): Unique identifier for the embedding.
            embedding (List[float]): The embedding vector.
            metadata (dict, optional): Additional metadata associated with the embedding.
        """
        if metadata is None:
            metadata = {}
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                self.collection.upsert(
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[hash_key]
                )
                logger.debug(f"Upserted embedding with ID: {hash_key}")
            except Exception as e:
                logger.error(
                    f"Failed to upsert embedding with ID {hash_key}: {e}")
                raise

    # ChromaDB's default max batch size is 41 666 items.
    _CHROMA_MAX_BATCH_SIZE = 40_000

    def store_data_batch(self,
                        hash_keys: List[str],
                        embeddings: List[List[float]],
                        metadatas: List[Dict]):
        """
        Stores multiple embeddings in the Chroma collection in chunked upsert calls.

        ChromaDB has an internal max batch size (~41 666). Very large hubs can exceed
        this limit, so we split the data into safe-sized chunks before upserting.

        Args:
            hash_keys (List[str]): Unique identifiers for each embedding.
            embeddings (List[List[float]]): The embedding vectors.
            metadatas (List[Dict]): Metadata dicts for each embedding.
        """
        if not hash_keys:
            return
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        chunk_size = self._CHROMA_MAX_BATCH_SIZE
        with self._lock:
            try:
                for start in range(0, len(hash_keys), chunk_size):
                    end = start + chunk_size
                    self.collection.upsert(
                        embeddings=embeddings[start:end],
                        metadatas=metadatas[start:end],
                        ids=hash_keys[start:end]
                    )
                logger.debug("Batch upserted %d embeddings.", len(hash_keys))
            except Exception as e:
                logger.error(
                    "Failed to batch upsert %d embeddings: %s", len(hash_keys), e)
                raise

    def delete_data_from_hub(self, hub_entity_id: str):
        """
        Deletes all embeddings associated with a specific hub_entity.

        Args:
            hub_entity_id (str): The unique identifier of the hub entity.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")
        with self._lock:
            try:
                self.collection.delete(where={"hub_entity": hub_entity_id})
                logger.debug(
                    f"Deleted embeddings for hub_entity {hub_entity_id}")
            except Exception as e:
                logger.error(
                    f"Failed to delete embeddings for hub_entity {hub_entity_id}: {e}")
                raise

    def retrieve_one_hub_path(self, path_hash: str) -> HubPath | None:
        """
        Returns one HubPath that contains the given path_hash as a metadata field.
        This is useful because a path is stored multiple times in the store on different
        levels: path text, triple, and entity level.

        With this function we can retrieve the first match of any of the three levels.

        Args:
            path_hash (str): The unique identifier for the path.

        Returns:
            List[HubPath]: A list of HubPath objects.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                results = self.collection.get(
                    where={"path_hash": path_hash},
                    include=["metadatas"],
                    limit=1
                )

                if not results["ids"]:
                    return None

                i = len(results["ids"][0])
                metadata = results["metadatas"][0][i]

                return self._parse_hub_path(
                    path_hash=path_hash,
                    metadata=metadata
                )

            except Exception as e:
                logger.error(
                    f"Failed to retrieve paths with path_hash {path_hash}: {e}")
                raise

    def retrieve_hub_path_by_key(self, hash_key: str) -> Optional[HubPath]:
        """
        Retrieves a HubPath from the Chroma collection based on the hash_key.
        The hash_key is unique to the embedding which means that only a single
        entry can be found if the hash_key is contained.

        Args:
            hash_key (str): The unique identifier for the embedding.

        Returns:
            HubPath: The HubPath object if found, else None.
        """
        results = self.retrieve_hub_paths_by_keys([hash_key])
        return results.get(hash_key)

    def retrieve_hub_paths_by_keys(self, hash_keys: List[str]) -> Dict[str, Optional[HubPath]]:
        """
        Retrieves multiple HubPaths from the Chroma collection in a single batch get call.

        This is significantly faster than calling retrieve_hub_path_by_key() in a loop
        because it replaces N individual DB round-trips with one.

        Args:
            hash_keys (List[str]): The unique identifiers for the embeddings to retrieve.

        Returns:
            Dict[str, Optional[HubPath]]: Mapping of hash_key -> HubPath (or None if not found).
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        result: Dict[str, Optional[HubPath]] = {key: None for key in hash_keys}

        with self._lock:
            try:
                results = self.collection.get(
                    ids=hash_keys,
                    include=["metadatas"]
                )
            except Exception as e:
                logger.error("Failed to batch retrieve HubPaths: %s", e)
                raise

        for record_id, metadata in zip(results["ids"], results["metadatas"]):
            result[record_id] = self._parse_hub_path(
                path_hash=record_id, metadata=metadata)

        return result

    def get_all_hub_paths_from_hub(self, hub_entity_id: str) -> Tuple[List[HubPath], List[List[float]]]:
        """
        Retrieves all HubPaths associated with a specific hub_entity.
        This method does not require a question and therefore it does not provide
        a similarity score. It is used to retrieve all paths that are stored
        in the vector store for a specific hub entity.

        Args:
            hub_entity_id (str): The unique identifier of the hub entity.

        Returns:
            List[HubPath]: A list of HubPath objects.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")
        with self._lock:
            try:
                results = self.collection.get(
                    where={"hub_entity": hub_entity_id},
                    include=["embeddings", "metadatas"]
                )
                if not results.get("ids") or not results["ids"]:
                    return [], []

                hub_paths = []
                for idx, record_id in enumerate(results["ids"]):
                    hub_paths.append(self._parse_hub_path(record_id,
                                                          results["metadatas"][idx]))
                return hub_paths, results["embeddings"]
            except Exception as e:
                logger.error(
                    f"Failed to retrieve HubPaths for hub_entity {hub_entity_id}: {e}")
                raise

    def similarity_search_by_hub_entity(self,
                                        query_embeddings: List[List[float]],
                                        hub_entity_id: str,
                                        n_results: int,
                                        excluded_path_hashs: Optional[List[str]] = None) -> List[HubPath]:
        """
        Retrieves all HubPaths associated with a specific hub_entity scored and ranked by their
        similarity to the query embeddings.

        This function only retrieves paths from a specific hub and it is also possible to
        exclude paths from the search.

        Note:
            This approach uses a reranking approach where the top n_results * 2 results are initially
            gathered, the results are reranked and then the top n_results are returned. This is useful
            as the reranking can apply the DiversityRanker to the results.

        Args:
            query_embeddings (List[List[float]]): The embedding vector for which to find similar
                embeddings.
            hub_entity_id (str): The unique identifier of the hub_entity.
            n_results (int): The number of top similar embeddings to retrieve.
            excluded_path_hashs (List[str], optional): List of path hashes to exclude from search.

        Returns:
            List[HubPath]: List of similar hub paths with their similarity scores.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                if not excluded_path_hashs:
                    results: QueryResult = self.collection.query(
                        query_embeddings=query_embeddings,
                        where={"hub_entity":  hub_entity_id},
                        n_results=n_results*2,
                        include=["metadatas", "distances"]
                    )
                else:
                    # For details on filtering see
                    # https://docs.trychroma.com/docs/querying-collections/metadata-filtering
                    where_filter = {"$and": []}
                    where_filter["$and"].append(
                        {"path_hash": {"$nin": excluded_path_hashs}})
                    where_filter["$and"].append(
                        {"hub_entity": {"$eq": hub_entity_id}})

                    results: QueryResult = self.collection.query(
                        query_embeddings=query_embeddings,
                        where=where_filter,
                        n_results=n_results*2,
                        include=["metadatas", "distances"]
                    )
                hub_paths_with_score = self._process_query_results_to_hub_paths(
                    results)
                return hub_paths_with_score[:n_results]

            except Exception as e:
                if "cannot return the results" in str(e).lower():
                    logger.warning(
                        "Nearest Neighbor Query Failed trying manually..")
                    logger.debug(
                        f"Parameters: {hub_entity_id}, {n_results}, {excluded_path_hashs}")
                    # This happens if the n_results is too large for the
                    # number of embeddings in the hub
                    return self._manually_calculate_similarity_score(
                        hub_entity_id=hub_entity_id,
                        query_embeddings=query_embeddings,
                        n_results=n_results
                    )
                logger.error(
                    f"Failed to perform similarity search for hub_entity {hub_entity_id}: {e}")
                raise

    def similarity_search_hubs(self,
                               query_embeddings: List[List[float]],
                               excluded_hub_ids: List[str],
                               n_results: int = 10,) -> Dict[str, List[HubPath]]:
        """
        Performs a similarity search excluding embeddings from specified hub entities.

        Note:
            This approach uses a reranking approach where the top n_results * 2 results are initially
            gathered, the results are reranked and then the top n_results are returned. This is useful
            as the reranking can apply the DiversityRanker to the results.

        Args:
            query_embeddings (List[List[float]]): The embedding vector for which to find similar
                embeddings.
            excluded_hub_ids (List[str]): List of hub entity IDs to exclude from search.
            n_results (int): The number of top similar embeddings to retrieve.

        Returns:
            Dict[str, List[HubPath]]: A dictionary containing hub entity IDs as keys and
                a list of similar hub paths with their similarity scores as values.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:

                logger.info("Retrieve n_results * 2 from collection")

                if not excluded_hub_ids:
                    results: QueryResult = self.collection.query(
                        query_embeddings=query_embeddings,
                        n_results=n_results * 2,
                        include=["embeddings", "metadatas", "distances"]
                    )
                else:
                    results: QueryResult = self.collection.query(
                        query_embeddings=query_embeddings,
                        where={"hub_entity": {"$nin": excluded_hub_ids}},
                        n_results=n_results * 2,
                        include=["embeddings", "metadatas", "distances"]
                    )

                if not results or not results["ids"]:
                    return []



                hub_paths_clustered_by_hub_id = self._convert_query_result_to_hubpaths_clustered_by_hub_id(
                    results)

                logger.info("Rerank results by score and return n_results")

                # Sort the hub paths by their scores in descending order
                for hub_id, hub_paths in hub_paths_clustered_by_hub_id.items():
                    if self.diversity_penalty > 0:
                        hub_paths_clustered_by_hub_id[hub_id] = self._diversity_ranker_for_triples(
                            paths=hub_paths)
                    else:
                        hub_paths.sort(key=lambda x: x.score, reverse=True)

                    # ensure that the n_results is not larger than the amount of paths
                    hub_paths_clustered_by_hub_id[hub_id] = hub_paths[:n_results]

                return hub_paths_clustered_by_hub_id

            except Exception as e:
                logger.error(
                    f"Failed to perform similarity search excluding hub entities: {e}"
                )
                return {}

    def _convert_query_result_to_hubpaths_clustered_by_hub_id(
            self,
            results: QueryResult) -> Dict[str, List[HubPath]]:
        """
        Converts the QueryResult to HubPath objects clustered by the id of the Hub.

        Args:
            results (QueryResult): The query results from ChromaDB.

        Returns:
            Dict[str, List[HubPath]]: A dictionary mapping hub entity IDs to lists of HubPath objects.
        """
        if not results or not results["ids"] or len(results["ids"]) == 0:
            return {}

        logger.info("Converts the QueryResult to HubPath objects")

        hub_paths_by_id: Dict[str, List[HubPath]] = {}

        for query_idx in range(len(results["ids"])):
            metadatas = results["metadatas"][query_idx]
            ids = results["ids"][query_idx]
            embeddings = results["embeddings"][query_idx]
            distances = results["distances"][query_idx]

            for _, (path_hash, metadata, _, distance) in enumerate(zip(ids, metadatas, embeddings, distances)):

                hub_entity_id = metadata.get("hub_entity")
                hub_path = self._parse_hub_path(path_hash, metadata)

                hub_path.score = 1 - distance
                hub_path.embedded_text = metadata.get("embedded_text")

                # Add hub path to the cluster dictionary
                if hub_entity_id not in hub_paths_by_id:
                    hub_paths_by_id[hub_entity_id] = []
                hub_paths_by_id[hub_entity_id].append(hub_path)

        return hub_paths_by_id

    def _process_query_results_to_hub_paths(self,
                                            results: QueryResult) -> List[HubPath]:
        """
        This function takes the results from the vector store query and converts the results
        to HubPath objects. It also sorts the results by their score.

        Args:
            results (QueryResult): The query results from ChromaDB.

        Returns:
            List[HubPath]: A list of HubPath objects sorted by their score.
        """
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        similar_paths: List[HubPath] = []
        for k in range(len(results["ids"])):
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][k][i]
                distance = results["distances"][k][i]
                path_hash = metadata.get("path_hash")
                hub_path = self._parse_hub_path(path_hash, metadata)
                hub_path.score = 1 - distance
                hub_path.embedded_text = metadata.get("embedded_text")
                similar_paths.append(hub_path)

        if self.diversity_penalty > 0:
            similar_paths = self._diversity_ranker_for_triples(
                paths=similar_paths)
        else:
            similar_paths.sort(key=lambda x: x.score, reverse=True)
        return similar_paths

    def _diversity_ranker_for_triples(self,
                                      paths: List[HubPath]) -> List[HubPath]:
        """
        The following implementation reranks the paths by applying a penality to those paths
        that repeat the same subject multiple times. This is done because when working with
        triples, often times the same subject is repeated if it has multiple paths. As such
        with this function, repeated subjects are penalized where paths that have a lower
        score are penalized more than those with a higher score.

        The idea is similar to the DiversityRanker that is introduced on Haystack:
        https://towardsdatascience.com/enhancing-rag-pipelines-in-haystack-45f14e2bc9f5/

        However, our implementation differs as we are working with triples.

        Args:
            paths (List[HubPath]): The list of HubPath objects to be reranked.

        Returns:
            List[HubPath]: The reranked list of HubPath objects.
        """

        paths.sort(key=lambda x: x.score, reverse=True)

        subject_appearance = {}

        for path in paths:
            embedded_text = path.embedded_text

            subject = None
            if embedded_text and embedded_text.startswith("("):
                subject = embedded_text.split(",")[0]

            if subject:
                # Get the current rank for this subject
                count = subject_appearance.get(subject, 0)
                # Apply an increasing penalty
                path.score -= self.diversity_penalty * count
                subject_appearance[subject] = count + 1

        paths.sort(key=lambda x: x.score, reverse=True)
        return paths

    def _parse_hub_path(self, path_hash: str, metadata: Dict) -> HubPath:
        """
        Creates a HubPath object based on the metadata. This is needed as we
        can only store strings in the vector store as metadata. With this parser
        we can convert the metadata back to a HubPath object.

        Args:
            path_hash (str): The unique identifier for the path.
            metadata (dict): The metadata dictionary containing the path information.

        Returns:
            HubPath: The HubPath object created from the metadata.
        """
        path_as_string = metadata.get("path")
        path_as_list = path_as_string.split("$$$||$$$")
        path = [Triple.model_validate_json(hub_path_json) for
                hub_path_json in path_as_list]
        return HubPath(
            path_hash=path_hash,
            path=path,
            path_text=metadata.get("path_text"),
        )

    def _manually_calculate_similarity_score(self,
                                             hub_entity_id: str,
                                             query_embeddings: List[List[float]],
                                             n_results: int = 10) -> List[HubPath]:
        """
        We encountered an error with chromadb:
        "Cannot return the results in a contigious 2D array. Probably ef or M is too small"
        The error should no longer appear as whe increased the parameters, but in case it
        still does, this is a fallback method.

        This method is a workaround to manually calculate the similarity score
        between the query embeddings and the embeddings in the collection.

        Args:
            hub_entity_id (str): The unique identifier of the hub entity.
            query_embeddings (List[List[float]]): The embedding vector for which to find similar
                embeddings.
            n_results (int): The number of top similar embeddings to retrieve.

        Returns:
            List[HubPath]: A list of HubPath objects sorted by their similarity score.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")
        with self._lock:
            try:
                res = self.collection.get(
                    where={"hub_entity": hub_entity_id},
                    include=["embeddings", "metadatas"]
                )
                ids = res.get("ids", [])
                if not ids:
                    logger.debug(f"No hub paths for hub {hub_entity_id}")
                    return []

                logger.debug(
                    f"Found {len(ids)} hub paths for hub: {hub_entity_id}"
                )
                embeddings_array = np.array(res["embeddings"])  # shape (N, D)

            except Exception as e:
                logger.error(
                    f"Failed to fetch embeddings for hub_entity {hub_entity_id}: {e}"
                )
                raise

        if not query_embeddings:
            logger.debug("No query embeddings provided; returning empty result list.")
            return []

        max_similarities = self._calculate_max_similarities(
            query_embeddings=query_embeddings,
            embeddings_array=embeddings_array
        )

        hub_paths_with_scores: List[HubPath] = []
        for path_hash, metadata, score in zip(ids, res["metadatas"], max_similarities):
            hub_path = self._parse_hub_path(path_hash, metadata)
            hub_path.score = float(score)
            hub_path.embedded_text = metadata.get("embedded_text")
            hub_paths_with_scores.append(hub_path)

        hub_paths_with_scores.sort(key=lambda x: x.score, reverse=True)
        return hub_paths_with_scores[:n_results]

    def _calculate_max_similarities(self,
                                    query_embeddings: List[List[float]],
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
        sims    = cosine_similarity(queries, emb, dense_output=True)
        return sims.max(axis=0)
