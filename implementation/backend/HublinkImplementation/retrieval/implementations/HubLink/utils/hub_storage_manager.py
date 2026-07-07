from datetime import datetime
from typing import List, Optional, Dict, Tuple

import hashlib
from chromadb import QueryResult

from core.logging.logging import get_logger
from core.data.models.triple import Triple
from knowledge_base.vector_store.storage.vector_store import VectorStore, VectorScoreResults
from language_model import EmbeddingAdapter
from .hub_path_util import parse_hub_path, path_to_hash, deserialize_path, serialize_path
from ..models import EntityWithDirection

from ..models.hub_path import HubPath

logger = get_logger(__name__)


class HubStorageManager:
    """
    Domain layer for caching and retrieving HubPaths. Contains only Hub/Triple
    domain knowledge (parsing metadata into HubPath, diversity ranking, ...).

    All persistence/similarity-search work is delegated to an injected
    VectorStore implementation, so the backing database (Chroma, Pinecone, ...)
    can be swapped without touching this class.

    Args:
        vector_store (VectorStore): Already-initialized vector store adapter.
        diversity_penalty (float): Penalty for repeated subjects in diversity ranking.
    """

    def __init__(self, vector_store: VectorStore, embedding_model:  EmbeddingAdapter, diversity_penalty: float):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.diversity_penalty = diversity_penalty

    def vector_store_name(self):
        return self.vector_store.name

    def store_hub_batch(self,
                            hub_root_entity: EntityWithDirection,
                            paths: List[List[Triple]],
                            path_texts: List[str]) -> None:
        """
        Builds all storable entries for a hub (full paths, entities, triples),
        deduplicates them, embeds them in a single batch, and stores them.
        """

        all_texts, all_keys, all_metadata = self._collect_records(
            hub_root_entity, paths, path_texts
        )
        deduped_texts, deduped_keys, deduped_metadata = self._deduplicate_entries(
            all_texts, all_keys, all_metadata
        )

        logger.debug("Embedding %d texts for hub %s",
                     len(deduped_texts), hub_root_entity.entity.uid)
        embeddings = self._embed_texts(deduped_texts)

        self.vector_store.store_data_batch(
            ids=deduped_keys,
            embeddings=embeddings,
            metadatas=deduped_metadata
        )


    def delete_data_from_hub(self, hub_entity_id: str) -> None:
        """
        Deletes all embeddings associated with a specific hub_entity.

        Args:
            hub_entity_id (str): The unique identifier of the hub entity.
        """
        self.vector_store.delete_records_with_filter({"hub_entity": hub_entity_id})

    def retrieve_one_hub_path(self, path_hash: str) -> HubPath | None:
        """
        Returns one HubPath that contains the given path_hash as a metadata field.
        This is useful because a path is stored multiple times in the store on
        different levels: path text, triple, and entity level. With this function
        we can retrieve the first match of any of the three levels.

        Args:
            path_hash (str): The unique identifier for the path.

        Returns:
            HubPath: The first matching HubPath, or None if nothing matched.
        """
        result = self.vector_store.get_records_with_metadata_by_filter(
            where_filter={"path_hash": path_hash},
            limit=1
        )

        if result is None or result.is_empty:
            return None

        record_metadata = result.metadata[0]

        return parse_hub_path(path_hash=path_hash, path_as_string=record_metadata.get("path"), path_text=record_metadata.get("path_text"))

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
        result_map: Dict[str, Optional[HubPath]] = {key: None for key in hash_keys}

        result = self.vector_store.get_records_with_metadata_by_ids(hash_keys)

        if result is None or result.is_empty:
            return result_map

        for record_id, metadata in zip(result.ids, result.metadata):
            result_map[record_id] = parse_hub_path(path_hash=record_id, path_as_string=metadata.get("path"), path_text=metadata.get("path_text"))

        return result_map


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

        result = self.vector_store.get_records_with_metadata_by_filter(
            where_filter={"hub_entity": hub_entity_id},
            limit=None
        )
        if result is None or result.is_empty:
            return [], []

        hub_paths = [
            parse_hub_path(path_hash=record_id, path_as_string=metadata.get("path"), path_text=metadata.get("path_text"))
            for record_id, metadata in zip(result.ids, result.metadata)
        ]
        return hub_paths, result.embeddings


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
        if not excluded_path_hashs:
            where_filter = {"hub_entity":  hub_entity_id}
        else:
            where_filter = {"$and": []}
            where_filter["$and"].append(
                {"path_hash": {"$nin": excluded_path_hashs}})
            where_filter["$and"].append(
                {"hub_entity": {"$eq": hub_entity_id}})

        try:
            results = self.vector_store.vector_similarity_search(
                query_embeddings=query_embeddings,
                where_filter=where_filter,
                n_results=n_results * 2,
            )
        except Exception as e:
            logger.error(
                f"Failed to perform similarity search for hub_entity {hub_entity_id}: {e}")
            raise

        hub_paths_with_score = self._process_query_results_to_hub_paths(results)
        return hub_paths_with_score[:n_results]

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
            query_embeddings (List[List[float]]): List of embedding vectors for which to find similar embeddings.
            excluded_hub_ids (List[str]): List of hub entity IDs to exclude from search.
            n_results (int): The number of top similar embeddings to retrieve.

        Returns:
            Dict[str, List[HubPath]]: A dictionary containing hub entity IDs as keys and
                a list of similar hub paths with their similarity scores as values.
        """

        try:

            if not excluded_hub_ids:
                where_filter = None
            else:
                where_filter = {"hub_entity": {"$nin": excluded_hub_ids}}

            logger.info("Retrieve n_results * 2 from collection")

            results: List[VectorScoreResults] = self.vector_store.vector_similarity_search(
                query_embeddings=query_embeddings,
                where_filter=where_filter,
                n_results=n_results * 2,
            )

            if not results:
                return {}

            hub_paths_clustered_by_hub_id = self._convert_query_result_to_hubpaths_clustered_by_hub_id(results)

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

    def rebuild_index(self) -> None:
        self.vector_store.rebuild()

    def ensure_distance_metric(self, distance_metric: str) -> None:
        self.vector_store.ensure_distance_metric(distance_metric)

    def print_index_stats(self) -> None:
        self.vector_store.print_stats()

    def _convert_query_result_to_hubpaths_clustered_by_hub_id(
            self,
            results: List[VectorScoreResults]) -> Dict[str, List[HubPath]]:
        """
        Converts the list of VectorScoreResults (one per query embedding) to HubPath
        objects clustered by the id of the Hub.

        Args:
            results (List[VectorScoreResults]): The per-query results from the vector store.

        Returns:
            Dict[str, List[HubPath]]: A dictionary mapping hub entity IDs to lists of HubPath objects.
        """
        if not results:
            return {}

        logger.info("Converts the QueryResult to HubPath objects")

        hub_paths_by_id: Dict[str, List[HubPath]] = {}

        for query_idx, query_result in enumerate(results):
            if query_result.is_empty:
                continue

            if not query_result.metadata:
                continue

            if not query_result.embeddings:
                continue

            if not query_result.distances:
                continue

            ids = query_result.ids
            embeddings = query_result.embeddings
            metadatas = query_result.metadata
            distances = query_result.distances

            logger.info("-----------Query Result: %s -----------", query_idx)
            logger.info("ID: %s", ids)
            logger.info("Embeddings: %s", embeddings)
            logger.info("Metadata: %s", metadatas)
            logger.info("Distances: %s", distances)

            for path_hash, metadata, embedding, distance in zip(ids, metadatas, embeddings, distances):
                hub_entity_id = metadata.get("hub_entity")

                hub_path = parse_hub_path(
                    path_hash=path_hash,
                    path_as_string=metadata.get("path"),
                    path_text=metadata.get("path_text"),
                )

                logger.info("Hub Path: %s", hub_path)

                hub_path.score = 1 - distance
                hub_path.embedded_text = metadata.get("embedded_text")

                if hub_entity_id not in hub_paths_by_id:
                    hub_paths_by_id[hub_entity_id] = []
                hub_paths_by_id[hub_entity_id].append(hub_path)

        return hub_paths_by_id

    def _process_query_results_to_hub_paths(self, results: List[VectorScoreResults]) -> List[HubPath]:
        """
        This function takes the results from the vector store query and converts the results
        to HubPath objects. It also sorts the results by their score.

        Args:
            results (QueryResult): The query results from ChromaDB.

        Returns:
            List[HubPath]: A list of HubPath objects sorted by their score.
        """
        similar_paths: List[HubPath] = []

        for result in results:
            if result.is_empty:
                continue

            for metadata, distance in zip(result.metadata, result.distances):
                path_hash = metadata.get("path_hash")
                hub_path = parse_hub_path(path_hash=path_hash, path_as_string=metadata.get("path"), path_text=metadata.get("path_text"))
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

    def _collect_records(self,
                          hub_root_entity: EntityWithDirection,
                          paths: List[List[Triple]],
                          path_texts: List[str]
                          ) -> Tuple[List[str], List[str], List[dict]]:
        """
        Builds flat lists of storage records (texts, keys, metadata) for all paths  of a hub.

        Args:
            hub_root_entity (EntityWithDirection): The hub entity that is the
                root/center of these paths.
            paths (List[List[Triple]]): List of paths where each path is a list
                of Triple objects.
            path_texts (List[str]): The LLM-generated text representation for
                each path, in the same order as `paths`.

        Returns:
            Tuple[List[str], List[str], List[dict]]: Combined (not yet deduped)
                texts, hash keys, and metadata dicts across all paths.
        """

        all_texts: List[str] = []
        all_keys: List[str] = []
        all_metadata: List[dict] = []

        for path, path_text in zip(paths, path_texts):
            all_texts_form_path, all_keys_form_path, all_metadata_form_path = self._build_storage_record_form_path(hub_root_entity, path, path_text)
            all_texts.extend(all_texts_form_path)
            all_keys.extend(all_keys_form_path)
            all_metadata.extend(all_metadata_form_path)

        return all_texts, all_keys, all_metadata


    def _build_storage_record_form_path(self,
                                 hub_root_entity: EntityWithDirection,
                                 path: List[Triple],
                                 path_text: str
                                 ) -> Tuple[List[str], List[str], List[dict]]:

        """
        Builds all storage records for a single path.

        Creates three record types per path:
          1. Full path
          2. Entity texts (subject, object, predicate) of every triple
          3. Triples: formatted "(subject, predicate, object)" per triple

        Shared base metadata per record:
            - path_hash (str): MD5 hash of the path.
            - path_text (str): LLM-generated text of the full path.
            - hub_entity (str): UID of the hub root entity.
            - length (int): Number of triples in the path.
            - path (str): Path serialized as triples' JSON, joined by "$$$||$$$".
            - embedded_text (str): Record-specific text that gets embedded.
            - added_timestamp (str): Timestamp of record creation.
        """

        path_hash = path_to_hash(path)
        path_as_string = serialize_path(path)

        base_metadata = {
            "path_hash": path_hash,
            "path_text": path_text,
            "hub_entity": hub_root_entity.entity.uid,
            "length": len(path),
            "path": path_as_string,
        }
        timestamp = str(datetime.now())

        texts, keys, metadatas = [], [], []

        # Full path
        texts.append(path_text)
        keys.append(path_hash)

        metadatas.append(
            {**base_metadata,
             "embedded_text": path_text,
             "added_timestamp": timestamp})

        # Entities
        for triple in path:
            for entity_text in [triple.entity_subject.text,
                                triple.entity_object.text,
                                triple.predicate]:
                entity_hash = hashlib.md5(
                    (hub_root_entity.entity.uid + "_" + entity_text).encode()
                ).hexdigest()

                texts.append(entity_text)
                keys.append(entity_hash)
                metadatas.append({
                    **base_metadata,
                    "embedded_text": entity_text,
                    "added_timestamp": timestamp
                })

        # Triples
        for triple in path:
            triple_text = (
                f"({triple.entity_subject.text}, "
                f"{triple.predicate}, "
                f"{triple.entity_object.text})"
            )
            triple_hash = hashlib.md5(
                (path_hash + "_" + str(triple)).encode()
            ).hexdigest()

            texts.append(triple_text)
            keys.append(triple_hash)
            metadatas.append({
                **base_metadata,
                "embedded_text": triple_text,
                "added_timestamp": timestamp
            })

        return texts, keys, metadatas


    def _deduplicate_entries(self,
                              texts: List[str],
                              keys: List[str],
                              metadatas: List[dict]
                              ) -> Tuple[List[str], List[str], List[dict]]:
        """
        Removes duplicate records based on their hash key, keeping the first occurrence.

        Duplicates can occur because the same entity text or triple may appear
        in multiple paths of the same hub, producing the same hash key more than
        once.

        Args:
            texts (List[str]): Texts to embedd.
            keys (List[str]): Hash keys used to detect
                duplicates.
            metadatas (List[dict]): Metadata dicts corresponding to `texts`.

        Returns:
            Tuple[List[str], List[str], List[dict]]: Deduplicated texts, keys,
                and metadata dicts, in original order.
        """
        seen: set = set()
        deduped_texts: List[str] = []
        deduped_keys: List[str] = []
        deduped_metadatas: List[dict] = []

        for text, key, meta in zip(texts, keys, metadatas):
            if key not in seen:
                seen.add(key)
                deduped_texts.append(text)
                deduped_keys.append(key)
                deduped_metadatas.append(meta)
        return deduped_texts, deduped_keys, deduped_metadatas

    def _embed_texts(self, texts: List[str], retries: int = 8) -> List[List[float]]:
        """
        Embeds a list of texts using the embedding model.

        Args:
            texts (List[str]): The texts to embed.

        Returns:
            List[List[float]]: The embeddings of the texts.
        """
        for i in range(retries):
            try:
                embeddings = self.embedding_model.embed_batch(
                    texts)
                break
            except Exception as e:
                logger.error(f"Error during embedding: {e}")
                if i == retries - 1:
                    raise e
        return embeddings