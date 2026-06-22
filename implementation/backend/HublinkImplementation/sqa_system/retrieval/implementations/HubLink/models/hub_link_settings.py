from typing import List, Optional, Tuple
import json
from pydantic import BaseModel

from core.config import DatasetConfig, AdditionalConfigParameter, RestrictionType
from knowledge_base.config.chunking_strategy_config import ChunkingStrategyConfig
from language_model.config.embedding_config import EmbeddingConfig
from language_model.config.vector_store_config import VectorStoreConfig
from sqa_system.retrieval.config.kg_retrieval_config import KGRetrievalConfig

# The default values used for the parameters in the retriever
_HUB_LINK_DEFAULTS = {
    # In most cases the default value of -1 is sufficient as
    # it means that a hub is not determined by the number of edges
    "hub_edges": -1,
    # The default value is set to 10 but this is very much dependend
    # on the underlying data
    "top_paths_to_keep": 10,
    # Depends on the hardware and is set to 8 by default. 
    "max_workers": 8,
    # Its recommended to have this parameter set to to true as typically
    # hubs are the same amount of hops away from the topic entity
    "compare_hubs_with_same_hop_amount": True,
    # This parameter is set to false by default as it increases the
    # runtime. However, if dynamic updates are expected it may be
    # useful to set it to true.
    "check_updates_during_retrieval": False,
    # This values has been set arbitrarily to 5. It is entirely dependent on the 
    # underlying data.
    "max_level": 5,
    # This parameter is set to false by default. It should only be set to true
    # if an update to the index happened.
    "force_index_update": False,
    # This parameter is set to -1 by default to not restrict the depth of the
    # indexing process. However, if the graph is very large it may be useful to
    # set it to a value greater than 0. 
    "max_indexing_depth": -1,
    # This parameter is set to true by default. However, it entirely depends on the
    # data whether addition source documents are useful.
    "use_source_documents": True,
    # This parameter is set to true by default. It is generally recommended to filter
    # the outputs else the amount of data returned is large.
    "filter_output_context": True,
    # This parameter is set to 0.05 by default. During implementation we found during
    # our debugging that this value is good compared to other values. However, it 
    # depends on the data which value to use.
    "diversity_ranking_penalty": 0.05,
    # This parameter is set to 5 by default. During implementation we found during
    # our debugging that this value is good compared to other values. However, it 
    # depends on the data which value to use.
    "path_weight_alpha": 5,
    # This parameter is arbitrarily set to 10. It is entirely dependent on the underlying data.
    "max_hub_path_length": 10,
    # This parameter is arbitrarily set to 5. It is entirely dependent on the underlying data.
    "number_of_hubs": 5,
    # This parameter is set to 10 by default. It is entirely dependent on the underlying data.
    "number_of_source_chunks": 10,
    # This parameter is set to true by default. However, it depends on whether a topic entity
    # is provided with the question.
    "use_topic_if_given": True,
    # This parameter is set to true by default and it is recommended to leave it there as we
    # generally find that it improves the performance on complex questions
    "extract_question_components": True,
    # This parameter is set to cosine by default because of the popularity of the metric.
    "distance_metric": "cosine",
    # This parameter is set to false by default. It should only be enabled when the returned
    # context should be document chunks from the linking process instead of triples
    "return_source_data_as_context": False,
    # This parameter is entirely dependent on the underlying data.
    "indexing_root_entity_types": [],
    # This parameter is entirely dependent on the underlying data.
    "indexing_root_entity_ids": [],
    # This parameter is entirely dependent on the underlying data. We set it to the values expected
    # by the ORKG.
    "hub_types": [("entity_type", "Paper"),
                  ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                   "http://schema.org/ScholarlyArticle")]
}

_DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    name="openai_text-embedding-3-small",
    additional_params={},
    endpoint="OpenAI",
    name_model="text-embedding-3-small"
)

_DEFAULT_VECTOR_STORE_CONFIG = VectorStoreConfig(
    name="chroma_vector_store",
    additional_params={
        "distance_metric": "cosine"
    },
    vector_store_type="chroma",
    chunking_strategy_config=ChunkingStrategyConfig(
        name="recursivecharacterchunkingstrategy_csize300_coverlap15",
        additional_params={},
        chunking_strategy_type="RecursiveCharacterChunkingStrategy",
        chunk_size=300,
        chunk_overlap=15
    ),
    embedding_config=EmbeddingConfig(
        name="googleai_text-embedding-004",
        additional_params={},
        endpoint="GoogleAI",
        name_model="models/text-embedding-004"
    ),
    dataset_config=DatasetConfig(
        name="merged_ecsa_icsa",
        additional_params={},
        file_name="merged_ecsa_icsa.json",
        loader="JsonPublicationLoader",
        loader_limit=-1
    ),
)

ADDITIONAL_CONFIG_PARAMS: List[AdditionalConfigParameter] = [
        AdditionalConfigParameter(
            name="embedding_config",
            description="Configuration for the embeddings model.",
            param_type=EmbeddingConfig,
            available_values=[],
            default_value=_DEFAULT_EMBEDDING_CONFIG
        ),
        AdditionalConfigParameter(
            name="max_workers",
            description="Number of parallel workers for hub finding using during indexing.",
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["max_workers"],
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="filter_output_context",
            description=("Whether to additionally filter the output context by an LLm to make "
                         "sure that only relevant contexts are returned. This has no effect "
                         "on the generated answer, as the filtering is done after the answer "
                         "generation."),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["filter_output_context"],
        ),
        AdditionalConfigParameter(
            name="extract_question_components",
            description=(
                "When enabled, the LLM extracts the components from the question and embeds them "
                "separately. These embeddings will then be used together with the question embedding "
                "to find the relevant hubs. "
            ),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["extract_question_components"]
        ),


        # Comparison Related Settings
        AdditionalConfigParameter(
            name="number_of_hubs",
            description="Number of hubs that should be compared.",
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["number_of_hubs"],
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="top_paths_to_keep",
            description="Number of top scoring paths to keep for partial answer generation.",
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["top_paths_to_keep"],
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="path_weight_alpha",
            description=(
                "This value is a scaling factor for the exponential weighting of the HubPath "
                "scores. It allows to have those paths with higher scores to have a higher "
                "influence than those paths with a lower score. This is done by applying a "
                "weight to each score, where those with a higher score are weighted "
                "exponentially higher than those with a lower score. The reason this is "
                "useful, comes from how the hubs are pruned. Those hubs that have a lower "
                "overall score are removed from the list of candidates. This means that "
                "if a hub has just a couple of very high scoring paths but a lot of low scoring "
                "paths, it may be removed over a hub that has a lot of paths with a "
                "average score. This parameter however allows to increase the influence of "
                "those paths with a higher score, so that they are not removed so easily. "),
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["path_weight_alpha"],
            param_restriction=RestrictionType.GREQ_TO_ZERO
        ),
        AdditionalConfigParameter(
            name="diversity_ranking_penalty",
            description=(
                "This value is a penalty scaling factor that penalizes those HubPaths "
                "that repeat the same subject in a triple. Basically, the higher this "
                "value, the more each subsequent HubPath that contains a subject that "
                "has already been seen is penalized. This is useful to ensure that the "
                "retrieved HubPaths include more diverse information. "
            ),
            param_type=float,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["diversity_ranking_penalty"],
            param_restriction=RestrictionType.BETWEEN_ONE_AND_ZERO
        ),

        # Indexing Related Settings
        AdditionalConfigParameter(
            name="distance_metric",
            description=(
                "The distance metric that is used for the vector store. "
            ),
            param_type=str,
            available_values=["cosine", "l2", "ip"],
            default_value=""
        ),
        AdditionalConfigParameter(
            name="indexing_root_entity_types",
            description="(Optional) Root entity types to start indexing from, comma-delimited.",
            param_type=str,
            available_values=[],
            default_value=""
        ),
        AdditionalConfigParameter(
            name="indexing_root_entity_ids",
            description=("(Optional) The concrete ids of the root entities in the graph from "
                         "which to start the indexing process."),
            param_type=str,
            available_values=[],
            default_value=""
        ),
        AdditionalConfigParameter(
            name="force_index_update",
            description=(
                "Force the index to update. This forces each hub to be checked for updates "
                "during the indexing process."
            ),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["force_index_update"]
        ),
        AdditionalConfigParameter(
            name="max_hub_path_length",
            description="Maximum length of a hub path.",
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["max_hub_path_length"],
            param_restriction=RestrictionType.GREQ_THAN_MINUS_1
        ),
        AdditionalConfigParameter(
            name="check_updates_during_retrieval",
            description=("This option allows to force update the hubs during the retrieval. "
                         "This means, that at query time, each time a hub is accessed, it is "
                         "checked whether it has been updated beyond what is indexed. "
                         "This allows to ensure that the information is up to date without "
                         "having to run the whole indexing process. This works ONLY when "
                         "the `use_topic_if_given` setting is set to TRUE."),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["check_updates_during_retrieval"]
        ),
        AdditionalConfigParameter(
            name="hub_types",
            description=(
                "The entity types in the graph that are considered a hub. "
                "Can be a list of ['(predicate, object)', ..] tuples or a "
                "list of type string ['type1', 'type2', ...]."),
            param_type=str,
            available_values=[],
            default_value="['Paper']"
        ),
        AdditionalConfigParameter(
            name="hub_edges",
            description=(
                "Number of edges a entity needs to be classified as a hub."),
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["hub_edges"],
            param_restriction=RestrictionType.GREQ_THAN_MINUS_1
        ),

        # Topic Entity Related Settings
        AdditionalConfigParameter(
            name="use_topic_if_given",
            description="Whether to use the topic entity if given.",
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["use_topic_if_given"]
        ),
        AdditionalConfigParameter(
            name="compare_hubs_with_same_hop_amount",
            description=("This setting only works if `use_topic_if_given` is set "
                         "to TRUE and a topic entity is given. When this setting is "
                         "set to TRUE, the comparison of Hubs during retrieval needs "
                         "each Hub that is compared to be the same amount of hops away "
                         "from the topic entity."),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["compare_hubs_with_same_hop_amount"]
        ),
        AdditionalConfigParameter(
            name="max_level",
            description=("This parameter is only relevant if 'use_topic_if_given' is set to True. "
                         "The answer generation is done in levels. First hubs are searched for "
                         "and then partial answers are generated. If no answer is found, this "
                         "parameter controls how often the search and generation is continued."),
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["max_level"],
            param_restriction=RestrictionType.GREQ_THAN_MINUS_1
        ),

        # Source Link Related Settings
        AdditionalConfigParameter(
            name="use_source_documents",
            description="Whether to use source documents for answer generation.",
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["use_source_documents"]
        ),
        AdditionalConfigParameter(
            name="source_vector_store_config",
            description="Vector store configuration for source documents.",
            param_type=VectorStoreConfig,
            available_values=[],
            default_value=_DEFAULT_VECTOR_STORE_CONFIG
        ),
        AdditionalConfigParameter(
            name="number_of_source_chunks",
            description="The amount of text chunks from the source document to use.",
            param_type=int,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["number_of_source_chunks"],
        ),
        AdditionalConfigParameter(
            name="return_source_data_as_context",
            description=(
                "Whether to return the triples from the graph as context or the "
                "source document chunks from the source document handler."
            ),
            param_type=bool,
            available_values=[],
            default_value=_HUB_LINK_DEFAULTS["return_source_data_as_context"]
        )
    ]


class HubLinkSettings(BaseModel):
    """
    Settings for the HubLink retriever.
    """
    hub_edges: int = _HUB_LINK_DEFAULTS["hub_edges"]
    top_paths_to_keep: int = _HUB_LINK_DEFAULTS["top_paths_to_keep"]
    max_workers: int = _HUB_LINK_DEFAULTS["max_workers"]
    compare_hubs_with_same_hop_amount: bool = _HUB_LINK_DEFAULTS[
        "compare_hubs_with_same_hop_amount"]
    check_updates_during_retrieval: bool = _HUB_LINK_DEFAULTS["check_updates_during_retrieval"]
    max_level: int = _HUB_LINK_DEFAULTS["max_level"]
    diversity_ranking_penalty: float = _HUB_LINK_DEFAULTS["diversity_ranking_penalty"]
    path_weight_alpha: int = _HUB_LINK_DEFAULTS["path_weight_alpha"]
    force_index_update: bool = _HUB_LINK_DEFAULTS["force_index_update"]
    max_indexing_depth: int = _HUB_LINK_DEFAULTS["max_indexing_depth"]
    use_source_documents: bool = _HUB_LINK_DEFAULTS["use_source_documents"]
    extract_question_components: bool = _HUB_LINK_DEFAULTS["extract_question_components"]
    max_hub_path_length: int = _HUB_LINK_DEFAULTS["max_hub_path_length"]
    number_of_hubs: int = _HUB_LINK_DEFAULTS["number_of_hubs"]
    number_of_source_chunks: int = _HUB_LINK_DEFAULTS["number_of_source_chunks"]
    use_topic_if_given: bool = _HUB_LINK_DEFAULTS["use_topic_if_given"]
    return_source_data_as_context: bool = _HUB_LINK_DEFAULTS["return_source_data_as_context"]
    hub_types: Optional[List[Tuple[str, str] | str]] = _HUB_LINK_DEFAULTS["hub_types"]
    indexing_root_entity_types: Optional[List[str]] = _HUB_LINK_DEFAULTS["indexing_root_entity_types"]
    indexing_root_entity_ids: Optional[List[str]] = _HUB_LINK_DEFAULTS["indexing_root_entity_ids"]
    filter_output_context: bool = _HUB_LINK_DEFAULTS["filter_output_context"]
    distance_metric: str = _HUB_LINK_DEFAULTS["distance_metric"]
    # When True, partial answers per hub are skipped and all hub paths are fed
    # directly into the final answer generation prompt.
    use_direct_final_answer: bool = False

    # More complex defaults
    embedding_config: EmbeddingConfig = _DEFAULT_EMBEDDING_CONFIG
    source_vector_store_config: VectorStoreConfig = _DEFAULT_VECTOR_STORE_CONFIG

    @classmethod
    def from_config(cls, config: KGRetrievalConfig) -> "HubLinkSettings":
        """
        Dynamically constructs a HubLinkSettings instance from KGRetrievalConfig by 
        leveraging `ADDITIONAL_CONFIG_PARAMS` and each parameter's `parse_value`.
        """
        param_values = {}

        for param in ADDITIONAL_CONFIG_PARAMS:
            raw_val = config.additional_params.get(
                param.name, param.default_value)
            
            if raw_val is None:
                param_values[param.name] = None
                continue
            
            if issubclass(param.param_type, BaseModel):
                param_values[param.name] = raw_val
                continue

            if isinstance(raw_val, dict):
                raw_str = json.dumps(raw_val)
            else:
                raw_str = str(raw_val)

            parsed_value = param.parse_value(raw_str)

            if param.name == "indexing_root_entity_types" and isinstance(parsed_value, str):
                parsed_value = [x.strip()
                                for x in parsed_value.split(",") if x.strip()]
            elif param.name == "indexing_root_entity_ids" and isinstance(parsed_value, str):
                parsed_value = [x.strip()
                                for x in parsed_value.split(",") if x.strip()]

            param_values[param.name] = parsed_value

        return cls(**param_values)
