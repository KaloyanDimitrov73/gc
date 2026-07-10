import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
from implementation.file_path_management import FilePathManager
from implementation.config.config_models import ExperimentConfig
from implementation.experimentation.experiment_runner import ExperimentRunner, ExperimentRunnerSettings


def main():
    """Main function for the test run."""

    # -----------------------
    # -- Prepare the paths --
    # -----------------------
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))
    folder_path = fpm.combine_paths(current_directory, "results")
    qa_dataset_path = fpm.combine_paths(
        fpm.get_parent_directory(current_directory, 1),
        "test_qa_dataset.csv"
    )
    test_run_config = ExperimentConfig.from_dict({
            "additional_params": {},
            "base_pipeline_config": {
                "additional_params": {},
                "pipes": [
                    {
                        "name": "pre_retrieval_augmentation",
                        "additional_params": {},
                        "type": "pre_retrieval_processing",
                        "pre_technique": "augmentation",
                        "llm_config": {
                            "name": "openai_gpt-4o-mini_tmp0.0_maxt-1",
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": -1
                        },
                        "enabled": False
                    },
                    {
                        "name": "retrieval_config",
                        "additional_params": {
                            "use_topic_if_given": False,
                            "embedding_config": {
                                "additional_params": {},
                                "endpoint": "Ollama",
                                "name_model": "mxbai-embed-large"
                            },
                            "diversity_ranking_penalty": 0.05,
                            "path_weight_alpha": 5,
                            "top_paths_to_keep": 10,
                            "number_of_hubs": 10,
                            "extract_question_components": True,
                            "distance_metric": "cosine",
                            "filter_output_context": True,
                            "use_source_documents": False,
                            "number_of_source_chunks": 10,                                       
                            "hub_edges": -1,						
                            "max_workers": 8,
                            "compare_hubs_with_same_hop_amount": False,
                            "check_updates_during_retrieval": False,
                            "max_level": 1,
                            "force_index_update": True,
                            "max_indexing_depth": -1,	
                            "max_hub_path_length": -1,
                            "return_source_data_as_context": False,
                            "indexing_root_entity_types": None,
                            "indexing_root_entity_ids": ["R659055"],
                            "hub_types": ["Paper"]                        
                        },
                        "type": "kg_retrieval",
                        "retriever_type": "hublink",
                        "llm_config": {
                            "endpoint": "OpenAI",
                            "name_model": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": -1
                        },
                        "knowledge_graph_config": {
                            "additional_params": {
                                "contribution_building_blocks": {
                                    "Paper Class 2": [
                                        "paper_class"
                                    ],
                                    "Research Level 2": [
                                        "research_level"
                                    ],
                                    "First Research Object 2": [
                                        "first_research_object"
                                    ],
                                    "Second Research Object 2": [
                                        "second_research_object"
                                    ],
                                    "Validity 2": [
                                        "validity"
                                    ],
                                    "Evidence 2": [
                                        "evidence"
                                    ]
                                },
                                "force_cache_update": False,
                                "force_publication_update": False,
                                "subgraph_root_entity_id": "R659055",
                                "orkg_base_url": "https://sandbox.orkg.org"
                            },
                            "graph_type": "orkg",
                            "dataset_config": {
                                "additional_params": {},
                                "file_name": "merged_ecsa_icsa.json",
                                "loader": "JsonPublicationLoader",
                                "loader_limit": -1
                            }
                        }
                    },
                    {
                        "additional_params": {},
                        "type": "generation",
                        "llm_config": {
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": -1
                        }
                    }
                ]
            },
            "parameter_ranges": [
                {
                    "config_name": "retrieval_config",
                    "parameter_name": "llm_config",
                    "values": [
                        {
                            "additional_params": {},
                            "endpoint": "Ollama",
                            "name_model": "llama3.1",
                            "temperature": 0.0,
                            "max_tokens": 4096
                        },
                        {
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "gpt-4o-mini",
                            "temperature": 0.0,
                            "max_tokens": -1
                        },
                        {
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "gpt-4o",
                            "temperature": 0.0,
                            "max_tokens": -1
                        },
                        {
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "o3-mini",
                            "temperature": None,
                            "max_tokens": -1,
                            "reasoning_effort": "low"
                        }
                    ]
                },
                {
                    "config_name": "retrieval_config",
                    "parameter_name": "additional_params",
                    "dict_key": "embedding_config",
                    "values": [
                        {
                            "additional_params": {},
                            "endpoint": "Ollama",
                            "name_model": "mxbai-embed-large"
                        },
                        {
                            "additional_params": {},
                            "endpoint": "Ollama",
                            "name_model": "granite-embedding"
                        },
                        {
                            "additional_params": {},
                            "endpoint": "OpenAI",
                            "name_model": "text-embedding-3-large"
                        }
                    ]
                }
            ],
            "evaluators": [
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "hit_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "entity"
                    },
                    "evaluator_type": "hit_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "hit_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "map_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "entity"
                    },
                    "evaluator_type": "map_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "mrr_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "entity"
                    },
                    "evaluator_type": "mrr_at_k"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "basic_score"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "entity"
                    },
                    "evaluator_type": "basic_score"
                },
                {
                    "additional_params": {
                        "k": -1,
                        "context_type": "triple"
                    },
                    "evaluator_type": "basic_score"
                },
                {
                    "additional_params": {
                        "k": -1,
                        "context_type": "entity"
                    },
                    "evaluator_type": "basic_score"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "triple"
                    },
                    "evaluator_type": "exact_match"
                },
                {
                    "additional_params": {
                        "k": 10,
                        "context_type": "entity"
                    },
                    "evaluator_type": "exact_match"
                },
                {
                    "additional_params": {
                        "k": -1,
                        "context_type": "triple"
                    },
                    "evaluator_type": "exact_match"
                },
                {
                    "additional_params": {
                        "k": -1,
                        "context_type": "entity"
                    },
                    "evaluator_type": "exact_match"
                },
                {
                    "additional_params": {},
                    "evaluator_type": "bleu_score"
                },
                {
                    "additional_params": {},
                    "evaluator_type": "rouge_score"
                }
            ]
        }
    )

    # ------------------------
    # -- Run the Experiment --
    # ------------------------
    runner = ExperimentRunner(
        experiment_config=test_run_config,
        settings=ExperimentRunnerSettings(
            results_folder_path=fpm.combine_paths(
                folder_path, "test_run", test_run_config.config_hash),
            qa_data_path=qa_dataset_path,
            debugging=True,
            log_to_results_folder=True,
            weave_project_name="experiment_tests",
            number_of_workers=1
        )
    )

    results = runner.run()
    print(results.head())

if __name__ == '__main__':
    main()