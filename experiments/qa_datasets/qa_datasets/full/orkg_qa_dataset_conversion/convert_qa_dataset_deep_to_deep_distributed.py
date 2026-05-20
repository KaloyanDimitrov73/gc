import os
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.config.models import KnowledgeGraphConfig
from sqa_system.qa_generator import QADatasetToGraphConverter

RESEARCH_FIELD_ID = "R659055"
KG_CONFIG = KnowledgeGraphConfig.from_dict({
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
        "force_cache_update": True,
        "force_publication_update": False,
        "subgraph_root_entity_id": RESEARCH_FIELD_ID,
        "orkg_base_url": "https://sandbox.orkg.org"
    },
    "graph_type": "orkg",
    "dataset_config": {
        "additional_params": {},
        "file_name": "merged_ecsa_icsa.json",
        "loader": "JsonPublicationLoader",
        "loader_limit": -1
    }
})

if __name__ == "__main__":
    fpm = FilePathManager()
    current_directory = os.path.dirname(os.path.realpath(__file__))

    csv_files = fpm.get_files_in_folder(current_directory, file_type="csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in '{current_directory}'.")
    
    converter = QADatasetToGraphConverter(
        kg_config=KG_CONFIG
    )

    for csv_file in csv_files:
        if "updated_qa_dataset" in csv_file:
            continue          
        converter.run_conversion(
            qa_dataset_path=csv_file,
            research_field_id=RESEARCH_FIELD_ID,
        )
   
    