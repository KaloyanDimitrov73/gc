"""
First-time setup script for the HubLink backend.

What this script does:
  1. Uploads papers from merged_ecsa_icsa.json to the ORKG sandbox (skips papers
     that already exist, unless --force-publication-update is passed).
  2. Downloads the subgraph from sandbox and saves it as a local JSON cache.

After this script runs once, the backend can start without credentials and will
serve from the JSON cache on every subsequent start.

Requirements:
  - ORKG_EMAIL and ORKG_PASSWORD must be set (as env vars or in .env).
  - The publication dataset file must be present at the path expected by
    JsonPublicationLoader (merged_ecsa_icsa.json).

Usage:
  python init_graph_cache.py
  python init_graph_cache.py --force-publication-update
"""

import argparse
import sys
from pathlib import Path

# Ensure sqa_system is importable from the HublinkImplementation submodule.
PROJECT_ROOT = Path(__file__).resolve().parent
HUBLINK_IMPL_DIR = PROJECT_ROOT / "HublinkImplementation"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HUBLINK_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(HUBLINK_IMPL_DIR))

from app.modules.qa.infrastructure.hublink.setup_manager import SetupManager
from sqa_system.core.config.models import KnowledgeGraphConfig
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraphManager


# Deep-distributed graph config — the variant used by the backend.
KG_CONFIG_DICT = {
    "graph_type": "orkg",
    "additional_params": {
        "orkg_base_url": "https://sandbox.orkg.org",
        "subgraph_root_entity_id": "R659055",
        "force_cache_update": False,
        "force_publication_update": False,   # overridden by --force-publication-update
        "limit_publications": -1,
        "contribution_building_blocks": {
            "Paper Class 2": ["paper_class"],
            "Research Level 2": ["research_level"],
            "First Research Object 2": ["first_research_object"],
            "Second Research Object 2": ["second_research_object"],
            "Validity 2": ["validity"],
            "Evidence 2": ["evidence"],
        },
    },
    "dataset_config": {
        "additional_params": {},
        "file_name": "merged_ecsa_icsa.json",
        "loader": "JsonPublicationLoader",
        "loader_limit": -1,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--force-publication-update",
        action="store_true",
        help="Delete and re-upload all papers in ORKG (default: skip existing papers).",
    )
    args = parser.parse_args()

    # 1: Store ORKG credentials so ORKGKnowledgeGraphFactory can authenticate.
    print("Setting up ORKG credentials...")
    setup_manager = SetupManager()
    setup_manager.set_up_orkg_credentials()
    print("Credentials configured.")

    # 2: Build config, optionally forcing publication re-upload.
    config_dict = KG_CONFIG_DICT.copy()
    config_dict["additional_params"] = dict(KG_CONFIG_DICT["additional_params"])
    if args.force_publication_update:
        print("--force-publication-update set: existing papers will be deleted and re-uploaded.")
        config_dict["additional_params"]["force_publication_update"] = True

    kg_config = KnowledgeGraphConfig.from_dict(config_dict)

    # 3: Run KnowledgeGraphManager — this uploads any missing papers to ORKG
    # (via ORKGKnowledgeGraphFactory) and then calls cache_subgraph() which downloads
    # the subgraph and saves it as a JSON file on disk.
    print("Uploading papers to ORKG sandbox (skipping papers that already exist)...")
    print("Then downloading and caching the subgraph. This may take several minutes.")
    kg_manager = KnowledgeGraphManager()
    kg_manager.get_item(kg_config)

    print("\nSetup complete. The backend can now be started without credentials.")
    print("The graph cache is stored locally and will be used on each backend start.")


if __name__ == "__main__":
    main()
