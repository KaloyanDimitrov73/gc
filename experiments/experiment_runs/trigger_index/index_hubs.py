"""Trigger dense HubLink hub traversal and indexing for one experiment config.

Run from the ``experiments`` directory, for example:

    python experiment_runs/trigger_index/index_hubs.py \
        experiment_runs/1_experiment/base_configs/base_config_kit.json
"""

import argparse
from collections.abc import Sequence

from dotenv import find_dotenv, load_dotenv

from backend.app.modules.indexing.infrastructure.hublink.hub_indexing_service import (
    HubIndexingService,
)
from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import (
    HubStoreService,
)
from _indexing_script_common import load_hublink_config
from hublink.indexing.util.root_entity_util import resolve_root_entities
from knowledge_base.knowledge_graph.storage.implementations.orkg_remote_graph import (
    ORKGRemoteGraph,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse the configured knowledge graph and build the dense "
            "HubLink hub index."
        )
    )
    parser.add_argument(
        "config",
        help="Path to a pipeline/baseline JSON containing one HubLink config.",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Rebuild hubs even when cached hub data already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config, config_path = load_hublink_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    graph_type = config.knowledge_graph_config.graph_type.lower()
    if graph_type != "orkg":
        parser.error(
            "Experiment hub indexing currently supports the ORKG graph type; "
            f"received '{graph_type}'."
        )

    load_dotenv(find_dotenv(), override=True)
    graph = ORKGRemoteGraph(config.knowledge_graph_config)
    graph.update_cache_if_not_exists()

    stores = HubStoreService(config)
    store_was_empty = stores.hub_storage_manager.vector_store_is_empty()
    indexing_service = HubIndexingService(
        config=config,
        graph=graph,
        hub_storage_manager=stores.hub_storage_manager,
    )
    if store_was_empty:
        indexed_root_count = len(resolve_root_entities(
            graph=graph,
            root_entity_ids=(
                indexing_service.hublink_settings.indexing_root_entity_ids
            ),
            root_entity_types=(
                indexing_service.hublink_settings.indexing_root_entity_types
            ),
        ))
    else:
        indexed_root_count = indexing_service.run_indexing(
            graph=graph,
            force_update=args.force_update,
        )

    if indexed_root_count == 0:
        raise RuntimeError(
            "No indexing roots were resolved. Check indexing_root_entity_ids "
            "and indexing_root_entity_types in the HubLink config."
        )

    print(
        f"Dense hub indexing completed for {indexed_root_count} root(s).\n"
        f"Config: {config_path}\n"
        f"Vector store: {stores.store_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
