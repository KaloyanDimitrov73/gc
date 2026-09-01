"""Trigger BM25 and/or SPLADE indexing for one experiment config.

Run from the ``experiments`` directory, for example:

    python experiment_runs/trigger_index/index_sparse.py \
        experiment_runs/1_experiment/base_configs/base_config_kit.json

With no channel flags, the script builds the sparse channels enabled in the
config. Pass ``--bm25``, ``--splade``, or both to override that selection.
"""

import argparse
from collections.abc import Sequence

from dotenv import find_dotenv, load_dotenv

from backend.app.modules.indexing.infrastructure.hublink.hub_store_service import (
    HubStoreService,
)
from _indexing_script_common import load_hublink_config
from hublink.indexing.experiment.sparse_indexing_service import (
    SparseIndexingService,
)
from hublink.indexing.experiment.sparse_store_service import (
    SparseStoreService,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build sparse path indexes from paths already stored by dense "
            "HubLink indexing."
        )
    )
    parser.add_argument(
        "config",
        help="Path to a pipeline/baseline JSON containing one HubLink config.",
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="Build BM25, regardless of whether it is enabled in the config.",
    )
    parser.add_argument(
        "--splade",
        action="store_true",
        help="Build SPLADE, regardless of whether it is enabled in the config.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        config, config_path = load_hublink_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    load_dotenv(find_dotenv(), override=True)
    hub_stores = HubStoreService(config)
    sparse_stores = SparseStoreService(
        config,
        load_bm25=False,
        load_splade=False,
    )
    service = SparseIndexingService(
        config=config,
        hub_storage_manager=hub_stores.hub_storage_manager,
        sparse_storage_manager=sparse_stores.sparse_storage_manager,
    )

    has_explicit_selection = args.bm25 or args.splade
    build_bm25 = (
        args.bm25
        if has_explicit_selection
        else service.settings.use_bm25_hybrid_search
    )
    build_splade = (
        args.splade
        if has_explicit_selection
        else service.settings.use_splade_hybrid_search
    )
    if not build_bm25 and not build_splade:
        parser.error(
            "Neither sparse channel is enabled in the config. Pass --bm25, "
            "--splade, or both."
        )

    indexed_document_count = service.run_indexing(
        build_bm25=build_bm25,
        build_splade=build_splade,
    )
    if indexed_document_count == 0:
        raise RuntimeError(
            "No cached hub paths were found. Run trigger_index/index_hubs.py "
            "with the same config before sparse indexing."
        )

    channels = [
        channel
        for channel, enabled in (
            ("BM25", build_bm25),
            ("SPLADE", build_splade),
        )
        if enabled
    ]
    print(
        f"Sparse indexing completed for {indexed_document_count} path(s).\n"
        f"Config: {config_path}\n"
        f"Channels: {', '.join(channels)}\n"
        f"Sparse index key: {service.index_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
