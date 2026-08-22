"""Compare BM25 and SPLADE results for multiple query formulations.

Run from the ``experiments`` directory, for example:

    python experiment_runs/trigger_index/query_sparse.py \
        experiment_runs/1_experiment/base_configs/base_config_kit.json
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import sys

import chromadb
from dotenv import find_dotenv, load_dotenv

from _indexing_script_common import load_hublink_config
from core.data.file_path_manager import FilePathManager
from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.core.sparse_index.sparse_search_result import (
    SparsePathHit,
    SparseSearchResult,
)
from hublink.indexing.experiment.sparse_store_service_for_experiment import (
    SparseStoreServiceForExperiment,
)
from knowledge_base.sparse_index_store.bm25_tokenizer import (
    tokenize_for_bm25,
)
from knowledge_base.sparse_index_store.splade_encoder import get_splade_encoder
from knowledge_base.vector_store.storage.vector_store_provider import (
    VectorStoreProvider,
)
from language_model import LLMProvider


DEFAULT_QUERIES = [
    "Which papers did Georg Buchgeher publish in 2017?",
    #"author: Georg Buchgeher",
    #"publication year: 2017",
    "How frequently is the research object Technical Debt investigated per publication year?"
]
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "experiment_runs"
    / "1_experiment"
    / "base_configs"
    / "base_config_kit.json"
)


@dataclass
class QueryRun:
    query: str
    bm25_terms: list["Bm25Term"]
    splade_terms: list[tuple[str, float]]
    dense_hits: list["DenseHubHit"]
    bm25_hits: list[SparsePathHit]
    splade_hits: list[SparsePathHit]


@dataclass(frozen=True)
class Bm25Term:
    term: str
    query_frequency: int
    document_frequency: int
    idf: float

    @property
    def effective_weight(self) -> float:
        """Return the query-frequency multiplier applied to the term's IDF."""
        return self.query_frequency * self.idf


@dataclass(frozen=True)
class DenseHubHit:
    hub_id: str
    path_hash: str
    score: float
    matched_text: str


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare hub-level BM25 and SPLADE query results."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=(
            "Optional HubLink config path. Defaults to "
            "experiment_runs/1_experiment/base_configs/base_config_kit.json."
        ),
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help=(
            "Query text. Repeat this option to compare multiple queries. "
            "By default, the configured example questions and keyword queries "
            "are compared."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of hub results per channel and query (default: 10).",
    )
    parser.add_argument(
        "--term-limit",
        type=int,
        default=20,
        help="Maximum number of SPLADE query terms to print (default: 20).",
    )
    return parser


def _load_path_texts(config, path_hashes: list[str]) -> dict[str, str]:
    """Load full path text without initializing an embedding model."""
    settings = HubLinkSettings.from_config(config)
    store_name = VectorStoreProvider.compute_store_name(
        config=config,
        settings=settings,
        indexing_llm_config=config.index_llm_config,
    )
    store_path = Path(
        FilePathManager().CACHE_DIR,
        "hublink_retriever",
        store_name,
    )
    if not store_path.is_dir() or not path_hashes:
        return {}

    collection = chromadb.PersistentClient(path=str(store_path)).get_collection(
        name="novel_retriever"
    )
    records = collection.get(ids=path_hashes, include=["metadatas"])
    return {
        record_id: metadata.get("path_text", "")
        for record_id, metadata in zip(
            records["ids"], records.get("metadatas") or []
        )
    }


def _load_dense_collection(config):
    settings = HubLinkSettings.from_config(config)
    store_name = VectorStoreProvider.compute_store_name(
        config=config,
        settings=settings,
        indexing_llm_config=config.index_llm_config,
    )
    store_path = Path(
        FilePathManager().CACHE_DIR,
        "hublink_retriever",
        store_name,
    )
    if not store_path.is_dir():
        raise RuntimeError(
            f"Missing cached dense hub index '{store_name}'. Build it first "
            "with index_hubs.py <config>."
        )
    return chromadb.PersistentClient(path=str(store_path)).get_collection(
        name="novel_retriever"
    )


def _search_dense_hubs(
        collection,
        query_embedding: list[float],
        limit: int) -> list[DenseHubHit]:
    """Return top distinct hubs using the ordinary dense Chroma index."""
    hits: list[DenseHubHit] = []
    excluded_hubs: set[str] = set()
    collection_size = collection.count()

    while len(hits) < limit:
        remaining = limit - len(hits)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(remaining * 2, collection_size),
            where=(
                {"hub_entity": {"$nin": sorted(excluded_hubs)}}
                if excluded_hubs
                else None
            ),
            include=["metadatas", "distances"],
        )
        metadata_rows = (result.get("metadatas") or [[]])[0]
        distance_rows = (result.get("distances") or [[]])[0]
        if not metadata_rows:
            break

        best_by_hub: dict[str, DenseHubHit] = {}
        for metadata, distance in zip(metadata_rows, distance_rows):
            hub_id = metadata.get("hub_entity")
            path_hash = metadata.get("path_hash")
            if not hub_id or not path_hash:
                continue
            hit = DenseHubHit(
                hub_id=hub_id,
                path_hash=path_hash,
                score=1.0 - float(distance),
                matched_text=metadata.get("embedded_text", ""),
            )
            existing = best_by_hub.get(hub_id)
            if existing is None or hit.score > existing.score:
                best_by_hub[hub_id] = hit

        ranked_batch = sorted(
            best_by_hub.values(),
            key=lambda hit: hit.score,
            reverse=True,
        )
        if not ranked_batch:
            break
        hits.extend(ranked_batch[:remaining])
        excluded_hubs.update(hit.hub_id for hit in ranked_batch)

    return hits


def _print_hits(
        channel: str,
        hits: Sequence[SparsePathHit | DenseHubHit],
        path_texts: dict[str, str]) -> None:
    print(f"\n--- {channel} ({len(hits)} hub result(s)) ---")
    if not hits:
        print("No matching hubs.")
        return

    for rank, hit in enumerate(hits, start=1):
        path_text = (
            path_texts.get(hit.path_hash, "").strip()
            or "<path text unavailable>"
        )
        print(
            f"\n{rank}. score={hit.score:.6f}\n"
            f"   hub_id={hit.hub_id}\n"
            f"   path_hash={hit.path_hash}\n"
            f"   path={path_text}"
        )
        if isinstance(hit, DenseHubHit):
            print(f"   matched_text={hit.matched_text}")


def _best_hit_per_hub(result: SparseSearchResult) -> list[SparsePathHit]:
    return [
        result.hits_by_hub[hub_id][0]
        for hub_id in result.ranked_hub_ids
        if result.hits_by_hub.get(hub_id)
    ]


def _compute_bm25_idf(method: str, document_frequency: int,
                      document_count: int) -> float:
    """Reproduce BM25S's IDF variants for query diagnostics."""
    df = document_frequency
    n = document_count
    if method == "lucene":
        return math.log(1 + (n - df + 0.5) / (df + 0.5))
    if method == "robertson":
        return math.log(max((n - df + 0.5) / (df + 0.5), 1))
    if method == "atire":
        return math.log(n / df)
    if method == "bm25l":
        return math.log((n + 1) / (df + 0.5))
    if method == "bm25+":
        return math.log((n + 1) / df)
    raise ValueError(f"Unsupported BM25 IDF method: {method}")


def _get_bm25_terms(bm25_store, query: str) -> list[Bm25Term]:
    """Describe the query terms using the loaded BM25 index statistics."""
    bm25 = getattr(bm25_store, "_bm25", None)
    if bm25 is None:
        raise RuntimeError("BM25 index is not loaded.")

    query_tokens = tokenize_for_bm25([query])[0]
    query_frequencies = Counter(query_tokens)
    document_count = int(bm25.scores["num_docs"])
    indptr = bm25.scores["indptr"]
    terms: list[Bm25Term] = []
    for term, query_frequency in query_frequencies.items():
        token_id = bm25.vocab_dict.get(term)
        if token_id is None:
            document_frequency = 0
            idf = 0.0
        else:
            document_frequency = int(
                indptr[token_id + 1] - indptr[token_id]
            )
            idf = _compute_bm25_idf(
                bm25.idf_method,
                document_frequency,
                document_count,
            )
        terms.append(Bm25Term(
            term=term,
            query_frequency=query_frequency,
            document_frequency=document_frequency,
            idf=idf,
        ))
    return terms


def _print_overlap_summary(runs: list[QueryRun]) -> None:
    print("\n=== Top-hub overlap summary ===")
    for channel, attribute in (
        ("Dense ANN (normal hubs)", "dense_hits"),
        ("BM25", "bm25_hits"),
        ("SPLADE", "splade_hits"),
    ):
        print(f"\n{channel}:")
        hub_rankings = [
            [hit.hub_id for hit in getattr(run, attribute)]
            for run in runs
        ]
        hub_sets = [set(ranking) for ranking in hub_rankings]
        for index, ranking in enumerate(hub_rankings, start=1):
            print(f"  Q{index} hubs: {', '.join(ranking) or '<none>'}")

        for first, second in combinations(range(len(runs)), 2):
            overlap = [
                hub_id
                for hub_id in hub_rankings[first]
                if hub_id in hub_sets[second]
            ]
            print(
                f"  Q{first + 1} intersection Q{second + 1}: "
                f"{', '.join(overlap) or '<none>'}"
            )

        if len(runs) > 2:
            common_hubs = set.intersection(*hub_sets)
            ordered_common = [
                hub_id for hub_id in hub_rankings[0]
                if hub_id in common_hubs
            ]
            print(f"  All queries: {', '.join(ordered_common) or '<none>'}")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.term_limit < 1:
        parser.error("--term-limit must be at least 1.")
    queries = args.queries or list(DEFAULT_QUERIES)

    try:
        config, config_path = load_hublink_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    load_dotenv(find_dotenv(), override=True)
    sparse_stores = SparseStoreServiceForExperiment(
        config,
        load_bm25=True,
        load_splade=True,
    )
    manager = sparse_stores.sparse_storage_manager
    missing_channels = [
        name
        for name, available in (
            ("BM25", manager.has_bm25),
            ("SPLADE", manager.has_splade),
        )
        if not available
    ]
    if missing_channels:
        raise RuntimeError(
            f"Missing cached {' and '.join(missing_channels)} index(es) for "
            f"'{manager.index_key}'. Build them first with index_sparse.py "
            "<config> --bm25 --splade."
        )

    splade_model_name = getattr(manager.splade_store, "_model_name")
    splade_encoder = get_splade_encoder(splade_model_name)
    dense_collection = _load_dense_collection(config)
    settings = HubLinkSettings.from_config(config)
    dense_embeddings = LLMProvider().get_embeddings(
        settings.embedding_config
    ).embed_batch(queries)
    if dense_embeddings is None:
        raise RuntimeError("Dense embedding model returned no query embeddings.")
    runs: list[QueryRun] = []
    for query, dense_embedding in zip(queries, dense_embeddings):
        splade_query_vector = splade_encoder.encode_query(
            [query],
            batch_size=1,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
        )
        splade_terms = splade_encoder.decode(
            splade_query_vector,
            top_k=args.term_limit,
        )[0]
        bm25_result = manager.search_bm25(
            query_text=query,
            top_hubs=args.limit,
            paths_per_hub=1,
        )
        splade_result = manager.search_splade(
            query_text=query,
            top_hubs=args.limit,
            paths_per_hub=1,
        )
        runs.append(QueryRun(
            query=query,
            bm25_terms=_get_bm25_terms(manager.bm25_store, query),
            splade_terms=splade_terms,
            dense_hits=_search_dense_hubs(
                dense_collection,
                dense_embedding,
                args.limit,
            ),
            bm25_hits=_best_hit_per_hub(bm25_result),
            splade_hits=_best_hit_per_hub(splade_result),
        ))

    path_hashes = list(dict.fromkeys(
        hit.path_hash
        for run in runs
        for hit in run.dense_hits + run.bm25_hits + run.splade_hits
    ))
    path_texts = _load_path_texts(config, path_hashes)

    print(
        f"Config: {config_path}\n"
        f"Sparse index key: {manager.index_key}"
    )
    for index, run in enumerate(runs, start=1):
        print(f"\n\n========== Q{index}: {run.query} ==========")
        print("\nQuery representations:")
        print(f"  BM25 weighted terms ({len(run.bm25_terms)} unique):")
        print(
            "    "
            f"{'term':<20} {'query_tf':>8} {'doc_freq':>9} "
            f"{'idf':>10} {'effective_weight':>16}"
        )
        for term in run.bm25_terms:
            print(
                f"    {term.term:<20} {term.query_frequency:>8} "
                f"{term.document_frequency:>9} {term.idf:>10.6f} "
                f"{term.effective_weight:>16.6f}"
            )
        print(
            "    effective_weight = query_tf * idf; each hit's final term "
            "contribution also depends on its document term frequency and "
            "length."
        )
        print(f"  SPLADE weighted terms (top {len(run.splade_terms)}):")
        for term, weight in run.splade_terms:
            print(f"    {term:<20} {weight:.6f}")
        _print_hits("Dense ANN (normal hubs)", run.dense_hits, path_texts)
        _print_hits("BM25", run.bm25_hits, path_texts)
        _print_hits("SPLADE", run.splade_hits, path_texts)

    _print_overlap_summary(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
