"""Diagnostics for inspecting persisted BM25S and SPLADE cache payloads.

Run this test with ``pytest -s`` to display the generated JSON report. SPLADE
pickle files must only be inspected when produced by a trusted indexing run.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import bm25s
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
BM25_CACHE_DIRECTORY = CACHE_ROOT / "hublink_bm25_index"
SPLADE_CACHE_DIRECTORY = CACHE_ROOT / "hublink_splade_index"


def _sequence_summary(values: list[str]) -> dict[str, Any]:
    return {
        "count": len(values),
        "unique_count": len(set(values)),
        "sample": values[:3],
    }


def _summarize_bm25_cache(index_path: Path) -> dict[str, Any]:
    bm25 = bm25s.BM25.load(
        index_path,
        load_corpus=True,
        mmap=False,
        show_progress=False,
    )
    rows = list(bm25.corpus) if bm25.corpus is not None else []
    hub_ids = [row.get("hub_id") for row in rows]
    path_hashes = [row.get("path_hash") for row in rows]
    document_count = bm25.scores["num_docs"]
    schema_issues = []
    if len(rows) != document_count:
        schema_issues.append(
            f"Metadata has {len(rows)} rows but BM25S has {document_count}."
        )
    if any(hub_id is None for hub_id in hub_ids):
        schema_issues.append("BM25S metadata contains an incomplete row.")
    if any(path_hash is None for path_hash in path_hashes):
        schema_issues.append("BM25S path metadata contains an incomplete row.")

    return {
        "path": str(index_path),
        "format": "bm25s",
        "files": sorted(path.name for path in index_path.iterdir()),
        "method": bm25.method,
        "idf_method": bm25.idf_method,
        "k1": bm25.k1,
        "b": bm25.b,
        "document_count": document_count,
        "vocabulary_size": len(bm25.vocab_dict),
        "hub_ids": _sequence_summary(hub_ids),
        "path_hashes": _sequence_summary(path_hashes),
        "schema_issues": schema_issues,
    }


def _summarize_legacy_bm25_cache(file_path: Path) -> dict[str, Any]:
    return {
        "path": str(file_path),
        "format": "legacy-rank-bm25-pickle",
        "size_bytes": file_path.stat().st_size,
        "schema_issues": [
            "Legacy BM25 cache is ignored; rebuild indexing to create BM25S files."
        ],
    }


def _splade_summary(matrix: Any) -> dict[str, Any] | None:
    if matrix is None:
        return None

    if isinstance(matrix, torch.Tensor):
        sparse_matrix = matrix.detach().cpu().to_sparse_coo().coalesce()
        indices = sparse_matrix.indices()
        values = sparse_matrix.values()
        first_row_mask = indices[0] == 0 if matrix.shape[0] else None
        top_terms: list[dict[str, Any]] = []
        first_row_nonzero_values = 0
        if first_row_mask is not None:
            first_row_columns = indices[1][first_row_mask].tolist()
            first_row_weights = values[first_row_mask].tolist()
            first_row_nonzero_values = len(first_row_columns)
            top_terms = [
                {
                    "vocabulary_column": int(column),
                    "weight": float(weight),
                }
                for column, weight in sorted(
                    zip(first_row_columns, first_row_weights),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ]

        nonzero_values = sparse_matrix._nnz()
        total_cells = matrix.shape[0] * matrix.shape[1]
        return {
            "type": f"{type(matrix).__module__}.{type(matrix).__name__}",
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "layout": str(matrix.layout),
            "device": str(matrix.device),
            "nonzero_values": nonzero_values,
            "density": nonzero_values / total_cells if total_cells else 0.0,
            "first_row_nonzero_values": first_row_nonzero_values,
            "first_row_highest_weighted_columns": top_terms,
        }

    first_row = matrix.getrow(0) if matrix.shape[0] else None
    top_terms: list[dict[str, Any]] = []
    if first_row is not None:
        weighted_columns = zip(first_row.indices, first_row.data)
        top_terms = [
            {
                "vocabulary_column": int(column),
                "weight": float(weight),
            }
            for column, weight in sorted(
                weighted_columns,
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ]

    total_cells = matrix.shape[0] * matrix.shape[1]
    return {
        "type": f"{type(matrix).__module__}.{type(matrix).__name__}",
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "nonzero_values": matrix.nnz,
        "density": matrix.nnz / total_cells if total_cells else 0.0,
        "csr_array_lengths": {
            "data": len(matrix.data),
            "indices": len(matrix.indices),
            "indptr": len(matrix.indptr),
        },
        "first_row_nonzero_values": first_row.nnz if first_row is not None else 0,
        "first_row_highest_weighted_columns": top_terms,
    }


def _schema_issues(payload: dict[str, Any]) -> list[str]:
    if "matrix" in payload:
        required_keys = {
            "hub_ids",
            "path_hashes",
            "matrix",
            "model_name",
        }
    else:
        return ["Payload is not a recognized SPLADE cache."]

    issues = [
        f"Missing required key: {key}"
        for key in sorted(required_keys - payload.keys())
    ]
    hub_count = len(payload.get("hub_ids", []))
    path_count = len(payload.get("path_hashes", []))

    if hub_count != path_count:
        issues.append(
            f"hub_ids has {hub_count} rows but path_hashes has {path_count}."
        )

    matrix = payload.get("matrix")
    if matrix is not None and hub_count != matrix.shape[0]:
        issues.append(
            f"hub_ids has {hub_count} rows but SPLADE has {matrix.shape[0]}."
        )

    return issues


def _summarize_cache(file_path: Path) -> dict[str, Any]:
    with file_path.open("rb") as cache_file:
        payload = pickle.load(cache_file)

    assert isinstance(payload, dict), f"Invalid payload in {file_path}"

    hub_ids = payload.get("hub_ids", [])
    path_hashes = payload.get("path_hashes", [])
    return {
        "file": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "keys": list(payload),
        "value_types": {
            key: f"{type(value).__module__}.{type(value).__name__}"
            for key, value in payload.items()
        },
        "hub_ids": _sequence_summary(hub_ids),
        "path_hashes": _sequence_summary(path_hashes),
        "first_row_mapping": {
            "hub_id": hub_ids[0] if hub_ids else None,
            "path_hash": path_hashes[0] if path_hashes else None,
        },
        "model_name": payload.get("model_name"),
        "splade": _splade_summary(payload.get("matrix")),
        "schema_issues": _schema_issues(payload),
    }


def test_output_persisted_sparse_index_cache_structure() -> None:
    """Print a structural report for every locally persisted sparse index."""
    bm25_indexes = (
        sorted(
            path.parent
            for path in BM25_CACHE_DIRECTORY.glob("*/path/params.index.json")
        )
        if BM25_CACHE_DIRECTORY.exists()
        else []
    )
    legacy_bm25_files = (
        sorted(BM25_CACHE_DIRECTORY.glob("*.pkl"))
        if BM25_CACHE_DIRECTORY.exists()
        else []
    )
    splade_files = (
        sorted(SPLADE_CACHE_DIRECTORY.glob("*_path.pkl"))
        if SPLADE_CACHE_DIRECTORY.exists()
        else []
    )
    if not bm25_indexes and not legacy_bm25_files and not splade_files:
        pytest.skip(f"No sparse index caches found below {CACHE_ROOT}")

    summaries = [
        *[_summarize_bm25_cache(index_path) for index_path in bm25_indexes],
        *[
            _summarize_legacy_bm25_cache(file_path)
            for file_path in legacy_bm25_files
        ],
        *[_summarize_cache(file_path) for file_path in splade_files],
    ]
    print("\nSparse index cache structure:")
    print(json.dumps(summaries, indent=2))

    assert len(summaries) == (
        len(bm25_indexes) + len(legacy_bm25_files) + len(splade_files)
    )
