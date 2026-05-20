"""
Dataset-level integration tests for HubLinkService.

Runs a configurable slice of questions from reduced_deep_distributed_graph_dataset.csv
through the HubLink service and saves the full results to a timestamped JSON file.

Result file produced:
  results/batch_retrieval_results_{EMBEDDING_MODEL}_{LLM_MODEL}_{RETRIEVAL_MODE}_{NUMBER_OF_HUBS}_{QUESTION_START}-{QUESTION_END}.json

Parameters (edit here to change the run configuration):
  EMBEDDING_MODEL — embedding model name in short form (e.g. "qwen")
  LLM_MODEL       — LLM model name passed to hublink_service.query
  RETRIEVAL_MODE  — "direct" or "graph"
  NUMBER_OF_HUBS  — how many hubs to retrieve per query
  QUESTION_START  — first row index to include (0-based, inclusive)
  QUESTION_END    — last row index to exclude (exclusive)
"""
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DATASET_PATH = (
    Path(__file__).resolve().parents[6] / "data" / "reduced_deep_distributed_graph_dataset.csv"
)
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

EMBEDDING_MODEL = "qwen"
LLM_MODEL = "o3"
RETRIEVAL_MODE = "direct"
NUMBER_OF_HUBS = 10
QUESTION_START = 0
QUESTION_END = 40


def _load_questions(path: Path, start: int, end: int) -> list[dict]:
    questions = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= end:
                break
            if i >= start:
                questions.append(row)
    return questions


def test_hublink_batch_all_questions(hublink_service):
    """
    Runs the configured slice of dataset questions through the HubLink service.
    Saves detailed results to results/batch_retrieval_results_*.json.
    Asserts that at least one question was answered successfully.
    """
    questions = _load_questions(_DATASET_PATH, QUESTION_START, QUESTION_END)
    results = []

    for idx, row in enumerate(questions):
        question = row["question"]
        topic_entity_id = row.get("topic_entity_id") or None
        golden_answer = row.get("golden_answer", "")

        entry = {
            "index": QUESTION_START + idx,
            "question": question,
            "topic_entity_id": topic_entity_id,
            "golden_answer": golden_answer,
            "answer": None,
            "num_nodes": 0,
            "num_sources": 0,
            "sources": [],
            "error": None,
        }

        try:
            answer, nodes, sources = asyncio.run(
                hublink_service.query(
                    question=question,
                    retrieval_mode=RETRIEVAL_MODE,
                    llm_model=LLM_MODEL,
                    number_of_hubs=NUMBER_OF_HUBS,
                    topic_entity_id=topic_entity_id,
                )
            )
            entry["answer"] = answer
            entry["num_nodes"] = len(nodes)
            entry["num_sources"] = len(sources)
            entry["sources"] = sources
        except Exception as exc:
            entry["error"] = str(exc)

        results.append(entry)
        print(f"[{idx + 1}/{len(questions)}] {question[:80]}")

    _RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "retrieval_mode": RETRIEVAL_MODE,
            "number_of_hubs": NUMBER_OF_HUBS,
            "question_start": QUESTION_START,
            "question_end": QUESTION_END,
        },
        "results": results,
    }

    _emb = EMBEDDING_MODEL.replace(":", "-")
    _llm = LLM_MODEL.replace(":", "-")
    output_file_name = f"batch_retrieval_results_{_emb}_{_llm}_{RETRIEVAL_MODE}_{NUMBER_OF_HUBS}_{QUESTION_START}-{QUESTION_END}.json"
    output_path = _RESULTS_DIR / output_file_name
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    succeeded = sum(1 for r in results if r["error"] is None)
    assert succeeded > 0, f"All {len(results)} HubLink queries failed"
