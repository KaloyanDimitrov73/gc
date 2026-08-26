import ast
import csv
import json
import os
import re
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
QA_DATASET_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "experiment_runs"
    / "1_experiment"
    / "runs"
    / "0_test_runs"
    / "test_qa_dataset.csv"
)
PROMPT_PATH = (
    REPOSITORY_ROOT
    / "implementation"
    / "backend"
    / "HublinkImplementation"
    / "data"
    / "prompts"
    / "novel_retriever"
    / "question_processing_prompt.yaml"
)
KIT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "experiment_runs"
    / "1_experiment"
    / "base_configs"
    / "base_config_kit.json"
)
KIT_TOOLBOX_BASE_URL = "https://ki-toolbox.scc.kit.edu/api"


def _parse_question_processing_output(output: str) -> dict[str, list[str]]:
    dictionary_candidates = re.findall(
        r"\{[^{}]*\}", output, flags=re.DOTALL
    )
    for candidate in reversed(dictionary_candidates):
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue

        if not isinstance(parsed, dict):
            continue
        if set(parsed) != {"components", "keywords"}:
            continue
        if not all(isinstance(value, list) for value in parsed.values()):
            continue
        if not all(
            isinstance(item, str)
            for value in parsed.values()
            for item in value
        ):
            continue
        return parsed

    raise AssertionError(
        f"The LLM did not return a valid question-processing dictionary: "
        f"{output!r}"
    )


@pytest.mark.integration
def test_question_processing_prompt_keyword_results() -> None:
    """Print keyword extraction results for every question in the test CSV.

    Run with ``pytest -s --run-integration`` so that the per-question output is
    displayed. Set ``QUESTION_PROCESSING_TEST_MODEL`` to override the model
    configured in ``base_config_kit.json``.
    """
    load_dotenv(REPOSITORY_ROOT / "experiments" / ".env", override=True)
    kit_api_key = os.environ.get("KIT_TOOLBOX_API_KEY")
    if not kit_api_key:
        pytest.skip("Set KIT_TOOLBOX_API_KEY to run this LLM prompt test.")

    with PROMPT_PATH.open(encoding="utf-8") as prompt_file:
        prompt_config = yaml.safe_load(prompt_file)
    with KIT_CONFIG_PATH.open(encoding="utf-8") as config_file:
        kit_config = json.load(config_file)

    configured_model = kit_config["pipes"][0]["query_llm_config"][
        "name_model"
    ]
    timeout_seconds = float(
        os.environ.get("QUESTION_PROCESSING_TEST_TIMEOUT", "300")
    )

    prompt = PromptTemplate(
        template=prompt_config["template"],
        input_variables=prompt_config["input_variables"],
    )
    model = ChatOpenAI(
        base_url=KIT_TOOLBOX_BASE_URL,
        api_key=kit_api_key,
        model=os.environ.get(
            "QUESTION_PROCESSING_TEST_MODEL", configured_model
        ),
        temperature=None,
        timeout=timeout_seconds,
        max_retries=0,
    )
    chain = prompt | model | StrOutputParser()

    with QA_DATASET_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    empty_keyword_count = 0
    failed_queries: list[tuple[int, str, str]] = []
    for index, row in enumerate(rows, start=1):
        question = row["question"]
        print(f"\n[{index}/{len(rows)}] Processing: {question}", flush=True)
        try:
            raw_output = chain.invoke({"question": question})
            result = _parse_question_processing_output(raw_output)
        except Exception as error:
            error_summary = f"{type(error).__name__}: {error}"
            failed_queries.append((index, question, error_summary))
            print(f"  ERROR: {error_summary}", flush=True)
            continue

        keywords = result["keywords"]
        retrieval_mode = "HYBRID" if keywords else "DENSE_ONLY"
        empty_keyword_count += not keywords

        print(f"  Components: {result['components']}")
        print(f"  Keywords:   {keywords}")
        print(f"  Mode:       {retrieval_mode}")

    successful_query_count = len(rows) - len(failed_queries)
    empty_percentage = (
        empty_keyword_count / successful_query_count * 100
        if successful_query_count
        else 0.0
    )
    print("\nQuestion processing summary")
    print(f"  Total queries:       {len(rows)}")
    print(f"  Successful queries:  {successful_query_count}")
    print(f"  Failed queries:      {len(failed_queries)}")
    print(f"  Empty keyword lists: {empty_keyword_count}")
    print(f"  Dense-only share:    {empty_percentage:.1f}% of successful queries")

    if failed_queries:
        print("\nFailed query details")
        for index, question, error_summary in failed_queries:
            print(f"  [{index}] {question}")
            print(f"      {error_summary}")
        pytest.fail(
            f"{len(failed_queries)} of {len(rows)} LLM requests failed. "
            "Increase QUESTION_PROCESSING_TEST_TIMEOUT and rerun if needed."
        )
