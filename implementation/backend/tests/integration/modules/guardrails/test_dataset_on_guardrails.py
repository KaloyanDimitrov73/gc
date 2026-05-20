"""
Dataset-level validation tests for GibberishValidator and ToxicLanguageValidator.

Each test runs all questions or answers from the CSV dataset through a specific
validator configuration and saves the results as a CSV file in results/.

Result files produced:
  results/results_reduced_questions_gibberish.csv       — questions with Gibberish
  results/results_reduced_answers_gibberish.csv         — answers with Gibberish 
  results/results_reduced_questions_toxic.csv  — questions with Toxic only
  results/results_reduced_answers_toxic.csv    — answers with Toxic only

GibberishValidator is excluded from the default service configuration.
"""
import csv
from pathlib import Path

import pytest

from backend.app.modules.guardrails.application.service import GuardrailsService
from backend.app.modules.guardrails.application.validators import (
    GibberishValidator,
    ToxicLanguageValidator,
    JailbreakValidator,
)

guardrails = pytest.importorskip("guardrails", reason="guardrails-ai not installed")

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_REDUCED_DATASET_PATH = Path(__file__).resolve().parents[6] / "data"/ "reduced_deep_distributed_graph_dataset.csv"
_AI_GENERATED_DATASET_PATH = Path(__file__).resolve().parent / "dataset"

def _write_results(output_path: Path, rows: list[dict]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "passed", "reason"])
        writer.writeheader()
        writer.writerows(rows)

@pytest.fixture(scope="module")
def gibberish_svc():
    """Service with GibberishValidator."""
    gibberish = GibberishValidator()
    if not gibberish.is_available():
        pytest.skip("GibberishText not installed")
    return GuardrailsService(input_validators=[gibberish], output_validators=[gibberish])


@pytest.fixture(scope="module")
def toxic_svc():
    """Service with ToxicLanguageValidator."""
    toxic = ToxicLanguageValidator()
    if not toxic.is_available():
        pytest.skip("ToxicLanguage not installed")
    return GuardrailsService(input_validators=[toxic], output_validators=[toxic])

@pytest.fixture(scope="module")
def jailbreak_svc():
    """Service with JailbreakValidator."""
    jailbreak = JailbreakValidator()
    if not jailbreak.is_available():
        pytest.skip("JailbreakValidator not available")
    return GuardrailsService(input_validators=[jailbreak], output_validators=[jailbreak])

def validate_input(validator, dataset_file_path, row_label, result_file_name):
    """
    Validates all questions with the specified input validator.
    Saves to results/{result_file_name}.
    """
    rows = []
    failed = 0
    with open(dataset_file_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question = row[row_label]
            passed, reason = validator.validate_input(question)
            rows.append({"text": question, "passed": passed, "reason": reason})
            if not passed:
                failed += 1

    _write_results(_RESULTS_DIR / result_file_name, rows)
    print(f"\n{failed} question(s) failed validation")

def validate_output(validator, dataset_file_path, row_label, result_file_name):
    """
    Validates all answers with the specified output validator.
    Saves to results/{result_file_name}.
    """
    rows = []
    failed = 0
    with open(dataset_file_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            answer = row[row_label]
            _, passed, warning = validator.validate_output(answer)
            rows.append({"text": answer, "passed": passed, "reason": warning})
            if not passed:
                failed += 1

    _write_results(_RESULTS_DIR / result_file_name, rows)
    print(f"\n{failed} answer(s) failed validation")


def test_gibberish_input_all_questions(gibberish_svc):
    """
    Validates all questions with Gibberish as input validators.
    Saves to results/results_reduced_questions_gibberish.csv.
    """
    validate_input(gibberish_svc, _REDUCED_DATASET_PATH, "question", "results_reduced_questions_gibberish.csv")


def test_gibberish_output_all_answers(gibberish_svc):
    """
    Validates all answers with Gibberish as output validators.
    Saves to results/results_reduced_answers_gibberish.csv.
    """
    validate_output(gibberish_svc, _REDUCED_DATASET_PATH, "golden_answer", "results_reduced_answers_gibberish.csv")


def test_toxic_input_all_questions(toxic_svc):
    """
    Validates all questions with ToxicLanguageValidator.
    Saves to results/results_reduced_questions_toxic.csv.
    """
    validate_input(toxic_svc, _REDUCED_DATASET_PATH, "question", "results_reduced_questions_toxic.csv")


def test_toxic_output_all_answers(toxic_svc):
    """
    Validates all answers with ToxicLanguageValidator.
    Saves to results/results_reduced_answers_toxic.csv.
    """
    validate_output(toxic_svc, _REDUCED_DATASET_PATH, "golden_answer", "results_reduced_answers_toxic.csv")


def test_jailbreak_input_all_questions(jailbreak_svc):
    """
    Validates all questions with JailbreakValidator.
    Saves to results/results_reduced_questions_jailbreak.csv.
    """
    validate_input(jailbreak_svc, _REDUCED_DATASET_PATH, "question", "results_reduced_questions_jailbreak.csv")

def test_jailbreak_output_all_answers(jailbreak_svc):
    """
    Validates all answers with JailbreakValidator.
    Saves to results/results_reduced_answers_jailbreak.csv.
    """
    validate_output(jailbreak_svc, _REDUCED_DATASET_PATH, "golden_answer", "results_reduced_answers_jailbreak.csv")


def test_jailbreak_bad_questions(jailbreak_svc):
    validate_input(jailbreak_svc, _AI_GENERATED_DATASET_PATH / "gpt_generated_sample_jailbreak_questions.csv", "question", "results_jailbreak_questions.csv")

def test_jailbreak_bad_answers(jailbreak_svc):
    validate_output(jailbreak_svc, _AI_GENERATED_DATASET_PATH / "gpt_generated_sample_jailbreak_answers.csv", "answer", "results_jailbreak_answers.csv")
