"""Run baseline evaluation and instruction-only DSPy optimization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any

# DSPy initializes its cache at import time. The workspace-local directory is
# writable in local, CI, and sandboxed experiment runs.
os.environ.setdefault(
    "DSPY_CACHEDIR",
    str(Path(__file__).resolve().parent / ".dspy_cache"),
)

import dspy
from dotenv import load_dotenv

from .dataset import load_dspy_examples
from .hublink_adapter import load_retrieval_adapter
from .metrics import HitAtKTripleMetric
from .program import (
    DEFAULT_QUESTION_PROCESSING_PROMPT,
    LegacyQuestionProcessingRetrievalProgram,
    QuestionProcessingRetrievalProgram,
    load_question_processing_instruction,
)


HERE = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = HERE.parent
DEFAULT_CONFIG = (
    EXPERIMENTS_ROOT
    / "experiment_runs"
    / "1_experiment"
    / "base_configs"
    / "base_config_kit.json"
)
KIT_TOOLBOX_BASE_URL = "https://ki-toolbox.scc.kit.edu/api"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize HubLink's schema-agnostic question-processing instruction "
            "against raw retrieval quality."
        )
    )
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--initial-prompt",
        type=Path,
        default=DEFAULT_QUESTION_PROCESSING_PROMPT,
        help=(
            "Existing HubLink question-processing YAML prompt used as the "
            "initial DSPy instruction."
        ),
    )
    parser.add_argument("--splits-dir", type=Path, default=HERE / "splits")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--dspy-model",
        help=(
            "LiteLLM-style model name, e.g. openai/gpt-4o-mini. Defaults to "
            "the query model in the pipeline config."
        ),
    )
    parser.add_argument("--dspy-api-base")
    parser.add_argument("--dspy-api-key-env")
    parser.add_argument(
        "--breadth",
        type=int,
        default=5,
        help="COPRO prompt candidates evaluated per optimization depth.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Number of iterative COPRO prompt-improvement rounds.",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    return parser


def _configure_dspy_lm(
    retrieval_config: Any,
    *,
    model_override: str | None,
    api_base_override: str | None,
    api_key_env_override: str | None,
) -> None:
    llm_config = retrieval_config.query_llm_config
    endpoint = llm_config.endpoint
    model = model_override
    api_base = api_base_override
    api_key_env = api_key_env_override

    if model is None:
        if endpoint in {"OpenAI", "KitToolbox"}:
            model = f"openai/{llm_config.name_model}"
        elif endpoint == "Ollama":
            model = f"ollama_chat/{llm_config.name_model}"
        else:
            raise ValueError(
                f"DSPy endpoint mapping is not defined for {endpoint!r}. "
                "Pass --dspy-model, --dspy-api-base, and "
                "--dspy-api-key-env explicitly."
            )

    if endpoint == "KitToolbox":
        api_base = api_base or KIT_TOOLBOX_BASE_URL
        api_key_env = api_key_env or "KIT_TOOLBOX_API_KEY"
    elif endpoint == "OpenAI":
        api_key_env = api_key_env or "OPENAI_API_KEY"

    lm_kwargs: dict[str, Any] = {}
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env!r} is required for DSPy."
            )
        lm_kwargs["api_key"] = api_key
    if llm_config.temperature is not None:
        lm_kwargs["temperature"] = llm_config.temperature

    dspy.configure(lm=dspy.LM(model, **lm_kwargs))


def _evaluate(program: Any, examples: list[Any], metric: Any) -> Any:
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
        max_errors=1,
    )
    return evaluator(program)


def _write_evaluation(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "uid",
                "question",
                "score",
                "components",
                "keywords",
                "retrieved_source_ids",
                "retrieved_triples",
            ),
        )
        writer.writeheader()
        for example, prediction, score in result.results:
            writer.writerow(
                {
                    "uid": getattr(example, "uid", ""),
                    "question": example.question,
                    "score": score,
                    "components": json.dumps(
                        getattr(prediction, "components", []),
                        ensure_ascii=False,
                    ),
                    "keywords": json.dumps(
                        getattr(prediction, "keywords", []),
                        ensure_ascii=False,
                    ),
                    "retrieved_source_ids": json.dumps(
                        getattr(prediction, "retrieved_source_ids", []),
                        ensure_ascii=False,
                    ),
                    "retrieved_triples": json.dumps(
                        getattr(prediction, "retrieved_triples", []),
                        ensure_ascii=False,
                    ),
                }
            )


def _write_optimized_instruction(path: Path, program: Any) -> None:
    """Persist the human-readable instruction selected by the optimizer."""

    instruction = program.process_question.signature.instructions
    path.write_text(instruction.strip() + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.breadth <= 1:
        raise ValueError("--breadth must be greater than one.")
    if args.depth <= 0:
        raise ValueError("--depth must be greater than zero.")
    load_dotenv(EXPERIMENTS_ROOT / ".env", override=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_dspy_examples(
        args.splits_dir / "train.csv", limit=args.train_limit
    )
    validation_examples = load_dspy_examples(
        args.splits_dir / "validation.csv", limit=args.validation_limit
    )
    test_examples = load_dspy_examples(
        args.splits_dir / "test.csv", limit=args.test_limit
    )

    adapter, retrieval_config = load_retrieval_adapter(args.pipeline_config)
    if not adapter.sparse_routing_enabled:
        warnings.warn(
            "Both sparse retrieval channels are disabled. DSPy can still "
            "optimize dense components, but keywords cannot affect the score.",
            stacklevel=1,
        )
    _configure_dspy_lm(
        retrieval_config,
        model_override=args.dspy_model,
        api_base_override=args.dspy_api_base,
        api_key_env_override=args.dspy_api_key_env,
    )

    metric = HitAtKTripleMetric(k=10)
    legacy_program = LegacyQuestionProcessingRetrievalProgram(adapter)
    initial_instruction = load_question_processing_instruction(args.initial_prompt)
    student_program = QuestionProcessingRetrievalProgram(
        adapter,
        initial_instruction=initial_instruction,
    )
    initial_instruction_path = args.output_dir / "initial_instruction.txt"
    initial_instruction_path.write_text(
        initial_instruction.strip() + "\n",
        encoding="utf-8",
    )

    legacy_validation = _evaluate(
        legacy_program, validation_examples, metric
    )
    _write_evaluation(
        args.output_dir / "legacy_validation.csv", legacy_validation
    )
    student_validation = _evaluate(
        student_program, validation_examples, metric
    )
    _write_evaluation(
        args.output_dir / "unoptimized_validation.csv", student_validation
    )

    optimizer = dspy.COPRO(
        metric=metric,
        breadth=args.breadth,
        depth=args.depth,
        track_stats=True,
    )
    copro_candidate = optimizer.compile(
        student_program,
        trainset=train_examples,
        eval_kwargs={
            "num_threads": args.num_threads,
            "display_progress": True,
            "display_table": False,
            "max_errors": 1,
        },
    )
    copro_validation = _evaluate(
        copro_candidate, validation_examples, metric
    )
    _write_evaluation(
        args.output_dir / "copro_candidate_validation.csv",
        copro_validation,
    )
    if copro_validation.score > student_validation.score:
        optimized_program = copro_candidate
        selected_program = "copro_candidate"
    else:
        optimized_program = student_program
        selected_program = "unoptimized"

    artifact_path = args.output_dir / "optimized_question_processing.json"
    optimized_program.save(str(artifact_path))
    instruction_path = args.output_dir / "optimized_instruction.txt"
    _write_optimized_instruction(instruction_path, optimized_program)

    legacy_test = _evaluate(legacy_program, test_examples, metric)
    unoptimized_test = _evaluate(student_program, test_examples, metric)
    copro_candidate_test = _evaluate(copro_candidate, test_examples, metric)
    optimized_test = (
        copro_candidate_test
        if selected_program == "copro_candidate"
        else unoptimized_test
    )
    test_results = {
        "legacy": legacy_test,
        "unoptimized": unoptimized_test,
        "copro_candidate": copro_candidate_test,
        "optimized": optimized_test,
    }
    summary = {
        "pipeline_config": str(args.pipeline_config.resolve()),
        "initial_prompt": str(args.initial_prompt.resolve()),
        "optimizer": "COPRO",
        "copro_breadth": args.breadth,
        "copro_depth": args.depth,
        "optimization_metric": f"hit@{metric.k}_triples",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "legacy_validation_score": legacy_validation.score,
        "unoptimized_validation_score": student_validation.score,
        "copro_candidate_validation_score": copro_validation.score,
        "selected_program": selected_program,
        "test_scores": {},
        "optimized_artifact": str(artifact_path.resolve()),
        "initial_instruction": str(initial_instruction_path.resolve()),
        "optimized_instruction": str(instruction_path.resolve()),
    }
    for name, result in test_results.items():
        _write_evaluation(args.output_dir / f"{name}_test.csv", result)
        summary["test_scores"][name] = result.score

    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
