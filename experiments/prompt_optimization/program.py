"""DSPy programs for optimized and legacy question processing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# DSPy reads this setting while importing. Keep its cache inside the experiment
# instead of relying on a writable user home directory.
os.environ.setdefault(
    "DSPY_CACHEDIR",
    str(Path(__file__).resolve().parent / ".dspy_cache"),
)

import dspy

from .hublink_adapter import HubLinkRawRetrievalAdapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION_PROCESSING_PROMPT = (
    REPOSITORY_ROOT
    / "implementation"
    / "backend"
    / "HublinkImplementation"
    / "data"
    / "prompts"
    / "novel_retriever"
    / "question_processing_prompt.yaml"
)
_QUESTION_SECTION_MARKER = "### Start Your Task:"


SCHEMA_AGNOSTIC_CONTRACT = """
The instruction and outputs must work without knowing a particular
knowledge-graph schema or ontology. Do not inject or rely on internal entity
classes, relation/property names, or identifiers that are not stated in the
natural-language question. Components may use concepts expressed by the user.
Keywords are exact lexical anchors from the question (for example names,
identifiers, dates, numbers, or exact titles) that may benefit sparse retrieval.
""".strip()


class ProcessQuestion(dspy.Signature):
    """Prepare a question for accurate knowledge-graph retrieval."""

    question: str = dspy.InputField(desc="The user's natural-language question.")
    retrieval_contract: str = dspy.InputField(
        desc="Fixed constraints that every optimized instruction must follow."
    )
    components: list[str] = dspy.OutputField(
        desc=(
            "Graph-aware, schema-independent semantic components for dense "
            "retrieval. Preserve entities, concepts, constraints, and "
            "relationship cues explicitly expressed in the question, including "
            "wording that may correspond to graph relations. Do not invent or "
            "assume schema terms that are not stated in the question."
        )
    )
    keywords: list[str] = dspy.OutputField(
        desc="Exact lexical anchors that determine whether sparse retrieval runs."
    )


def load_question_processing_instruction(
    prompt_path: Path | str = DEFAULT_QUESTION_PROCESSING_PROMPT,
) -> str:
    """Load the existing HubLink YAML prompt as the DSPy seed instruction.

    DSPy supplies ``question`` as a typed input field, so the final runtime
    question section from the LangChain template is removed. Double braces in
    YAML examples are LangChain escaping and are converted back to literal
    dictionary braces for DSPy's instruction.
    """

    prompt_path = Path(prompt_path)
    with prompt_path.open(encoding="utf-8") as prompt_file:
        prompt_config = yaml.safe_load(prompt_file)

    if not isinstance(prompt_config, dict):
        raise ValueError(f"Prompt file must contain a YAML mapping: {prompt_path}")
    template = prompt_config.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"Prompt file has no non-empty template: {prompt_path}")

    instruction, separator, _ = template.partition(_QUESTION_SECTION_MARKER)
    if not separator:
        raise ValueError(
            f"Prompt template has no {_QUESTION_SECTION_MARKER!r} marker: "
            f"{prompt_path}"
        )
    return instruction.rstrip().replace("{{", "{").replace("}}", "}")


def _normalize_string_list(value: Any) -> tuple[list[str], bool]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        return [], False
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = item.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result, True


class QuestionProcessingRetrievalProgram(dspy.Module):
    """Optimizable question processor followed by fixed HubLink retrieval."""

    def __init__(
        self,
        adapter: HubLinkRawRetrievalAdapter,
        *,
        initial_instruction: str | None = None,
    ):
        super().__init__()
        if initial_instruction is None:
            initial_instruction = load_question_processing_instruction()
        seeded_signature = ProcessQuestion.with_instructions(initial_instruction)
        self.process_question = dspy.Predict(seeded_signature)
        self.adapter = adapter

    def forward(self, question: str) -> dspy.Prediction:
        processed = self.process_question(
            question=question,
            retrieval_contract=SCHEMA_AGNOSTIC_CONTRACT,
        )
        components, components_valid = _normalize_string_list(
            processed.components
        )
        keywords, keywords_valid = _normalize_string_list(processed.keywords)
        valid_output = components_valid and keywords_valid
        if not valid_output:
            return dspy.Prediction(
                components=components,
                keywords=keywords,
                retrieved_source_ids=[],
                retrieved_triples=[],
                valid_output=False,
            )

        retrieval = self.adapter.retrieve(question, components, keywords)
        return dspy.Prediction(
            components=components,
            keywords=keywords,
            retrieved_source_ids=retrieval.source_ids,
            retrieved_triples=retrieval.triples,
            valid_output=True,
        )


class LegacyQuestionProcessingRetrievalProgram(dspy.Module):
    """Current YAML prompt followed by the same raw HubLink retrieval."""

    def __init__(self, adapter: HubLinkRawRetrievalAdapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, question: str) -> dspy.Prediction:
        components, keywords = self.adapter.process_with_legacy_prompt(question)
        retrieval = self.adapter.retrieve(question, components, keywords)
        return dspy.Prediction(
            components=components,
            keywords=keywords,
            retrieved_source_ids=retrieval.source_ids,
            retrieved_triples=retrieval.triples,
            valid_output=True,
        )
