from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from sqa_system.core.data.context_tracer import ContextTracer, ContextTrace
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.knowledge_base.knowledge_graph.storage.utils.graph_converter import GraphConverter
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models import QAPair, Knowledge, Subgraph, Triple
from sqa_system.core.language_model.prompt_provider import PromptProvider
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.extraction.paper_content_extractor import (
    TextWithOriginal,
    Entity
)

from .related_text_extractor import RelatedTextExtractor
from .answer_validator import AnswerValidator
from .llm_qa_generator import LLMQAGenerator, ContextMapping, QAGenerationData
logger = get_logger(__name__)


class SubgraphGeneratorOptions(BaseModel):
    """
    Options for the subgraph generator.
    """
    root_entity: Optional[Knowledge] = Field(
        default=None,
        description="The root entity of the subgraph."
    )
    subgraph: Subgraph = Field(
        ...,
        description="The subgraph to generate questions and answers from."
    )
    template_text: str = Field(
        ...,
        description="The template text to use for the generation."
    )
    additional_requirements: Optional[str] = Field(
        None,
        description="Additional requirements for the generation."
    )
    strategy_name: str = Field(
        None,
        description="The strategy name to use for the generation."
    )
    validate_contexts: bool = Field(
        True,
        description="Whether to validate the contexts."
    )
    classify_qa_pairs: bool = Field(
        True,
        description="Whether to classify the QA pairs according to the question catalog."
    )
    use_traced_context: bool = Field(
        False,
        description="Whether to use the traced context from the original text instead of the triples"
    )
    convert_path_to_text: bool = Field(
        True,
        description=("Whether to convert the extracted paths from the graph to a text representation."
                     " Only relevant if the tracing doesnt find an original text")
    )
    subgraph_size_limit: int | None = Field(
        50,
        description="The maximum number of triples to include in the subgraph."
    )


class FromSubgraphGenerator:
    """
    Generator that generates questions and answers from a subgraph using an LLM.

    Args:
        graph (KnowledgeGraph): The knowledge graph to use for the generation.
        llm_adapter (LLMAdapter): The LLM adapter to use for the generation.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self._prepare_utils()

    def _prepare_utils(self):
        self.prompt_provider = PromptProvider()
        self.graph_converter = GraphConverter(
            graph=self.graph,
            llm_adapter=self.llm_adapter)
        self.progress_handler = ProgressHandler()
        self.answer_validator = AnswerValidator(
            llm_adapter=self.llm_adapter)
        self.document_extractor = RelatedTextExtractor(
            llm_adapter=self.llm_adapter)
        self.llm_qa_generator = LLMQAGenerator(
            graph=self.graph,
            llm_adapter=self.llm_adapter)

    def generate_from_subgraph(
            self,
            options: SubgraphGeneratorOptions) -> Tuple[List[QAPair], List[str]]:
        """
        Generates a question and answer pair from a given subgraph and template 
        using an LLM. The function:
        1. Iterates through every triple in the subgraph
        2. Traces each triple back to the original text
        3. Generates a context mapping from the original text
        4. Runs the LLM generator on this context

        Args:
            options (SubgraphGeneratorOptions): The options for the generation.

        Returns:
            Tuple[List[QAPair], List[str]]: A tuple containing the generated QA pairs
        """

        # Here we limit the context windows to not query too much information
        # to the LLM
        maximum_context_length = options.subgraph_size_limit

        # Limit the subgraph to the maximum context length
        pruned_subgraph = []
        if maximum_context_length and len(options.subgraph.root) > maximum_context_length:
            pruned_subgraph = options.subgraph.root[:maximum_context_length]
        else:
            pruned_subgraph = options.subgraph.root
        pruned_subgraph = Subgraph(root=pruned_subgraph)

        # Trace the subgraph to get the context mapping
        trace_result = self._prepare_subgraph(
            options.subgraph, options.convert_path_to_text, options.use_traced_context)
        index_to_context_mapping = trace_result["index_to_context_mapping"]
        context_texts = trace_result["context_texts"]

        # Create the context mappings for the LLM call
        context_mappings: List[ContextMapping] = self._generate_context_mappings(
            index_to_context_mapping=index_to_context_mapping,
        )

        # Run the LLM generator with the context mappings
        generated_qa_pairs, predicates = self.llm_qa_generator.generate_qa_pairs(
            QAGenerationData(
                context_text="\n".join(context_texts),
                template_text=options.template_text,
                context_mapping=context_mappings,
                topic_entity=options.root_entity,
                additional_requirements=options.additional_requirements,
                strategy_name=options.strategy_name,
                validate_context=options.validate_contexts
            )
        )

        return generated_qa_pairs, predicates

    def _prepare_subgraph(self,
                          subgraph: Subgraph,
                          convert_paths_to_text: bool = True,
                          use_traced_context: bool = False) -> dict:
        """
        Prepares the subgraph for the generation by tracing the triples back to
        the original text and generating a context mapping.
        Args:
            subgraph (Subgraph): The subgraph to prepare.
            convert_paths_to_text (bool): Whether to convert the paths to text.
            use_traced_context (bool): Whether to use the traced context from the
                original text instead of the triples.
        Returns:
            dict: A dictionary containing the index to context mapping and the
                context texts.
        """
        # Get the tracing context from the ContextTracer
        tracing = ContextTracer(
            context_id=f"content_block_{self.graph.config.config_hash}")
        index_to_context_mapping = {}
        context_texts: List[str] = []
        context_to_index_mapping = {}
        index = 0
        for triple in subgraph:
            # Get the tracing for the triple
            triple_trace = None
            context = None
            if use_traced_context:
                context, triple_trace = self._get_triple_trace(
                    tracing=tracing,
                    triple=triple
                )
            if not triple_trace and convert_paths_to_text:
                context = self.graph_converter.path_to_text([triple])
            if not context:
                context = TextWithOriginal(
                    original_text=f"({triple.entity_subject.text}, {triple.predicate}, {triple.entity_object.text})",
                    text=f"({triple.entity_subject.text}, {triple.predicate}, {triple.entity_object.text})"
                )

            paper_entity = self.graph.get_paper_from_entity(
                entity=triple.entity_subject)
            if not paper_entity:
                raise ValueError(
                    f"Could not find paper entity for triple: {triple}")
            # Check if the text is already in the mapping to ensure that we are
            # not adding duplicates to the prompt text
            key = (paper_entity.uid, context.original_text)
            if key not in context_to_index_mapping:
                context_to_index_mapping[key] = index
                index_to_context_mapping[index] = [{
                    "context": context,
                    "triple": triple,
                    "paper_entity": paper_entity,
                    "from_trace": bool(triple_trace)
                }]
                context_texts.append(
                    f"[Context ID {index}. from paper {paper_entity.text}: {context.original_text}]\n")
                index += 1
            else:
                # If already existing, add to the existing context
                i = context_to_index_mapping[key]
                index_to_context_mapping[i].append({
                    "context": context,
                    "triple": triple,
                    "paper_entity": paper_entity,
                    "from_trace": bool(triple_trace)
                })

        return {
            "index_to_context_mapping": index_to_context_mapping,
            "context_texts": context_texts
        }

    def _get_triple_trace(
            self,
            tracing: ContextTracer,
            triple: Triple) -> Tuple[TextWithOriginal | None, ContextTrace | None]:
        """
        Gets the tracing for a given triple from the ContextTracer.
        Args:
            tracing (ContextTracer): The ContextTracer to use.
            triple (Triple): The triple to get the tracing for.
        Returns:
            Tuple[TextWithOriginal | None, ContextTrace | None]: A tuple containing
                the context and the trace for the triple.
        """
        # Get the tracing for the triple
        trace_id = str(triple)
        triple_trace = tracing.get_trace(trace_id)
        context = None
        if not triple_trace:
            return None, None
        # Convert the trace to the pydantic object
        try:
            if 'description' in triple_trace.contexts:
                context = Entity.model_validate(triple_trace.contexts)
            else:
                context = TextWithOriginal.model_validate(
                    triple_trace.contexts)
        except Exception as e:
            logger.error(f"Failed to validate context: {e}")
            return None, None
        return context, triple_trace

    def _generate_context_mappings(self,
                                   index_to_context_mapping: dict) -> List[ContextMapping]:
        """
        Generates the context mappings for the LLM call.

        Args:
            index_to_context_mapping (dict): The index to context mapping.

        Returns:
            List[ContextMapping]: A list of context mappings for the LLM call.
        """
        context_mappings: List[ContextMapping] = []
        for index, contexts in index_to_context_mapping.items():
            context_list = []
            triples_list = []
            source_name = None
            source_id = None
            from_trace = False
            for context in contexts:
                # Because all contexts are from the same paper, we can just
                # take the first one
                if not source_name:
                    source_name = context['paper_entity'].text if context['paper_entity'] else None
                if not source_id:
                    relations = self.graph.get_relations_of_head_entity(
                        context['paper_entity'])
                    for relation in relations:
                        if "doi" in relation.predicate:
                            source_id = relation.entity_object.text
                            break
                if not from_trace:
                    from_trace = context['from_trace']
                ctx = context['context']
                if isinstance(ctx, list):
                    logger.error(
                        f"Expected single instance, but got list: {ctx}")
                    raise ValueError("Expected single instance, but got list")
                context_list.append(context['context'])
                triples_list.append(context['triple'])

            context_mappings.append(ContextMapping(
                context_id=index,
                context=context_list,
                triples=triples_list,
                source_id=source_id,
                source_name=source_name,
                context_from_trace=from_trace
            ))
        return context_mappings
