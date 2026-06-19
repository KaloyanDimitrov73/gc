from typing import List
from pydantic import BaseModel, Field, RootModel
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sqa_system.core.data.models import Triple, Publication
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model.prompt_provider import PromptProvider

from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class ReverseEngineeredText(BaseModel):
    """The reverse engineered text from the document."""
    triple: str = Field(
        default=None,
        description="The triple used to reverse engineer the text.")
    text: str = Field(
        default=None,
        description="The reverse engineered text.")


class ReverseEngineeredTexts(RootModel):
    """The related texts extracted from a document."""
    root: List[ReverseEngineeredText] = Field(
        default_factory=list,
        description="The related texts copied from the document.")


class RelatedTextExtractor:
    """
    Class responsible for extracting related text from the fulltext
    of a publication. This is done using an LLM.

    Args:
        llm_adapter (LLMAdapter): The LLM adapter to be used for extraction.
    """

    def __init__(self,
                 llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()

    def extract_related_texts(self,
                              publication: Publication,
                              triples_to_extract: List[Triple]) -> List[ReverseEngineeredText]:
        """
        Given a publication and a list of triples, this method extracts those
        text chunks from the publication that are related to the triples.
        This extraction is done using an LLM.

        Args:
            publication (Publication): The publication to extract related text from.
            triples_to_extract (List[Triple]): The triples to extract related text for.

        Returns:
            List[ReverseEngineeredText]: A list of related texts extracted from the publication.
        """

        # Get the main prompt for the content extraction
        prompt_provider = PromptProvider()
        prompt_text, _, _ = prompt_provider.get_prompt(
            "qa_generation/related_text_extraction_prompt.yaml")

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=50,
        )
        text_chunks = text_splitter.split_text(str(publication))

        related_texts = self._run_extraction(
            text_chunks,
            triples_to_extract,
            prompt_text
        )
        return related_texts

    def _run_extraction(self,
                        text_chunks: List[str],
                        triples_to_extract: List[Triple],
                        prompt_text: str) -> List[ReverseEngineeredText]:
        """
        This method runs the extraction of related text from the
        text chunks using the LLM. 

        Args:
            text_chunks (List[str]): The text chunks to extract related text from.
            triples_to_extract (List[Triple]): The triples to extract related text for.
            prompt_text (str): The prompt text to use for the extraction.

        Returns:
            List[ReverseEngineeredText]: A list of related texts extracted from the text chunks.
        """
        related_texts = []
        for chunk in text_chunks:
            retries = 3
            attempt = 0
            errors_from_last_call = []
            while attempt < retries:
                try:
                    prompt = PromptTemplate(
                        template=prompt_text,
                        input_variables=["triples", "text"],
                    )

                    chain = prompt | self.llm_adapter.llm.with_structured_output(
                        schema=ReverseEngineeredTexts)
                    result = chain.invoke(
                        {"triples": Triple.convert_list_to_string(triples_to_extract), "text": chunk})
                    break
                except TimeoutError:
                    logger.debug("Timeout error in LLM call, retrying...")
                    attempt += 1
                except Exception as e:
                    logger.debug(f"Error in LLM call: {e}. Retrying...")
                    attempt += 1
                    error_text = str(e).replace("{", "{{").replace("}", "}}")
                    errors_from_last_call.append(str(error_text))
            if attempt >= retries:
                logger.error("Failed to extract related text from chunk.")
                continue

            for value in result.root:
                if value.text and value.text != "" and "None" not in value.text:
                    related_texts.append(value)

        return related_texts
