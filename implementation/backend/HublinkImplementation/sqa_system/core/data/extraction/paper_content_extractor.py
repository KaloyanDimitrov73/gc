from typing import List, Optional, Union
import json
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    PydanticOutputParser
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.text_splitter import TokenTextSplitter
from pydantic import BaseModel, Field

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.config.models.llm_config import LLMConfig
from sqa_system.core.language_model.prompt_provider import PromptProvider
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)

# The following classes are special pydantic models that we are using for the
# extraction. They are prepared in such a way that when given the LLM, it
# is understandable for the LLM. The LLM fills the content of these models
# and returns it.


class Entity(BaseModel):
    """A representation of any information entity."""
    name: str = Field(..., description="The name of the entity.")
    description: str = Field(..., description="A description of the entity.")
    original_text: str = Field(
        ...,
        description=("The original sentences or texts from which the concept "
                     "information was extracted. (Exact copies)"))


class TextWithOriginal(BaseModel):
    """A representation of a text with its original form."""
    text: str = Field(..., description="The prepared text.")
    original_text: str = Field(
        ..., description=("The original sentences or texts from which the concept "
                          "information was extracted. (Exact copies)"))


class PaperContent(BaseModel):
    """A summary of the content of a research paper."""

    research_problems: Optional[List[TextWithOriginal]] = Field(
        default_factory=list,
        description=("The main research problems formulated by the authors"
                     "that the paper is trying to solve or address."))
    research_questions: Optional[List[TextWithOriginal]] = Field(
        default_factory=list,
        description=("The exact (word-by-word) research questions that are stated by "
                     "the authors. Empty if the authors did not state any."))
    background_concepts: Optional[List[Entity]] = Field(
        default_factory=list,
        description=("The concepts that are needed to understand the paper and are "
                     "discussed in a 'Background' or 'Fundamentals' section."))
    contributions: Optional[List[Entity]] = Field(
        default_factory=list,
        description=("The key contribution(s) stated by the authors. "
                     "Only include the most important contributions "
                     "from the paper. Do not list more "
                     "than 3 distinct contributions. If you think that a "
                     "contribution is more important than another, remove "
                     "the less important one."))


class PaperContentExtractor:
    """
    The PaperContentExtractor class is responsible for extracting structured content
    from research papers using an LLM.
    This content represents a structured summary of the paper.

    Args:
        llm_config (LLMConfig): The configuration for the language model to be used.
    """

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.prompt_provider = PromptProvider()
        self.llm = self._prepare_llm_runnable(llm_config)
        self.paper_content_cache = self._load_paper_content_cache()

    def _load_paper_content_cache(self) -> dict:
        """
        The extraction is stored in a cache file. This function loads the cache
        from the file if it exists. The cache is a dictionary that maps the
        publication DOI to the extracted content and stored in a JSON file.

        Returns:
            dict: A dictionary containing the cached paper content.
        """
        file_path_manager = FilePathManager()
        cache_path = file_path_manager.get_path("paper_extraction_cache.json")
        if file_path_manager.file_path_exists(cache_path):
            logger.debug("Loading paper content cache from %s", cache_path)
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    def _save_paper_content_cache(self):
        """
        Saves the paper content cache to a JSON file.
        """
        file_path_manager = FilePathManager()
        cache_path = file_path_manager.get_path("paper_extraction_cache.json")
        file_path_manager.ensure_dir_exists(cache_path)
        logger.debug("Saving paper content cache to %s", cache_path)
        with open(cache_path, "w", encoding="utf-8") as file:
            json.dump(self.paper_content_cache, file, indent=2)

    def extract_paper_content(self,
                              publication: Publication,
                              update_cache: bool = False,
                              context_size: int = 4000,
                              chunk_repetitions: int = 2) -> PaperContent | None:
        """
        Extracts structured content from a paper text by processing it in chunks.

        This function splits the input text into manageable chunks and processes each chunk
        using a language model to extract structured information about the paper. The results
        from each chunk are aggregated into a paper content object.

        Args:
            publication (Publication): The publication object containing the paper text
                and metadata.
            update_cache (bool, optional): Whether to update the cache with the extracted
                paper content. If set to True, no extracted content will be used instead it 
                will extract again and store the new content in the cache.
            context_size (int, optional): The size of the text chunks to process. 
            chunk_repetitions (int, optional): The number of times to repeat the processing
                for each chunk to improve accuracy.

        Returns:
            Union[None, PaperContent]: A PaperContent object containing the structured information
            extracted from the paper, or None if processing failed.
        """
        if not publication.full_text:
            logger.error("Publication does not contain full text.")
            return None

        if not update_cache:
            cached_paper_content = self.paper_content_cache.get(
                f"paper_content_{self.llm_config.config_hash}_{publication.doi}",
            )
            if cached_paper_content is not None:
                return PaperContent.model_validate_json(cached_paper_content)

        # Split text into manageable chunks
        text_splitter = TokenTextSplitter(
            chunk_size=context_size,
            chunk_overlap=0,
        )

        logger.debug((
            f"Running content extraction for publication {publication.doi}"
            f" with context size {context_size}"))

        text_chunks = text_splitter.split_text(publication.full_text)
        aggregated_content = self._run_extraction(
            text_chunks, chunk_repetitions)

        if aggregated_content is None:
            return None

        # Validate the extracted paper content
        aggregated_content = self._validate_paper_content(
            aggregated_content
        )

        self._log_paper_content_extraction(aggregated_content)

        # If we successfully extracted the paper content, we cache it
        self.paper_content_cache[
            f"paper_content_{self.llm_config.config_hash}_{publication.doi}"] = aggregated_content.model_dump_json(indent=2)
        self._save_paper_content_cache()

        return aggregated_content

    def _run_extraction(self,
                        text_chunks: List[str],
                        chunk_repetitions: int) -> Union[None, PaperContent]:
        """
        The main function that runs the content extraction process.
        It processes each chunk of text and aggregates the results into a single
        PaperContent object.

        Args:
            text_chunks (List[str]): A list of text chunks to process.
            chunk_repetitions (int): The number of times to repeat the processing
                for each chunk to improve accuracy.

        Returns:
            Union[None, PaperContent]: A PaperContent object containing the structured information
            extracted from the paper, or None if processing failed.
        """
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "extraction/paper_content_extraction_prompt.yaml")

        messages = [SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=prompt_text,
                input_variables=[]
            ))]
        aggregated_content = None
        for index, text_chunk in enumerate(text_chunks):
            repeats = chunk_repetitions
            while repeats > 0:
                chunk_messages = self._prepare_chunk_messages(
                    messages, index, aggregated_content
                )
                logger.debug(
                    (f"Running LLM for content extraction on chunk {index} "
                        f"with repetitions remaining: {repeats - 1}"))
                aggregated_content = self._run_chunk_extraction(
                    chunk_messages, text_chunk, index, aggregated_content)

                repeats -= 1
        return aggregated_content

    def _run_chunk_extraction(self,
                              chunk_messages: List,
                              text_chunk: str,
                              index: int,
                              aggregated_content: PaperContent) -> Union[None, PaperContent]:
        """
        The function that runs the extraction process for a single chunk of text.

        Args:
            chunk_messages (List): The messages to send to the LLM for processing.
            text_chunk (str): The text chunk to be processed.
            index (int): The index of the current chunk.
            aggregated_content (PaperContent): The aggregated content extracted so far.

        Returns:
            Union[None, PaperContent]: The updated aggregated content or None if processing failed.
        """
        retries = 3
        prompt = ChatPromptTemplate.from_messages(chunk_messages)
        chain = prompt | self.llm.with_structured_output(schema=PaperContent)
        error_text = ""
        while retries > 0:
            try:
                logger.debug("Asking LLM for chunk content...")
                chunk_content = chain.invoke(
                    {"text_chunk": text_chunk, "index": index})

                if chunk_content is None:
                    raise ValueError("LLM response is None")
                if not isinstance(chunk_content, PaperContent):
                    raise ValueError("Invalid response received from LLM.")

                aggregated_content = self._aggregate_content(
                    chunk_content, aggregated_content)
                break

            except TimeoutError:
                logger.debug("Timeout error in LLM call, retrying...")
            except OutputParserException as e:
                logger.debug("Error in LLM call:. Trying to fix..")
                if "chunk_content" in locals() and chunk_content:
                    fixed_content = self._correct_pydantic_format(
                        error_text=str(e),
                        wrong_format=chunk_content
                    )
                else:
                    fixed_content = None
                if fixed_content is not None:
                    aggregated_content = self._aggregate_content(
                        fixed_content, aggregated_content)
                    break
                logger.debug(f"Failed to fix error: {e}, Retrying...")
                error_text = e
            except Exception as e:
                logger.error(
                    f"Failed to process chunk {index}: {e}")
                error_text = e
            retries -= 1
            error_text = str(error_text).replace("{", "{{").replace("}", "}}")
            prompt.append(AIMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=f"**Error**: {error_text}",
                    input_variables=[]
                )
            ))
            prompt.append(
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        template=("In your last call you made an error make "
                                  "sure that you avoid it this time"),
                        input_variables=[]
                    )
                )
            )

        if retries == 0:
            logger.warning(
                f"Failed to process chunk {index} after {retries} attempts. Skipping..")
            return None
        return aggregated_content

    def _correct_pydantic_format(self, error_text: str, wrong_format: str) -> PaperContent:
        """
        This function tries to correct the pydantic format of the error text
        by using the LLM.

        Args:
            error_text (str): The error text to be corrected.
            wrong_format (str): The wrong format that needs to be corrected.

        Returns:
            PaperContent: The corrected PaperContent object. 
        """
        logger.debug("Trying to correct the pydantic format of the error text: %s",
                     error_text)
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "extraction/correct_wrong_output_format_prompt.yaml")
        parser = PydanticOutputParser(pydantic_object=PaperContent)

        try:
            prompt = PromptTemplate(
                template=(prompt_text),
                input_variables=["json_structure",
                                 "error_text", "invalid_json"]
            )

            chain = prompt | self.llm | parser
            response = chain.invoke(
                {
                    "json_structure": parser.get_format_instructions(),
                    "error_text": error_text,
                    "invalid_json": wrong_format
                }
            )
            logger.debug("Corrected Response form LLM: %s", str(response))
        except Exception as e:
            logger.error(
                f"Failed to correct the pydantic format of the error text: {e}"
            )
            return None
        return response

    def _validate_paper_content(self,
                                paper_content: PaperContent):
        """
        Validates the extracted paper content to ensure that it contains the necessary
        information and that the extracted entities are valid.

        Args:
            paper_content (PaperContent): The extracted paper content to be validated.
        """
        logger.debug("Validating extracted paper content...")
        prompt_text, _, _ = self.prompt_provider.get_prompt(
            "extraction/paper_content_validation_prompt.yaml")

        max_retries = 3
        errors = []
        for attempt in range(max_retries):
            try:
                template = prompt_text
                if errors:
                    template += ("\n\n In your last call you made the following errors. "
                                 "Make sure you fix them this time:" + "\n".join(errors))
                prompt = PromptTemplate(
                    template=("Task Description: " + prompt_text),
                    input_variables=["question", "contexts"],
                )

                chain = prompt | self.llm.with_structured_output(
                    schema=PaperContent)
                response = chain.invoke(
                    {
                        "paper_content": paper_content,
                    }
                )

                if not isinstance(response, PaperContent):
                    raise ValueError(
                        "Invalid response received, expected PaperContent.")

                logger.debug("Successfully validated paper content.")
                return response
            except Exception as e:
                logger.error(
                    "Failed to get final answer from generation "
                    f"pipe {attempt + 1} failed: {e}"
                )
                errors.append(str(e).replace("{", "").replace("}", ""))
                if attempt == max_retries - 1:
                    return paper_content
        return paper_content

    def _log_paper_content_extraction(self, paper_content: PaperContent):
        """
        Logs the extracted paper content for debugging purposes.

        Args:
            paper_content (PaperContent): The extracted paper content to be logged.
        """
        logger.debug("Extracted Paper Content:")
        logger.debug(paper_content.model_dump_json(indent=2))
        logger.debug("Amount of extracted contents")
        # pylint: disable=not-an-iterable
        for field in PaperContent.model_fields:
            logger.debug(f"{field}: {len(getattr(paper_content, field))}")

    def _prepare_llm_runnable(self, llm_config: LLMConfig):
        """
        Prepares the LLM runnable based on the provided configuration.
        This function retrieves the LLM adapter and ensures that the LLM is
        initialized correctly.

        Args:
            llm_config (LLMConfig): The configuration for the language model to be used.
        """
        llm_adapter = LLMProvider().get_llm_adapter(llm_config)
        llm_runnable = llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        return llm_runnable

    def _prepare_chunk_messages(self,
                                messages: List,
                                index: int,
                                aggregated_content: PaperContent) -> List:
        """
        This function creates a new message for the LLM that includes the
        current chunk of text and the aggregated content.

        Args:
            messages (List): The list of messages to be sent to the LLM.
            index (int): The index of the current chunk.
            aggregated_content (PaperContent): The aggregated content extracted so far.

        Returns:
            List: The updated list of messages to be sent to the LLM.
        """
        chunk_messages = messages.copy()
        if index == 0:
            human_message_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(
                template="**Paper text part {index}**:\n {text_chunk}\n",
                input_variables=["text_chunk", "index"]
            ))
            chunk_messages.append(human_message_prompt)
        else:
            existing_structure = aggregated_content.model_dump_json(indent=2).replace(
                "{", "").replace("}", "")
            human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=("This is a continuation. Process the next part of the paper "
                              "and add any new information to the existing structure. \n The "
                              f"existing structure is: \n {existing_structure} \n"
                              "**The {index} part of the paper is**: \n \n {text_chunk}\n"),
                    input_variables=["text_chunk", "index"]
                )
            )
            chunk_messages.append(human_message_prompt)
        return chunk_messages

    def _aggregate_content(self,
                           chunk_content: PaperContent,
                           aggregated_content: PaperContent = None) -> PaperContent:
        """
        In this function the paper content for each chunk is aggregated into a single
        paper content object. This is done by merging the content of each chunk into
        the aggregated content object.

        Args:
            chunk_content (PaperContent): The content extracted from the current chunk.
            aggregated_content (PaperContent, optional): The aggregated content extracted so far.
                If None, a new PaperContent object will be created.
        Returns:
            PaperContent: The aggregated content object containing the merged information.
        """
        if aggregated_content is None:
            return chunk_content

        # If the field is empty we add it to the aggregated content
        # if the field is not empty we merge it but only if it is a list
        # else we keep the existing value
        # pylint: disable=not-an-iterable
        for field in PaperContent.model_fields:
            current_value = getattr(chunk_content, field)
            if not current_value:
                continue

            aggregated_value = getattr(aggregated_content, field)
            if aggregated_value is None:
                setattr(aggregated_content, field, current_value)
                continue

            if not isinstance(current_value, list):
                continue

            existing = aggregated_value
            already_exists = False
            for existing_value in existing:
                if isinstance(existing_value, Entity):
                    if current_value[0].name == existing_value.name:
                        already_exists = True
                        break
            if already_exists:
                continue

            existing.extend(
                [x for x in current_value if x not in existing])
        return aggregated_content
