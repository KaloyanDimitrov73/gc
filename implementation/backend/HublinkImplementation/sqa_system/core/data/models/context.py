import hashlib
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class ContextType(Enum):
    """
    A enum that represents the type of context that is being retrieved.
    """
    KG = "knowledge_graph"
    DOC = "document"
    UNKNOWN = "unknown"


class Context(BaseModel):
    """
    This class represents a context object that contains information about a retrieved context 
    from a knowledge graph or document. It is filled and returned by retrievers and is used to
    evaluate the performance of the retriever.
    
    Args:
        uid (str): A unique identifier for the context.
        text (str): The text of the context.
        context_type (ContextType): The type of context (e.g., knowledge graph, document).
        metadata (Optional[dict]): Additional metadata associated with the context.
        score (Optional[float]): A score associated with the context.
    """
    uid: str = Field(default_factory=lambda: str(uuid4()), exclude=True)
    text: str
    context_type: ContextType = ContextType.UNKNOWN
    metadata: Optional[dict] = None
    score: Optional[float] = None

    def __str__(self) -> str:
        """
        Converts the Context object to a string representation.
        """
        return f"[Context: {self.text} \n Metadata: {self.metadata}]".strip()

    def to_document(self) -> Document:
        """
        Converts the current instance of the class to a LangChain Document object.

        Returns:
            Document: A Document object with the page content set to the text of 
                the current instance and the metadata set to a dictionary containing 
                the uid and source_doi of the current instance.
        """
        if self.metadata is None:
            new_metadata = {"context_type": self.context_type.value}
        else:
            new_metadata = self.metadata.copy()
            new_metadata["context_type"] = self.context_type.value

        id_text = self.text + str(new_metadata)
        hash_id = hashlib.md5(id_text.encode()).hexdigest()
        return Document(
            id=hash_id,
            page_content=self.text,
            metadata=new_metadata
        )

    @classmethod
    def from_document(cls, document: Document) -> "Context":
        """
        Creates a new `Context` object from a LangChain `Document` object.

        Args:
            document (Document): The LangChain `Document` object to create the `Context` from.

        Returns:
            Context: The newly created `Context` object.

        """
        context_type = ContextType.UNKNOWN
        if document.metadata.get("context_type") is not None:
            context_type = ContextType(document.metadata["context_type"])
        if document.metadata.get("uid") is None:
            return cls(
                text=document.page_content,
                context_type=context_type,
                metadata=document.metadata
            )

        return cls(
            uid=document.metadata["uid"],
            text=document.page_content,
            context_type=context_type,
            metadata=document.metadata
        )

    @classmethod
    def convert_contexts_to_texts(cls, contexts: List) -> List[str]:
        """
        Converts a list of Context objects into a list of text strings.

        Args:
            Contexts (List[Context]): A list of Context objects to be converted.

        Returns:
            List[str]: A list of text strings created from the input Context.

        """
        texts = []
        for context in contexts:
            texts.append(context.text)
        return texts
