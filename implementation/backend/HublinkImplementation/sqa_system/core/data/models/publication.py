from typing import List, Optional
from pydantic import BaseModel
from langchain_core.documents import Document


class Publication(BaseModel):
    """
    The data object for a publication as managed by the SQA system.
    It includes the metadata of the publication and the full text of the publication.
    """
    doi: str
    title: str
    authors: Optional[List[str]] = None
    url: Optional[str] = None
    month: Optional[int] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    research_field: Optional[str] = None
    full_text: Optional[str] = None
    keywords: Optional[List[str]] = None
    additional_fields: Optional[dict] = None

    def __str__(self) -> str:
        return str(self.to_document())

    def to_document(self) -> Document:
        """
        Converts the publication object to a LangChain Document object.

        Returns:
            Document: The Document object 
        """
        content = f"Doi: {self.doi}\n"
        content += f"Authors: {', '.join(self.authors) if self.authors is not None else ''}\n"
        content += f"Title: {self.title}\n"
        content += f"URL: {self.url}\n"
        content += f"Month: {self.month}\n"
        content += f"Year: {self.year}\n"
        content += f"Venue: {self.venue}\n"
        content += f"Publisher: {self.publisher}\n"
        content += f"Abstract: {self.abstract}\n"
        content += f"Research Field: {self.research_field}\n"
        content += f"Keywords: {', '.join(self.keywords) if self.keywords is not None else ''}\n"
        content += f"Additional Fields: {self.additional_fields}\n"
        content += f"Full Text: {self.full_text}\n"
        return Document(
            page_content=content,
            metadata={
                "doi": self.doi if self.doi is not None else "",
                "authors": ", ".join(self.authors) if self.authors is not None else "",
                "title": self.title if self.title is not None else "",
                "month": self.month if self.month is not None else "",
                "year": self.year if self.year is not None else "",
                "venue": self.venue if self.venue is not None else "",
                "publisher": self.publisher if self.publisher is not None else "",
                "research_field": self.research_field if self.research_field is not None else "",
                "keywords": ", ".join(self.keywords) if self.keywords is not None else "",
            }
        )

    def from_document(self, document: Document) -> "Publication":
        """
        Creates a Publication object from a given LangChain Document object.

        Args:
            document (Document): The Document object containing the 
                publication's metadata and full text.

        Returns:
            Publication: The created Publication object.
        """
        return Publication(
            doi=document.metadata["doi"],
            authors=document.metadata["authors"],
            title=document.metadata["title"],
            year=document.metadata["year"],
            venue=document.metadata["venue"],
            research_field=document.metadata["research_field"],
            full_text=document.page_content,
            month=document.metadata["month"],
            publisher=document.metadata["publisher"],
            keywords=document.metadata["keywords"],
        )
