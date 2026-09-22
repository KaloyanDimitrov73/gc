from typing import List
from pydantic import BaseModel, Field

from sqa_system.core.data.models import Context

class SourceDocumentSummary(BaseModel):
    """
    This is the data object that is used for the Linking process in HubLink.
    
    It contains the information about the source document that a given Hub
    can be linked to.
    """
    source_identifier: str = Field(...,
                                   description="Identifies the source of information")
    source_name: str = Field(..., description="The name of the source.")
    contexts: List[Context] = Field(...,
                                    description="The context of the source document.")