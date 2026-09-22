from pydantic import BaseModel, Field

class LLMStats(BaseModel):
    """Saves stats of an LLM execution"""
    total_tokens: int = Field(default=0,
                              description="Total number of tokens used in the LLM execution")
    prompt_tokens: int = Field(default=0,
                               description="Number of tokens used in the prompt")
    completion_tokens: int = Field(default=0,
                                    description="Number of tokens used in the completion")
    cost: float = Field(default=0.0,
                        description="Cost associated with the LLM execution")
