from typing import List, Any, Optional
from pydantic import BaseModel, Field

class ParameterRange(BaseModel):
    """
    This model allows parameters in a configuration to be 
    defined like a range in the experiments of the pipeline.
    
    This is used for hyperparameter optimization of the pipeline.
    """
    config_name: str = Field(
        ...,
        description="The name of the config where the parameter is located."
    )
    parameter_name: str = Field(
        ...,
        description="The name of the parameter to be optimized."
    )
    values: List[Any] = Field(
        ...,
        description="The values the parameter should be tuned to."
    )
    list_index: Optional[int] = Field(
        None,
        description=("Used when the parameter is a list and the value "
                     "should be tuned at a specific index.")
    )
    dict_key: Optional[str | list[str]] = Field(
        None,
        description=("Used when the parameter is a dictionary and the value "
                     "should be tuned at a specific key.")
    )
