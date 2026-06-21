import ast
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, ValidationError


class RestrictionType(Enum):
    """
    Enum class to define the types of restrictions that can be applied to a parameter.
    """
    NONE = "none"
    BETWEEN_ONE_AND_ZERO = "between_one_and_zero"
    GREATER_THAN_ZERO = "greater_than_zero"
    GREQ_THAN_MINUS_1 = "greater_or_equal_than_minus_1"
    GREQ_TO_ZERO = "greater_or_equal_to_zero"


class AdditionalConfigParameter(BaseModel):
    """
    This class is used by various components in the SQA system to define additional 
    configuration parameters beyond the base configuration parameters.
    It is an essential part of the configuration system as it allows other components to
    dynamically define and validate their own parameter beyond the base configuration.

    Note:
        - `param_type` is assumed to be a valid Python type or a Pydantic model class.
    """
    name: str = Field(..., description="The name of the parameter that is defined.")
    param_type: Any = Field(
        ..., description="The type of the parameter."
    )
    description: Optional[str] = Field(
        default_factory=str,
        description="A description of the parameter."
    )
    available_values: Optional[List[Any]] = Field(
        default_factory=list,
        description="The available values for the parameter which are enforced."
    )
    default_value: Optional[Any] = Field(
        None,
        description="The default value for the parameter."
    )
    param_restriction: RestrictionType = Field(
        RestrictionType.NONE,
        description=("The restriction that is applied to the parameter by checking whether "
                    "the value satisfies the restriction function associated.")
    )

    def __repr__(self):
        return (
            f"AdditionalConfigParameter(name={self.name}, "
            f"param_type={self.param_type}, "
            f"description={self.description}, "
            f"available_values={self.available_values}, "
            f"default_value={self.default_value}, "
            f"param_restriction={self.param_restriction})"
        )

    def __eq__(self, other):
        return (
            self.name,
            self.param_type,
            self.description,
            self.available_values,
            self.default_value,
            self.param_restriction
        ) == (
            other.name,
            other.param_type,
            other.description,
            other.available_values,
            other.default_value,
            other.param_restriction
        )

    def parse_value(self, value: str) -> Any:
        """
        Validates "value" (provided as a string) to match "param_type". Also checks whether
        the value satisfies any specified restrictions and "available_values" constraint.
        
        This method ensures that the value that is passed to the parameter is of the correct type
        and meets any additional constraints defined by the parameter's configuration.

        Args:
            value (str): The value (as a string) to be parsed and validated.

        Returns:
            Any: The value coerced to the correct type (could be a builtin type or a 
                Pydantic model).

        Raises:
            ValueError: If the value cannot be coerced to the required type or violates 
                restrictions.
        """
        try:
            parsed_value = self._parse_input(value)
        except (ValueError, TypeError, ValidationError) as e:
            raise ValueError(
                f"Cannot parse value='{value}' for parameter '{self.name}' as {self.param_type}: {e}"
            ) from e

        # Check numeric restrictions, if applicable
        if isinstance(parsed_value, (int, float)):
            if self.param_restriction == RestrictionType.BETWEEN_ONE_AND_ZERO:
                if not (0 <= parsed_value <= 1):
                    raise ValueError(
                        f"Value for parameter '{self.name}' must be between 0 and 1; got {parsed_value}"
                    )
            elif self.param_restriction == RestrictionType.GREATER_THAN_ZERO:
                if not (parsed_value > 0):
                    raise ValueError(
                        f"Value for parameter '{self.name}' must be > 0; got {parsed_value}"
                    )
            elif self.param_restriction == RestrictionType.GREQ_THAN_MINUS_1:
                if not (parsed_value >= -1):
                    raise ValueError(
                        f"Value for parameter '{self.name}' must be > -1; got {parsed_value}"
                    )
            elif self.param_restriction == RestrictionType.GREQ_TO_ZERO:
                if not (parsed_value >= 0):
                    raise ValueError(
                        f"Value for parameter '{self.name}' must be >= 0; got {parsed_value}"
                    )

        # Check available_values if specified
        if self.available_values and parsed_value not in self.available_values:
            raise ValueError(
                f"Value for parameter '{self.name}' must be one of {self.available_values}; "
                f"got {parsed_value}"
            )

        return parsed_value

    def _parse_input(self, raw_value: Any) -> Any:
        """
        Parses the input string `raw_value` into the actual Python type `self.param_type`.
        """
        if not isinstance(raw_value, str) and not isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            if raw_value in {"None", "", "none"}:
                return None
        
            # Check if list type
            if raw_value.startswith("[") and raw_value.endswith("]"):
                return self._handle_list(raw_value)
            
            # Check if dict type
            if raw_value.startswith("{") and raw_value.endswith("}"):
                return self._handle_dict(raw_value)

        if isinstance(self.param_type, type) and issubclass(self.param_type, BaseModel):
            return self.param_type(**raw_value)

        # Special handling for booleans
        if self.param_type is bool:
            return self._handle_boolean(raw_value)

        # Otherwise we directly call the type
        return self.param_type(raw_value)
    
    def _handle_list(self, raw_value: str) -> List:
        try:
            return ast.literal_eval(raw_value)
        except Exception:
            return raw_value
        
    def _handle_dict(self, raw_value: str) -> dict:
        try:
            result = ast.literal_eval(raw_value)
            if isinstance(result, dict):
                return result
            return raw_value
        except Exception:
            return raw_value
        
    def _handle_boolean(self, raw_value: str) -> bool:
        lower_val = raw_value.lower()
        if lower_val in {"true", "1", "yes"}:
            return True
        if lower_val in {"false", "0", "no"}:
            return False

        raise ValueError(f"Cannot interpret '{raw_value}' as boolean.")

    @classmethod
    def validate_dict(cls,
                      additional_params: List["AdditionalConfigParameter"],
                      additional_params_dict: dict) -> dict:
        """
        Validates the dictionary of additional parameters against the list of additional parameters.

        Args:
            additional_params (List[AdditionalConfigParameter]): The list of additional parameters.
            additional_params_dict (dict): The dictionary of additional parameters.
            
        Returns:
            dict: A dictionary of validated additional parameters.
        """
        settings = {}
        for param in additional_params:
            config_param_value = additional_params_dict.get(
                param.name, None)
            if config_param_value is None:
                settings[param.name] = param.default_value
            else:
                settings[param.name] = param.parse_value(config_param_value)
        return settings
