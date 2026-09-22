"""
filters.py

Backend-agnostic filter representation for vector store queries.

Filters are created as `WhereFilter` objects using helper functions
(e.g. `equals`, `not_in`) and translated into the backend-specific
filter syntax only when the query is executed.

This keeps filter construction independent of the underlying vector
store implementation.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union


class FilterOperator(str, Enum):
    """Comparison operators for a single filter condition."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IS_IN_LIST = "is_in_list"
    IS_NOT_IN_LIST = "is_not_in_list"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"


class LogicalOperator(str, Enum):
    """Logical operators for combining multiple filter conditions.."""
    AND = "and"
    OR = "or"


@dataclass(frozen=True)
class FilterCondition:
    """A single filter condition, such as hub_entity equals ‘abc123’."""
    field: str
    operator: FilterOperator
    value: Any


@dataclass(frozen=True)
class FilterGroup:
    """Combines multiple filter conditions or filter groups using a logical operator."""
    operator: LogicalOperator
    conditions: Tuple["WhereFilter", ...]



WhereFilter = Union[FilterCondition, FilterGroup]