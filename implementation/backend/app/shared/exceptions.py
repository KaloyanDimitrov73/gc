"""
Shared application exception hierarchy.

These base types carry no module-specific assumptions and may be caught
by cross-cutting handlers (e.g. the global handler in main.py).
"""


class AppError(RuntimeError):
    """Base for all application-level exceptions."""


class InputRejectedError(AppError):
    """Input was rejected by a validation or guardrails check. Maps to HTTP 422."""


class ServiceUnavailableError(AppError):
    """A required service dependency is unavailable. Maps to HTTP 503."""
