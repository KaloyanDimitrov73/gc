import pytest
from backend.app.modules.guardrails.application.validators import ToxicLanguageValidator

guardrails = pytest.importorskip("guardrails", reason="guardrails-ai not installed")


@pytest.fixture(scope="module")
def validator():
    v = ToxicLanguageValidator()
    if not v.is_available():
        pytest.skip("ToxicLanguage not installed")
    return v


def test_passes_normal_input(validator):
    passed, reason = validator.validate(
        "Who are the authors of the paper with the title "
        "'A Taxonomy of Blockchain-Based Systems for Architecture Design'?"
    )
    assert passed is True
    assert reason == ""


def test_rejects_toxic_input(validator):
    passed, reason = validator.validate("Can't you die?")
    assert passed is False
    assert "Validation failed" in reason


def test_flags_toxic_output(validator):
    passed, reason = validator.validate(
        "If you don't do what I say, I will kill your mother."
    )
    assert passed is False
    assert "Validation failed" in reason
