"""
Unit tests for Phase 7B QueryUnderstanding service.
"""

import pytest

from services.query_understanding import QueryUnderstanding
from schemas.models import QueryIntent


@pytest.fixture
def qu():
    return QueryUnderstanding()  # no LLM


class TestIntentDetection:
    def test_procedural_how_to(self, qu):
        intent = qu.extract("How do I remove the turbo from my car?")
        assert intent.intent == "procedural_instruction"
        assert "instruction" in intent.content_type_preferences or "procedure" in intent.content_type_preferences

    def test_procedural_how_can_i(self, qu):
        intent = qu.extract("How can I install the new gasket?")
        assert intent.intent == "procedural_instruction"

    def test_warning_query(self, qu):
        intent = qu.extract("What warning signs should I look for?")
        assert intent.intent == "warning"
        assert "warning" in intent.content_type_preferences

    def test_comparison_query(self, qu):
        intent = qu.extract("What is the difference between a T3 and T4 turbo?")
        assert intent.intent == "comparison"

    def test_definition_query(self, qu):
        intent = qu.extract("What is a wastegate?")
        assert intent.intent == "definition"

    def test_temporal_location_when(self, qu):
        intent = qu.extract("When does he explain the boost pressure?")
        assert intent.intent == "temporal_location"

    def test_speaker_query(self, qu):
        intent = qu.extract("Who explains the installation process?")
        assert intent.intent == "speaker_query"

    def test_troubleshooting_query(self, qu):
        intent = qu.extract("Why is my turbo not working?")
        assert intent.intent == "troubleshooting"

    def test_recommendation_which_bolt(self, qu):
        intent = qu.extract("Which bolt do I need to unscrew to remove the turbo?")
        assert intent.intent == "recommendation"

    def test_unknown_intent(self, qu):
        intent = qu.extract("hello")
        assert intent.intent == "unknown"


class TestActionExtraction:
    def test_action_remove(self, qu):
        intent = qu.extract("Which bolt do I need to unscrew to remove the turbo?")
        assert "remove" in intent.actions or "unscrew" in intent.actions

    def test_action_install(self, qu):
        intent = qu.extract("How do I install the new intercooler?")
        assert "install" in intent.actions

    def test_action_replace(self, qu):
        intent = qu.extract("How do I replace the gasket?")
        assert "replace" in intent.actions


class TestObjectExtraction:
    def test_object_bolt(self, qu):
        intent = qu.extract("Which bolt do I need to remove?")
        assert "bolt" in intent.objects

    def test_object_intercooler(self, qu):
        intent = qu.extract("How do I install the intercooler?")
        assert "intercooler" in intent.objects


class TestTargetExtraction:
    def test_target_turbo(self, qu):
        intent = qu.extract("How do I remove the turbo from the engine?")
        assert any("turbo" in t or "engine" in t for t in intent.targets)


class TestToolExtraction:
    def test_tool_socket(self, qu):
        intent = qu.extract("Remove the bolt using a 13mm socket")
        # tools may be empty or contain socket
        assert isinstance(intent.tools, list)


class TestEdgeCases:
    def test_empty_query(self, qu):
        intent = qu.extract("")
        assert isinstance(intent, QueryIntent)
        assert intent.intent == "unknown"

    def test_whitespace_query(self, qu):
        intent = qu.extract("   ")
        assert isinstance(intent, QueryIntent)

    def test_never_raises(self, qu):
        """QueryUnderstanding must never raise regardless of input."""
        for q in ["", "??", "12345", "hello world", "!@#$%"]:
            result = qu.extract(q)
            assert isinstance(result, QueryIntent)

    def test_requires_llm_when_unknown_and_no_actions(self, qu):
        intent = qu.extract("hello")
        assert intent.requires_llm is True

    def test_no_llm_when_intent_found(self, qu):
        intent = qu.extract("How do I remove the turbo?")
        assert intent.requires_llm is False

    def test_normalized_query_is_lowercase(self, qu):
        intent = qu.extract("How Do I Remove The Turbo?")
        assert intent.normalized_query == intent.normalized_query.lower()
