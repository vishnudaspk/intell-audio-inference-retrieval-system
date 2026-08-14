"""
Unit tests for Content Semantic Analyzer.
"""

import json
from unittest.mock import MagicMock
import pytest

from schemas.models import TranscriptChunk
from services.content_analyzer import ContentAnalyzer


def test_content_analyzer_disabled_by_default():
    analyzer = ContentAnalyzer()
    analyzer.enabled = False

    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Remove the two 13mm bolts underneath the turbo housing.",
            start_time=0.0,
            end_time=5.0,
        )
    ]
    analyzed = analyzer.analyze_chunks(chunks)
    assert analyzed[0].content_type is None
    assert analyzed[0].actions == []


def test_content_analyzer_successful_batch_parsing():
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True

    response_payload = [
        {
            "chunk_id": "chk_0001",
            "content_type": "instruction",
            "intent": "remove_component",
            "topic": "turbo removal",
            "actions": ["remove", "unscrew"],
            "objects": ["bolts"],
            "targets": ["turbo housing"],
            "tools": ["13mm socket"],
            "quantities": ["two", "13mm"],
            "procedure_step": 1,
            "chunk_summary": "Remove the two 13mm bolts under the turbo housing.",
        }
    ]
    mock_llm.generate.return_value = json.dumps(response_payload)

    analyzer = ContentAnalyzer(llm_provider=mock_llm)
    analyzer.enabled = True

    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Remove the two 13mm bolts underneath the turbo housing.",
            start_time=0.0,
            end_time=5.0,
        )
    ]

    analyzed = analyzer.analyze_chunks(chunks)
    c = analyzed[0]
    assert c.content_type == "instruction"
    assert c.intent == "remove_component"
    assert c.topic == "turbo removal"
    assert "remove" in c.actions
    assert "bolts" in c.objects
    assert "turbo housing" in c.targets
    assert "13mm socket" in c.tools
    assert "two" in c.quantities
    assert c.procedure_step == 1
    assert c.chunk_summary == "Remove the two 13mm bolts under the turbo housing."


def test_content_analyzer_llm_failure_degrades_gracefully():
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.side_effect = Exception("LM Studio connection timeout")

    analyzer = ContentAnalyzer(llm_provider=mock_llm)
    analyzer.enabled = True
    analyzer.max_retries = 1

    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Remove the two 13mm bolts.",
            start_time=0.0,
            end_time=5.0,
        )
    ]

    # Must not raise; chunks returned unmodified
    analyzed = analyzer.analyze_chunks(chunks)
    assert analyzed[0].content_type is None
    assert analyzed[0].actions == []
