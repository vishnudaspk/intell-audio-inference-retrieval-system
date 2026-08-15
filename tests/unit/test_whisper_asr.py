"""
Unit tests for Whisper ASR Engine — V3 Phase 1J
Tests output structure and mapping logic without loading actual model weights.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.transcription_service import TranscriptionService, _map_language_code
from schemas.enums import LanguageCode


# ── Language mapping ─────────────────────────────────────────────────────────

class TestLanguageMapping:
    def test_english(self):
        assert _map_language_code("en") == LanguageCode.ENGLISH

    def test_arabic(self):
        assert _map_language_code("ar") == LanguageCode.ARABIC

    def test_hindi(self):
        assert _map_language_code("hi") == LanguageCode.HINDI

    def test_malayalam(self):
        assert _map_language_code("ml") == LanguageCode.MALAYALAM

    def test_unknown(self):
        assert _map_language_code("xyz") == LanguageCode.UNKNOWN

    def test_case_insensitive(self):
        assert _map_language_code("EN") == LanguageCode.ENGLISH


# ── Transcription service with mock engine ────────────────────────────────────

class TestTranscriptionService:
    def _make_engine_mock(self):
        engine = MagicMock()
        engine.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "language_probability": 0.99,
            "duration": 3.0,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello world",
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.01,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 1.0, "probability": 0.95},
                        {"word": "world", "start": 1.2, "end": 2.5, "probability": 0.97},
                    ],
                }
            ],
        }
        return engine

    def test_transcribe_audio_returns_transcript(self, tmp_path):
        engine = self._make_engine_mock()
        service = TranscriptionService(engine=engine)
        wav = tmp_path / "test.wav"
        wav.touch()

        transcript = service.transcribe_audio(audio_id="test-123", wav_path=wav)

        assert transcript.audio_id == "test-123"
        assert transcript.text == "Hello world"
        assert transcript.language == LanguageCode.ENGLISH
        assert transcript.duration == 3.0
        assert len(transcript.segments) == 1

    def test_transcribe_audio_segments_have_words(self, tmp_path):
        engine = self._make_engine_mock()
        service = TranscriptionService(engine=engine)
        wav = tmp_path / "test.wav"
        wav.touch()

        transcript = service.transcribe_audio(audio_id="test-123", wav_path=wav)

        seg = transcript.segments[0]
        assert len(seg.words) == 2
        assert seg.words[0].word == "Hello"
        assert seg.words[0].start == 0.0
        assert seg.words[0].end == 1.0
        assert seg.words[0].confidence == pytest.approx(0.95)

    def test_transcribe_audio_flattens_words(self, tmp_path):
        engine = self._make_engine_mock()
        service = TranscriptionService(engine=engine)
        wav = tmp_path / "test.wav"
        wav.touch()

        transcript = service.transcribe_audio(audio_id="test-123", wav_path=wav)

        assert len(transcript.words) == 2

    def test_transcript_id_backfilled_in_segments(self, tmp_path):
        engine = self._make_engine_mock()
        service = TranscriptionService(engine=engine)
        wav = tmp_path / "test.wav"
        wav.touch()

        transcript = service.transcribe_audio(audio_id="test-123", wav_path=wav)

        for seg in transcript.segments:
            assert seg.transcript_id == transcript.id

    def test_transcribe_empty_result(self, tmp_path):
        engine = MagicMock()
        engine.transcribe.return_value = {
            "text": "",
            "language": "en",
            "language_probability": 0.0,
            "duration": 0.0,
            "segments": [],
        }
        service = TranscriptionService(engine=engine)
        wav = tmp_path / "test.wav"
        wav.touch()

        transcript = service.transcribe_audio(audio_id="empty-test", wav_path=wav)

        assert transcript.text == ""
        assert transcript.segments == []
        assert transcript.words == []
