"""
Intell Audio Platform Python SDK.
High-level developer SDK for consuming the audio intelligence pipeline via REST & WebSockets.
"""

from sdk.intell_audio.analyzer import AudioAnalyzer
from sdk.intell_audio.exceptions import (
    APIConnectionError,
    IntellSDKError,
    JobFailedError,
    JobTimeoutError,
)
from schemas.analysis import (
    AnalysisResult,
    ConversationAnalytics,
    DiarizationResult,
    SpeakerProfile,
    TranscriptionResult,
)

__all__ = [
    "AudioAnalyzer",
    "AnalysisResult",
    "SpeakerProfile",
    "TranscriptionResult",
    "DiarizationResult",
    "ConversationAnalytics",
    "IntellSDKError",
    "JobFailedError",
    "JobTimeoutError",
    "APIConnectionError",
]
