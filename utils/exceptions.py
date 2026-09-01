"""
Custom application-level domain exceptions.
"""


class IntellAudioError(Exception):
    """Base exception class for Intell Audio system."""

    pass


class AudioProcessingError(IntellAudioError):
    """Raised when audio acquisition, validation, or normalization fails."""

    pass


class TranscriptionError(IntellAudioError):
    """Raised when speech recognition/ASR processing fails."""

    pass


class AlignmentError(IntellAudioError):
    """Raised when forced alignment fails or Gentle server is unreachable."""

    pass


class StorageError(IntellAudioError):
    """Raised when database or filesystem persistence operations fail."""

    pass


class RetrievalError(IntellAudioError):
    """Raised when search or transcript retrieval operations fail."""

    pass


class ServiceUnavailableError(IntellAudioError):
    """Raised when a required dependent service (e.g. Gentle) is unavailable."""

    pass
