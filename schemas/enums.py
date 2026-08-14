"""
Enums for system entities and processing state machines.
"""

from enum import Enum


class SourceType(str, Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class JobStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    NORMALIZING = "NORMALIZING"
    TRANSCRIBING = "TRANSCRIBING"
    ALIGNING = "ALIGNING"
    PERSISTING = "PERSISTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class LanguageCode(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    MALAYALAM = "ml"
    UNKNOWN = "unknown"
