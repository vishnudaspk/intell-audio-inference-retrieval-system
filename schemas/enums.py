"""
Enums for system entities and processing state machines.
"""

from enum import Enum


class SourceType(str, Enum):
    UPLOAD = "upload"
    YOUTUBE = "youtube"


class JobStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    VALIDATING = "VALIDATING"
    NORMALIZING = "NORMALIZING"
    TRANSCRIBING = "TRANSCRIBING"
    ALIGNING = "ALIGNING"
    RUNNING = "RUNNING"
    PERSISTING = "PERSISTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LanguageCode(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    MALAYALAM = "ml"
    UNKNOWN = "unknown"
