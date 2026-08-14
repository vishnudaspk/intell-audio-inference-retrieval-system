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
    SEGMENTING = "SEGMENTING"
    ANALYZING = "ANALYZING"
    CHAPTERING = "CHAPTERING"
    PERSISTING = "PERSISTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ContentType(str, Enum):
    introduction = "introduction"
    explanation = "explanation"
    instruction = "instruction"
    procedure = "procedure"
    demonstration = "demonstration"
    definition = "definition"
    comparison = "comparison"
    question = "question"
    answer = "answer"
    warning = "warning"
    recommendation = "recommendation"
    troubleshooting = "troubleshooting"
    summary = "summary"
    discussion = "discussion"
    conclusion = "conclusion"
    unknown = "unknown"


class LanguageCode(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar"
    HINDI = "hi"
    MALAYALAM = "ml"
    UNKNOWN = "unknown"

