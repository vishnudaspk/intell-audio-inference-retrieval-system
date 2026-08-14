"""
Factory for creating diarization and speaker turn segmentation engines.
"""

from typing import Optional

from config.settings import settings
from engines.diarization.base import DiarizationEngine
from engines.diarization.heuristic_engine import HeuristicTurnSegmentationEngine
from engines.diarization.null_engine import NullDiarizationEngine


def get_diarization_engine(engine_name: Optional[str] = None) -> DiarizationEngine:
    """Instantiate and return the configured DiarizationEngine."""
    name = (engine_name or settings.DIARIZATION_ENGINE).lower()

    if name == "heuristic":
        return HeuristicTurnSegmentationEngine()
    elif name == "none":
        return NullDiarizationEngine()

    return NullDiarizationEngine()
