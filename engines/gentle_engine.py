"""
Gentle forced-alignment server client implementation.
"""

from pathlib import Path
from typing import Optional

import requests

from config.settings import settings
from engines.base import AlignmentEngine
from schemas.models import AlignmentResult, TranscriptWord
from utils.exceptions import AlignmentError, ServiceUnavailableError
from utils.logger import logger


class GentleAlignmentEngine(AlignmentEngine):
    """Gentle HTTP server client for word-level forced alignment."""

    def __init__(self, gentle_url: Optional[str] = None):
        self.gentle_url = gentle_url or settings.GENTLE_URL

    def is_available(self) -> bool:
        """Check whether Gentle server is reachable."""
        try:
            # Gentle server base endpoint check
            base_url = self.gentle_url.split("/transcriptions")[0]
            res = requests.get(base_url, timeout=3)
            return res.status_code in (200, 404, 405)
        except Exception:
            return False

    def align(self, audio_path: Path, transcript: str) -> AlignmentResult:
        if not audio_path.exists():
            raise AlignmentError(f"Audio file for alignment not found: {audio_path}")

        logger.info(f"Connecting to Gentle forced alignment server at {self.gentle_url}")

        try:
            with open(audio_path, "rb") as audio_file:
                files = {"audio": audio_file}
                data = {"transcript": transcript}

                response = requests.post(
                    self.gentle_url,
                    files=files,
                    data=data,
                    timeout=300,
                )

            if response.status_code != 200:
                logger.error(f"Gentle returned HTTP status {response.status_code}")
                raise AlignmentError(f"Gentle alignment server error (HTTP {response.status_code})")

            raw_data = response.json()
            words: list[TranscriptWord] = []

            for w in raw_data.get("words", []):
                start_val = w.get("start")
                if start_val is None and "startOffset" in w:
                    start_val = w["startOffset"] / 1000.0

                end_val = w.get("end")
                if end_val is None and "endOffset" in w:
                    end_val = w["endOffset"] / 1000.0

                words.append(
                    TranscriptWord(
                        word=w.get("word", "").strip(),
                        start=float(start_val) if start_val is not None else None,
                        end=float(end_val) if end_val is not None else None,
                        confidence=w.get("confidence"),
                    )
                )

            logger.info(f"Gentle alignment completed successfully ({len(words)} aligned words)")
            return AlignmentResult(
                audio_id=audio_path.stem,
                words=words,
                raw_response=raw_data,
            )

        except requests.exceptions.ConnectionError as exc:
            logger.error(f"Failed to connect to Gentle server at {self.gentle_url}")
            raise ServiceUnavailableError(f"Gentle server unreachable at {self.gentle_url}") from exc
        except Exception as exc:
            logger.error(f"Gentle alignment failed: {exc}")
            raise AlignmentError(f"Forced alignment failure: {exc}") from exc
