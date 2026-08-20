"""
AudioAnalyzer client implementation for the Python SDK.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import requests

from sdk.intell_audio.exceptions import (
    APIConnectionError,
    IntellSDKError,
    JobFailedError,
    JobTimeoutError,
)
from schemas.analysis import AnalysisResult


class AudioAnalyzer:
    """
    Developer-facing client for submitting audio files and retrieving structured intelligence results.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit_file(self, file_path: Union[str, Path]) -> str:
        """Submit an audio file for asynchronous analysis and return the job_id."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        url = f"{self.base_url}/api/v1/analyze"
        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "audio/wav")}
                res = requests.post(url, files=files, timeout=30.0)
            if res.status_code != 202:
                raise IntellSDKError(f"Submission failed ({res.status_code}): {res.text}")
            return res.json()["job_id"]
        except requests.RequestException as e:
            raise APIConnectionError(f"Failed to connect to Intell API at {url}: {e}") from e

    def submit_url(self, youtube_url: str) -> str:
        """Submit a YouTube/media URL for asynchronous analysis and return the job_id."""
        url = f"{self.base_url}/api/v1/analyze"
        try:
            res = requests.post(url, params={"url": youtube_url}, timeout=30.0)
            if res.status_code != 202:
                raise IntellSDKError(f"Submission failed ({res.status_code}): {res.text}")
            return res.json()["job_id"]
        except requests.RequestException as e:
            raise APIConnectionError(f"Failed to connect to Intell API at {url}: {e}") from e

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Check the status and stage timing breakdown of a job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/status"
        try:
            res = requests.get(url, timeout=10.0)
            if res.status_code == 404:
                raise IntellSDKError(f"Job not found: {job_id}")
            return res.json()
        except requests.RequestException as e:
            raise APIConnectionError(f"Failed to fetch job status: {e}") from e

    def get_result(self, job_id: str) -> Optional[AnalysisResult]:
        """Fetch the completed AnalysisResult or None if still running."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        try:
            res = requests.get(url, timeout=30.0)
            if res.status_code == 200:
                return AnalysisResult.model_validate(res.json())
            elif res.status_code == 202:
                return None
            elif res.status_code == 404:
                raise IntellSDKError(f"Job not found: {job_id}")
            else:
                raise JobFailedError(f"Job result request failed ({res.status_code}): {res.text}")
        except requests.RequestException as e:
            raise APIConnectionError(f"Failed to fetch job result: {e}") from e

    def process(
        self,
        path_or_url: Union[str, Path],
        poll_interval_sec: float = 1.0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> AnalysisResult:
        """
        Synchronously submit and wait for full audio intelligence result.
        Polls backend status and invokes on_progress callback if provided.
        """
        str_val = str(path_or_url)
        if str_val.startswith("http://") or str_val.startswith("https://"):
            job_id = self.submit_url(str_val)
        else:
            job_id = self.submit_file(str_val)

        t_start = time.time()
        while time.time() - t_start < self.timeout:
            status_info = self.get_status(job_id)
            if on_progress:
                on_progress(status_info)

            st = status_info.get("status")
            if st == "COMPLETED":
                result = self.get_result(job_id)
                if result:
                    return result
            elif st == "FAILED":
                err = status_info.get("error_message", "Unknown error")
                raise JobFailedError(f"Processing failed on server: {err}")

            time.sleep(poll_interval_sec)

        raise JobTimeoutError(f"Processing exceeded timeout of {self.timeout}s for job {job_id}")
