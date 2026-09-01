"""
LLM Provider interface and LM Studio OpenAI-compatible implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from config.settings import settings
from utils.exceptions import IntellAudioError
from utils.logger import logger


class LLMProvider(ABC):
    """Abstract interface for LLM generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        """Generate text or JSON completion for prompt and optional system prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM provider endpoint is reachable."""
        pass


class LMStudioLLMProvider(LLMProvider):
    """
    LLM Provider communicating with LM Studio's OpenAI-compatible /v1/chat/completions endpoint.
    Serves local LLM models such as Qwen3-8B-Q4_K_M.gguf.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.LM_STUDIO_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.LM_STUDIO_CHAT_MODEL
        self.timeout = timeout or settings.LM_STUDIO_TIMEOUT

    def _resolve_active_model(self) -> Optional[str]:
        """Auto-detect active LLM model ID from LM Studio /v1/models."""
        try:
            url = f"{self.base_url}/v1/models"
            res = requests.get(url, timeout=5.0)
            if res.status_code == 200:
                models = res.json().get("data", [])
                # Prefer model without 'embed' in ID
                for m in models:
                    m_id = m.get("id", "")
                    if "embed" not in m_id.lower():
                        return m_id
                if models:
                    return models[0].get("id")
        except Exception as exc:
            logger.debug(f"Failed to auto-discover LLM models: {exc}")
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        """Execute chat completion request against LM Studio API."""
        url = f"{self.base_url}/v1/chat/completions"

        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_rag_output",
                    "strict": "true",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "grounded": {"type": "boolean"},
                        },
                        "required": ["answer", "evidence_ids", "grounded"],
                    },
                },
            }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)

            # If 400 error, attempt auto-discovery of loaded LLM model
            if response.status_code == 400:
                active_model = self._resolve_active_model()
                if active_model and active_model != self.model_name:
                    logger.info(f"Auto-discovered LM Studio LLM model '{active_model}'. Retrying.")
                    self.model_name = active_model
                    payload["model"] = active_model
                    response = requests.post(url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                logger.error(f"LM Studio LLM request failed with status {response.status_code}: {response.text}")
                raise IntellAudioError(f"LLM API error HTTP {response.status_code}")

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise IntellAudioError("LM Studio returned response with empty choices.")

            content = choices[0].get("message", {}).get("content", "")
            return content.strip()

        except Exception as exc:
            logger.error(f"LLM generation request failed at {url}: {exc}")
            raise IntellAudioError(f"LLM generation failed: {exc}") from exc

    def is_available(self) -> bool:
        """Check if LM Studio chat completions endpoint is reachable."""
        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception as exc:
            logger.debug(f"LM Studio LLM health check failed: {exc}")
            return False
