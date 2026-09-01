"""
Embedding Provider interface and LM Studio OpenAI-compatible implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from config.settings import settings
from utils.exceptions import IntellAudioError
from utils.logger import logger


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate vector embedding for a single search query string."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the embedding provider endpoint and model are ready."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector embedding dimension."""
        pass


class LMStudioEmbeddingProvider(EmbeddingProvider):
    """
    Embedding Provider communicating with LM Studio's OpenAI-compatible /v1/embeddings endpoint.
    Serves models such as Qwen3-Embedding-0.6B-Q8_0.gguf without hard-coding local file paths.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.LM_STUDIO_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.LM_STUDIO_EMBEDDING_MODEL
        self.timeout = timeout or settings.LM_STUDIO_TIMEOUT
        self._cached_dimension: Optional[int] = None

    def _resolve_active_model(self) -> Optional[str]:
        """Auto-detect active embedding model ID from LM Studio /v1/models."""
        try:
            url = f"{self.base_url}/v1/models"
            res = requests.get(url, timeout=5.0)
            if res.status_code == 200:
                models = res.json().get("data", [])
                # Prefer model with 'embedding' in ID
                for m in models:
                    m_id = m.get("id", "")
                    if "embed" in m_id.lower():
                        return m_id
                if models:
                    return models[0].get("id")
        except Exception as exc:
            logger.debug(f"Failed to auto-discover embedding models: {exc}")
        return None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using batching where needed."""
        if not texts:
            return []

        cleaned_texts = [t if t and t.strip() else " " for t in texts]
        url = f"{self.base_url}/v1/embeddings"
        payload = {
            "input": cleaned_texts,
            "model": self.model_name,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)

            # If 400 error (e.g. model mismatch), attempt auto-discovery of loaded embedding model
            if response.status_code == 400:
                active_model = self._resolve_active_model()
                if active_model and active_model != self.model_name:
                    logger.info(f"Auto-discovered LM Studio embedding model '{active_model}'. Retrying.")
                    self.model_name = active_model
                    payload["model"] = active_model
                    response = requests.post(url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                logger.error(f"LM Studio embedding request failed with status {response.status_code}: {response.text}")
                raise IntellAudioError(f"Embedding API error HTTP {response.status_code}")

            data = response.json()
            embeddings_data = data.get("data", [])

            # Sort by index returned in response
            embeddings_data = sorted(embeddings_data, key=lambda x: x.get("index", 0))
            embeddings = [item["embedding"] for item in embeddings_data]

            if embeddings and self._cached_dimension is None:
                self._cached_dimension = len(embeddings[0])

            return embeddings
        except Exception as exc:
            logger.error(f"Failed to generate embeddings from LM Studio at {url}: {exc}")
            raise IntellAudioError(f"Embedding generation failed: {exc}") from exc

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        results = self.embed_texts([query])
        if not results:
            raise IntellAudioError("Embedding service returned empty response for query.")
        return results[0]

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        try:
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("data", [])
                # If model_name specified, check if present or return True if API up
                return True
            return False
        except Exception as exc:
            logger.debug(f"LM Studio embedding health check failed: {exc}")
            return False

    def get_dimension(self) -> int:
        """Get vector dimension, probing endpoint if not already cached."""
        if self._cached_dimension is not None:
            return self._cached_dimension

        try:
            sample_emb = self.embed_query("test probe")
            self._cached_dimension = len(sample_emb)
            return self._cached_dimension
        except Exception:
            return 1024  # Default fallback dimension for Qwen3-Embedding-0.6B
