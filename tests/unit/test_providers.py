"""
Unit tests for EmbeddingProvider and LLMProvider with HTTP mocking.
"""

from unittest.mock import MagicMock, patch

from services.embedding_service import LMStudioEmbeddingProvider
from services.llm_service import LMStudioLLMProvider


@patch("requests.post")
def test_embedding_provider(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
            {"index": 1, "embedding": [0.5, 0.6, 0.7, 0.8]},
        ]
    }
    mock_post.return_value = mock_response

    provider = LMStudioEmbeddingProvider(base_url="http://localhost:1234", model_name="Qwen3-Embedding-0.6B")
    embeddings = provider.embed_texts(["first chunk", "second chunk"])

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4]
    assert provider.get_dimension() == 4


@patch("requests.post")
def test_llm_provider(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": '{"answer": "Deployment pipeline redesigned.", "evidence_ids": ["chk_01"], "grounded": true}'}}
        ]
    }
    mock_post.return_value = mock_response

    provider = LMStudioLLMProvider(base_url="http://localhost:1234", model_name="qwen3-8b")
    response_text = provider.generate("What was discussed?", system_prompt="Answer grounded.", json_mode=True)

    assert "Deployment pipeline redesigned" in response_text
    mock_post.assert_called_once()
