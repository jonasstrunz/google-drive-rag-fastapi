from unittest.mock import MagicMock

from src.app.services.llm import LLMService


def test_llm_service_openai_direct():
    llm = LLMService(provider="openai", openai_api_key="fake_key", openai_model="gpt-4o-mini")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    llm._openai_client = mock_client

    fake_snippet = MagicMock()
    fake_snippet.metadata.file_name = "test.txt"
    fake_snippet.metadata.chunk_index = 0
    fake_snippet.metadata.total_chunks = 1
    fake_snippet.text = "Doc text"

    ans, model = llm.generate_answer("Hello", [fake_snippet])
    assert ans == "OpenAI response"
    assert model == "openai:gpt-4o-mini"


def test_llm_service_openrouter_failover_to_openai():
    llm = LLMService(
        provider="openrouter",
        openrouter_api_key="fake_or_key",
        openai_api_key="fake_oa_key",
    )

    # OpenRouter raises error
    mock_or_client = MagicMock()
    mock_or_client.chat.completions.create.side_effect = Exception("OpenRouter rate limit")
    llm._openrouter_client = mock_or_client

    # OpenAI succeeds
    mock_oa_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Failover OpenAI response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_oa_client.chat.completions.create.return_value = mock_response
    llm._openai_client = mock_oa_client

    fake_snippet = MagicMock()
    fake_snippet.metadata.file_name = "test.txt"
    fake_snippet.metadata.chunk_index = 0
    fake_snippet.metadata.total_chunks = 1
    fake_snippet.text = "Doc text"

    ans, model = llm.generate_answer("Hello", [fake_snippet])
    assert ans == "Failover OpenAI response"
    assert model == "openai:gpt-4o-mini"
