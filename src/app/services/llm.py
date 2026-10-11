from typing import Dict, List, Optional

import openai

from src.app.core.config import settings
from src.app.core.logging import logger
from src.app.schemas.rag import SearchResultItem

SYSTEM_PROMPT = """You are an intelligent knowledge assistant that answers questions based on documents retrieved from Google Drive.
Analyze the provided Context Sources carefully and answer the user's question accurately and concisely.

Rules:
1. Ground your answer in the provided Context Sources.
2. If the context does not contain enough information to answer the question, state clearly that the answer cannot be found in the indexed documents.
3. Cite source files by mentioning [File: <filename>] when referring to specific facts.
4. Be clear, direct, and helpful.
"""


class LLMService:
    """Unified LLM service supporting OpenAI, OpenRouter, and fallback extraction."""

    def __init__(
        self,
        provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        openrouter_model: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.openrouter_api_key = openrouter_api_key or settings.OPENROUTER_API_KEY
        self.openai_model = openai_model or settings.OPENAI_CHAT_MODEL
        self.openrouter_model = openrouter_model or settings.OPENROUTER_CHAT_MODEL

        self._openai_client: Optional[openai.OpenAI] = None
        self._openrouter_client: Optional[openai.OpenAI] = None

        self._init_clients()

    def _init_clients(self) -> None:
        if self.openai_api_key:
            self._openai_client = openai.OpenAI(api_key=self.openai_api_key)

        if self.openrouter_api_key:
            default_headers: Dict[str, str] = {}
            if settings.OPENROUTER_HTTP_REFERER:
                default_headers["HTTP-Referer"] = settings.OPENROUTER_HTTP_REFERER
            if settings.OPENROUTER_TITLE:
                default_headers["X-Title"] = settings.OPENROUTER_TITLE

            self._openrouter_client = openai.OpenAI(
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=self.openrouter_api_key,
                default_headers=default_headers if default_headers else None,
            )

    def generate_answer(
        self,
        question: str,
        results: List[SearchResultItem],
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.2,
    ) -> tuple[str, str]:
        """Generates RAG response using configured LLM provider (OpenRouter, OpenAI, or Fallback).

        Returns:
            tuple[answer_text, model_or_provider_used]
        """
        if not results:
            return "No relevant documents found in Google Drive to answer your question.", "none"

        # Build context prompt
        context_parts = []
        for i, item in enumerate(results, 1):
            source_header = (
                f"[Source {i}: File '{item.metadata.file_name}', "
                f"Chunk {item.metadata.chunk_index + 1}/{item.metadata.total_chunks}]"
            )
            context_parts.append(f"{source_header}\n{item.text}")

        context_str = "\n\n---\n\n".join(context_parts)
        user_prompt = f"Context Sources:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"

        # 1. OpenRouter Provider
        if self.provider == "openrouter" or (not self.openai_api_key and self.openrouter_api_key):
            if self._openrouter_client and self.openrouter_api_key:
                try:
                    response = self._openrouter_client.chat.completions.create(
                        model=self.openrouter_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                    )
                    answer = response.choices[0].message.content or ""
                    return answer, f"openrouter:{self.openrouter_model}"
                except Exception as e:
                    logger.error(f"OpenRouter chat completion error: {e}")
                    # Attempt fallback to OpenAI if available
                    if self._openai_client:
                        logger.info("Failing over from OpenRouter to OpenAI")
                        return self._call_openai(user_prompt, system_prompt, temperature)
                    return self._generate_fallback_answer(question, results), "extractive-fallback"

        # 2. OpenAI Provider
        if self._openai_client and self.openai_api_key:
            return self._call_openai(user_prompt, system_prompt, temperature)

        # 3. Fallback extractive answer
        return self._generate_fallback_answer(question, results), "extractive-fallback"

    def _call_openai(self, user_prompt: str, system_prompt: str, temperature: float) -> tuple[str, str]:
        if not self._openai_client:
            return "", "extractive-fallback"
        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            answer = response.choices[0].message.content or ""
            return answer, f"openai:{self.openai_model}"
        except Exception as e:
            logger.error(f"OpenAI chat completion error: {e}")
            return "", "extractive-fallback"

    @staticmethod
    def _generate_fallback_answer(question: str, results: List[SearchResultItem]) -> str:
        """Deterministic extractive response when no live LLM API key is present."""
        top_snippet = results[0]
        return f"Based on [{top_snippet.metadata.file_name}]:\n\n{top_snippet.text}"
