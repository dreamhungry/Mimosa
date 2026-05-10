# -*- coding: utf-8 -*-
"""OpenAI-compatible LLM implementation.

Supports OpenAI, DeepSeek, Gemini, Ollama, and any API following the OpenAI format.
"""

from typing import AsyncIterator, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

from .llm_interface import LLMInterface


class OpenAICompatibleLLM(LLMInterface):
    """LLM implementation using OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """Initialize the OpenAI-compatible LLM client.

        :param model: Model name (e.g., 'gpt-4o-mini', 'deepseek-chat').
        :param base_url: API base URL.
        :param api_key: API key for authentication.
        :param temperature: Sampling temperature.
        :param max_tokens: Maximum tokens in response.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use "sk-no-key" placeholder for local models without auth
        effective_key = api_key if api_key else "sk-no-key"

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=effective_key,
        )
        logger.info(f"Initialized LLM: model={model}, base_url={base_url}")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        :param messages: Conversation messages.
        :param system: Optional system prompt.
        :yields: Text chunks from LLM response.
        """
        full_messages = []

        if system:
            full_messages.append({"role": "system", "content": system})

        full_messages.extend(messages)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            yield f"[Error: LLM request failed - {e}]"
