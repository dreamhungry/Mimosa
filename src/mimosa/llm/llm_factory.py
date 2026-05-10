# -*- coding: utf-8 -*-
"""LLM factory for creating LLM instances."""

from loguru import logger

from ..config import LLMConfig
from .llm_interface import LLMInterface
from .openai_compatible_llm import OpenAICompatibleLLM


def create_llm(config: LLMConfig) -> LLMInterface:
    """Create an LLM instance based on configuration.

    :param config: LLM configuration.
    :returns: An LLM instance implementing LLMInterface.
    :raises ValueError: If the provider is not supported.
    """
    provider = config.provider.lower()

    if provider in ("openai_compatible", "openai", "deepseek", "gemini", "ollama"):
        llm = OpenAICompatibleLLM(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        logger.info(f"Created LLM with provider: {provider}")
        return llm

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        f"Supported: openai_compatible, openai, deepseek, gemini, ollama"
    )
