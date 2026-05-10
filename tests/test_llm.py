# -*- coding: utf-8 -*-
"""Basic smoke tests for the LLM module."""

import pytest

from src.mimosa.config import LLMConfig
from src.mimosa.llm.llm_factory import create_llm
from src.mimosa.llm.openai_compatible_llm import OpenAICompatibleLLM


def test_create_llm_openai_compatible():
    """Test LLM factory creates OpenAI-compatible instance."""
    config = LLMConfig(
        provider="openai_compatible",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
    )
    llm = create_llm(config)
    assert isinstance(llm, OpenAICompatibleLLM)


def test_create_llm_deepseek():
    """Test LLM factory handles deepseek provider."""
    config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="test-key",
    )
    llm = create_llm(config)
    assert isinstance(llm, OpenAICompatibleLLM)


def test_create_llm_ollama():
    """Test LLM factory handles ollama provider."""
    config = LLMConfig(
        provider="ollama",
        model="llama3",
        base_url="http://localhost:11434/v1",
        api_key="",
    )
    llm = create_llm(config)
    assert isinstance(llm, OpenAICompatibleLLM)


def test_create_llm_unsupported():
    """Test LLM factory raises for unsupported provider."""
    config = LLMConfig(
        provider="unsupported_provider",
        model="test",
        base_url="http://localhost",
    )
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm(config)


def test_config_load():
    """Test config loading."""
    from src.mimosa.config import MimosaConfig, load_config

    config = MimosaConfig()
    assert config.server.port == 8000
    assert config.llm.provider == "openai_compatible"
    assert config.character.name == "Mimosa"
