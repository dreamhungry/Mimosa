# -*- coding: utf-8 -*-
"""Configuration management for Mimosa."""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel


# Load .env file (won't override existing env vars)
load_dotenv()


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "openai_compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


class ASRConfig(BaseModel):
    """ASR (Automatic Speech Recognition) configuration."""

    engine: str = "sherpa_onnx"
    model_path: str = ""
    model: str = "paraformer-zh"
    device: str = "cuda:0"
    vad_model: str = "fsmn-vad"
    punc_model: str = "ct-punc"
    sample_rate: int = 16000
    cache_dir: str = ""


class TTSConfig(BaseModel):
    """TTS (Text-to-Speech) configuration."""

    engine: str = "edge_tts"
    voice: str = "zh-CN-XiaoxiaoNeural"
    rate: str = "+0%"
    volume: str = "+0%"


class VADConfig(BaseModel):
    """VAD (Voice Activity Detection) configuration."""

    engine: str = "silero"
    threshold: float = 0.5
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 300


class MemoryConfig(BaseModel):
    """Memory and conversation history configuration."""

    extraction_interval: int = 10  # Extract memory every N conversation turns


class CharacterConfig(BaseModel):
    """Character personality configuration."""

    name: str = "Mimosa"
    persona_prompt_path: str = "prompts/persona_mimosa.txt"
    live2d_model_name: str = "mao_pro"


class MimosaConfig(BaseModel):
    """Root configuration model."""

    server: ServerConfig = ServerConfig()
    character: CharacterConfig = CharacterConfig()
    llm: LLMConfig = LLMConfig()
    asr: ASRConfig = ASRConfig()
    tts: TTSConfig = TTSConfig()
    vad: VADConfig = VADConfig()
    memory: MemoryConfig = MemoryConfig()
    cache_dir: str = ""


def _apply_env_overrides(config: MimosaConfig) -> MimosaConfig:
    """Override config fields with environment variables.

    Env var naming convention: MIMOSA_<SECTION>_<FIELD>
    e.g. MIMOSA_LLM_API_KEY, MIMOSA_LLM_BASE_URL, MIMOSA_LLM_MODEL
    """
    env_mappings = {
        "MIMOSA_LLM_API_KEY": ("llm", "api_key"),
        "MIMOSA_LLM_BASE_URL": ("llm", "base_url"),
        "MIMOSA_LLM_MODEL": ("llm", "model"),
        "MIMOSA_LLM_PROVIDER": ("llm", "provider"),
        "MIMOSA_SERVER_HOST": ("server", "host"),
        "MIMOSA_SERVER_PORT": ("server", "port"),
    }

    for env_key, (section, field) in env_mappings.items():
        value = os.environ.get(env_key)
        if value:
            sub_config = getattr(config, section)
            # For int fields, cast the string value
            if field == "port":
                value = int(value)
            setattr(config, section, sub_config.model_copy(update={field: value}))
            logger.debug(f"Env override: {env_key} -> {section}.{field}")

    return config


def load_config(config_path: Optional[str] = None) -> MimosaConfig:
    """Load configuration from YAML file, then apply env var overrides.

    Priority: env vars (.env) > conf.yaml > defaults

    conf.yaml is gitignored. Use conf.yaml.example as template.

    :param config_path: Path to config file. Defaults to 'conf.yaml' in project root.
    :returns: Validated MimosaConfig instance.
    """
    if config_path is None:
        config_path = "conf.yaml"

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {path}, using defaults")
        config = MimosaConfig()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            logger.warning("Config file is empty, using defaults")
            config = MimosaConfig()
        else:
            config = MimosaConfig(**raw)
            logger.info(f"Loaded config from {path}")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    return config
