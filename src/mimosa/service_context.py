# -*- coding: utf-8 -*-
"""Service context container for per-session state.

Composes AgentCore (pure text) with media services (ASR/TTS/VAD/Live2D).
"""

import os
from pathlib import Path

from loguru import logger

from .asr.asr_factory import create_asr
from .asr.asr_interface import ASRInterface
from .config import MimosaConfig
from .conversation.interaction_phrase_pool import InteractionPhrasePool
from .core import AgentCore
from .live2d.live2d_model import Live2DModel
from .tts.tts_factory import create_tts
from .tts.tts_interface import TTSInterface
from .vad.vad_factory import create_vad
from .vad.vad_interface import VADInterface


def _setup_global_cache_dir(config: MimosaConfig):
    """Set global cache directories for all model downloads.

    This must be called before any model loading to redirect cache
    from default system locations (C drive) to user-specified path.
    """
    # Use global cache_dir, fallback to asr.cache_dir for backward compat
    cache_dir = config.cache_dir or config.asr.cache_dir
    if not cache_dir:
        return

    cache_path = os.path.abspath(cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    # ModelScope cache (FunASR model downloads)
    os.environ.setdefault("MODELSCOPE_CACHE", cache_path)
    # HuggingFace cache
    os.environ.setdefault("HF_HOME", os.path.join(cache_path, "huggingface"))
    # PyTorch hub cache (Silero VAD downloads)
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_path, "torch"))

    logger.info(f"Global model cache directory: {cache_path}")


def _load_persona_prompt(path: str) -> str:
    """Load persona prompt from file.

    :param path: Path to persona prompt file.
    :returns: Prompt text content.
    """
    file_path = Path(path)
    if not file_path.exists():
        logger.warning(f"Persona prompt not found: {path}, using default")
        return "You are Mimosa, a friendly virtual companion."

    with open(file_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    logger.info(f"Loaded persona prompt from: {path}")
    return prompt


class ServiceContext:
    """Container holding all service instances for a session.

    Composes:
    - AgentCore: pure-text agent (LLM + history + memory + personality)
    - Media services: ASR, TTS, VAD, Live2D
    """

    def __init__(self, config: MimosaConfig):
        """Initialize all services from configuration.

        :param config: Root configuration.
        """
        self.config = config

        # Set global cache dir BEFORE any model loading
        _setup_global_cache_dir(config)

        # Core agent (pure text, no media dependencies)
        persona_prompt = _load_persona_prompt(config.character.persona_prompt_path)
        self.agent = AgentCore(
            llm_config=config.llm,
            persona_prompt=persona_prompt,
            personality_config_dir=config.personality.config_dir,
            memory_extraction_interval=config.memory.extraction_interval,
        )

        # Load the most recent conversation history for continuity
        self.agent.load_latest_history()

        # Media services
        self.tts: TTSInterface = create_tts(config.tts)
        self.asr: ASRInterface = create_asr(config.asr)
        self.vad: VADInterface = create_vad(config.vad)

        # Live2D model
        self.live2d = Live2DModel()
        self.live2d.set_model(config.character.live2d_model_name)

        # Interaction phrase pool (LLM-generated click responses)
        self.interaction_phrases = InteractionPhrasePool()

        logger.info(f"ServiceContext initialized for character: {config.character.name}")

    # --- Delegate to AgentCore for backward compatibility ---

    @property
    def llm(self):
        """LLM interface (delegated to AgentCore)."""
        return self.agent.llm

    @property
    def chat_history(self):
        """Chat history (delegated to AgentCore)."""
        return self.agent.chat_history

    @property
    def long_term_memory(self):
        """Long-term memory (delegated to AgentCore)."""
        return self.agent.long_term_memory

    @property
    def personality(self):
        """Personality manager (delegated to AgentCore)."""
        return self.agent.personality

    @property
    def persona_prompt(self) -> str:
        """Persona prompt text (delegated to AgentCore)."""
        return self.agent.persona_prompt

    @property
    def full_system_prompt(self) -> str:
        """Get full system prompt with persona + personality + long-term memory.

        :returns: Combined system prompt string.
        """
        return self.agent.full_system_prompt

    def update_llm_config(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ):
        """Hot-reload LLM configuration without restarting the server.

        :param model: New model name (optional).
        :param temperature: New temperature 0~2 (optional).
        :param max_tokens: New max tokens (optional).
        :param api_key: New API key (optional, empty string keeps old).
        :raises ValueError: If parameter values are invalid.
        """
        self.agent.update_llm(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        # Keep config in sync
        self.config = self.config.model_copy(
            update={"llm": self.agent._llm_config}
        )
