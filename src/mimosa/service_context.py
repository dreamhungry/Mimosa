# -*- coding: utf-8 -*-
"""Service context container for per-session state."""

import os
from pathlib import Path

from loguru import logger

from .asr.asr_factory import create_asr
from .asr.asr_interface import ASRInterface
from .config import MimosaConfig
from .conversation.interaction_phrase_pool import InteractionPhrasePool
from .live2d.live2d_model import Live2DModel
from .llm.llm_factory import create_llm
from .llm.llm_interface import LLMInterface
from .memory.chat_history import ChatHistory
from .memory.long_term_memory import LongTermMemory
from .personality import PersonalityManager
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


class ServiceContext:
    """Container holding all service instances for a session."""

    def __init__(self, config: MimosaConfig):
        """Initialize all services from configuration.

        :param config: Root configuration.
        """
        self.config = config

        # Set global cache dir BEFORE any model loading
        _setup_global_cache_dir(config)

        # Core services
        self.llm: LLMInterface = create_llm(config.llm)
        self.tts: TTSInterface = create_tts(config.tts)
        self.asr: ASRInterface = create_asr(config.asr)
        self.vad: VADInterface = create_vad(config.vad)

        # Memory
        self.chat_history = ChatHistory()
        self.long_term_memory = LongTermMemory()

        # Interaction phrase pool (LLM-generated click responses)
        self.interaction_phrases = InteractionPhrasePool()

        # Personality
        self.personality = PersonalityManager(
            config_dir=config.personality.config_dir
        )

        # Load the most recent conversation history for continuity
        self._load_latest_history()

        # Live2D model
        self.live2d = Live2DModel()
        self.live2d.set_model(config.character.live2d_model_name)

        # Load persona prompt
        self.persona_prompt = self._load_persona_prompt(config.character.persona_prompt_path)

        logger.info(f"ServiceContext initialized for character: {config.character.name}")

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
        updates = {}

        if model is not None:
            updates["model"] = model

        if temperature is not None:
            if not (0 <= temperature <= 2):
                raise ValueError("temperature must be between 0 and 2")
            updates["temperature"] = temperature

        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            updates["max_tokens"] = max_tokens

        if api_key is not None and api_key.strip():
            updates["api_key"] = api_key

        if not updates:
            return

        # Update config model
        self.config = self.config.model_copy(
            update={"llm": self.config.llm.model_copy(update=updates)}
        )

        # Recreate LLM instance with new config
        self.llm = create_llm(self.config.llm)
        logger.info(f"LLM config hot-reloaded: {list(updates.keys())}")

    def _load_latest_history(self):
        """Load the most recent conversation history for session continuity."""
        conversations_dir = self.chat_history.storage_dir
        if not conversations_dir.exists():
            return

        # Find the most recent conversation file
        json_files = sorted(
            conversations_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            return

        # Load the latest one
        latest = json_files[0].stem
        if self.chat_history.load(latest):
            logger.info(f"Restored previous conversation: {latest}")

    @property
    def full_system_prompt(self) -> str:
        """Get full system prompt with persona + personality + long-term memory.

        :returns: Combined system prompt string.
        """
        personality_section = self.personality.get_prompt_section()
        memory_section = self.long_term_memory.get_prompt_injection()
        return self.persona_prompt + personality_section + memory_section

    def _load_persona_prompt(self, path: str) -> str:
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
