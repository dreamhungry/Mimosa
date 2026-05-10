# -*- coding: utf-8 -*-
"""Service context container for per-session state."""

import os
from pathlib import Path

from loguru import logger

from .asr.asr_factory import create_asr
from .asr.asr_interface import ASRInterface
from .config import MimosaConfig
from .live2d.live2d_model import Live2DModel
from .llm.llm_factory import create_llm
from .llm.llm_interface import LLMInterface
from .memory.chat_history import ChatHistory
from .memory.long_term_memory import LongTermMemory
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

        # Load the most recent conversation history for continuity
        self._load_latest_history()

        # Live2D model
        self.live2d = Live2DModel()
        self.live2d.set_model(config.character.live2d_model_name)

        # Load persona prompt
        self.persona_prompt = self._load_persona_prompt(config.character.persona_prompt_path)

        logger.info(f"ServiceContext initialized for character: {config.character.name}")

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
        """Get full system prompt with persona + long-term memory.

        :returns: Combined system prompt string.
        """
        memory_section = self.long_term_memory.get_prompt_injection()
        return self.persona_prompt + memory_section

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
