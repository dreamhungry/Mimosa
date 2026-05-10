# -*- coding: utf-8 -*-
"""TTS factory for creating TTS instances."""

from loguru import logger

from ..config import TTSConfig
from .edge_tts_engine import EdgeTTSEngine
from .tts_interface import TTSInterface


def create_tts(config: TTSConfig) -> TTSInterface:
    """Create a TTS instance based on configuration.

    :param config: TTS configuration.
    :returns: A TTS instance implementing TTSInterface.
    :raises ValueError: If the engine is not supported.
    """
    engine = config.engine.lower()

    if engine == "edge_tts":
        tts = EdgeTTSEngine(
            voice=config.voice,
            rate=config.rate,
            volume=config.volume,
        )
        logger.info("Created Edge TTS engine")
        return tts

    raise ValueError(
        f"Unsupported TTS engine: {engine}. Supported: edge_tts"
    )
