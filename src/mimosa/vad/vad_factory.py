# -*- coding: utf-8 -*-
"""VAD factory for creating VAD instances."""

from loguru import logger

from ..config import VADConfig
from .silero_vad import SileroVAD
from .vad_interface import VADInterface


def create_vad(config: VADConfig) -> VADInterface:
    """Create a VAD instance based on configuration.

    :param config: VAD configuration.
    :returns: A VAD instance implementing VADInterface.
    :raises ValueError: If the engine is not supported.
    """
    engine = config.engine.lower()

    if engine == "silero":
        vad = SileroVAD(
            threshold=config.threshold,
            min_silence_duration_ms=config.min_silence_duration_ms,
            speech_pad_ms=config.speech_pad_ms,
        )
        logger.info("Created Silero VAD engine")
        return vad

    raise ValueError(
        f"Unsupported VAD engine: {engine}. Supported: silero"
    )
