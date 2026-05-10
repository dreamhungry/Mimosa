# -*- coding: utf-8 -*-
"""ASR factory for creating ASR instances."""

from loguru import logger

from ..config import ASRConfig
from .asr_interface import ASRInterface


def create_asr(config: ASRConfig) -> ASRInterface:
    """Create an ASR instance based on configuration.

    :param config: ASR configuration.
    :returns: An ASR instance implementing ASRInterface.
    :raises ValueError: If the engine is not supported.
    """
    engine = config.engine.lower()

    if engine == "sherpa_onnx":
        from .sherpa_onnx_asr import SherpaOnnxASR

        asr = SherpaOnnxASR(
            model_path=config.model_path,
            sample_rate=config.sample_rate,
        )
        logger.info("Created Sherpa ONNX ASR engine")
        return asr

    if engine == "funasr":
        from .funasr_engine import FunASREngine

        asr = FunASREngine(
            model=config.model,
            device=config.device,
            vad_model=config.vad_model,
            punc_model=config.punc_model,
            sample_rate=config.sample_rate,
            cache_dir=config.cache_dir,
        )
        logger.info("Created FunASR engine")
        return asr

    raise ValueError(
        f"Unsupported ASR engine: {engine}. Supported: sherpa_onnx, funasr"
    )
