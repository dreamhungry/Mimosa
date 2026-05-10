# -*- coding: utf-8 -*-
"""FunASR engine implementation using Paraformer model."""

import asyncio
import os

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface


class FunASREngine(ASRInterface):
    """ASR implementation using FunASR (Alibaba DAMO Academy).

    Supports paraformer-zh for Chinese/English recognition with
    significantly better accuracy than sherpa-onnx.
    """

    def __init__(
        self,
        model: str = "paraformer-zh",
        device: str = "cuda:0",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        sample_rate: int = 16000,
        cache_dir: str = "",
    ):
        """Initialize FunASR engine.

        :param model: Model name or local path.
        :param device: Device to run on ('cuda:0' or 'cpu').
        :param vad_model: VAD model for segmenting long audio.
        :param punc_model: Punctuation restoration model.
        :param sample_rate: Expected audio sample rate.
        :param cache_dir: Directory for model cache. Empty uses default.
        """
        self._model = None
        self._sample_rate = sample_rate
        self._model_name = model
        self._device = device
        self._vad_model = vad_model
        self._punc_model = punc_model
        self._cache_dir = cache_dir

        self._setup_cache_dir()
        self._init_model()

    def _setup_cache_dir(self):
        """Redirect model cache to user-specified directory."""
        if not self._cache_dir:
            return

        cache_path = os.path.abspath(self._cache_dir)
        os.makedirs(cache_path, exist_ok=True)

        # ModelScope cache (where FunASR downloads models from)
        os.environ.setdefault("MODELSCOPE_CACHE", cache_path)
        # HuggingFace cache (fallback for some models)
        os.environ.setdefault("HF_HOME", os.path.join(cache_path, "huggingface"))
        # PyTorch hub cache (for silero VAD etc.)
        os.environ.setdefault("TORCH_HOME", os.path.join(cache_path, "torch"))

        logger.info(f"Model cache directory set to: {cache_path}")

    def _init_model(self):
        """Initialize the FunASR model."""
        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self._model_name,
                vad_model=self._vad_model,
                punc_model=self._punc_model,
                device=self._device,
            )
            logger.info(
                f"FunASR initialized: model={self._model_name}, "
                f"device={self._device}"
            )
        except ImportError:
            logger.error(
                "FunASR not installed. Run: pip install funasr modelscope"
            )
        except Exception as e:
            logger.error(f"FunASR initialization failed: {e}")

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text using FunASR.

        :param audio: Audio data as numpy float32 array.
        :param sample_rate: Sample rate of the audio.
        :returns: Transcribed text.
        """
        if self._model is None:
            logger.warning("[ASR] FunASR model not initialized")
            return ""

        logger.info(
            f"[ASR] Starting transcription: "
            f"audio_shape={audio.shape}, sample_rate={sample_rate}"
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._recognize_sync, audio, sample_rate
        )
        return result

    def _recognize_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        """Synchronous recognition."""
        try:
            # FunASR expects int16 or float32 numpy array
            # Ensure audio is in the correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # FunASR generate accepts numpy array directly
            res = self._model.generate(
                input=audio,
                batch_size_s=300,
                is_final=True,
            )

            if not res:
                logger.info("[ASR] Recognition result: \"\" (len=0)")
                return ""

            # Extract text from result
            text = res[0].get("text", "").strip()
            logger.info(f"[ASR] Recognition result: \"{text}\" (len={len(text)})")
            return text

        except Exception as e:
            logger.error(f"[ASR] Transcription failed: {e}")
            return ""
