# -*- coding: utf-8 -*-
"""Sherpa ONNX ASR implementation."""

import asyncio
from pathlib import Path

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface


class SherpaOnnxASR(ASRInterface):
    """ASR implementation using sherpa-onnx for offline recognition."""

    def __init__(self, model_path: str = "", sample_rate: int = 16000):
        """Initialize Sherpa ONNX ASR.

        :param model_path: Path to sherpa-onnx model directory.
        :param sample_rate: Expected audio sample rate.
        """
        self.sample_rate = sample_rate
        self._recognizer = None
        self._model_path = model_path
        self._init_recognizer()

    def _find_model_file(self, model_dir: Path, prefix: str, suffix: str = ".onnx") -> str:
        """Find model file by prefix, supporting names with epoch numbers.

        Looks for files like 'encoder.onnx' or 'encoder-epoch-34-avg-19.onnx'.
        Prefers int8 version if available.
        """
        # Try exact name first
        exact = model_dir / f"{prefix}{suffix}"
        if exact.exists():
            return str(exact)

        # Search for files matching the prefix pattern
        candidates = list(model_dir.glob(f"{prefix}*{suffix}"))
        if not candidates:
            return ""

        # Prefer int8 version for smaller memory footprint
        int8_candidates = [c for c in candidates if "int8" in c.name]
        if int8_candidates:
            return str(int8_candidates[0])

        return str(candidates[0])

    def _init_recognizer(self):
        """Initialize the sherpa-onnx recognizer."""
        try:
            import sherpa_onnx

            if not self._model_path or not Path(self._model_path).exists():
                logger.warning(
                    "No ASR model path configured. "
                    "Please download a sherpa-onnx model and set asr.model_path in conf.yaml. "
                    "See: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html"
                )
                return

            model_dir = Path(self._model_path)

            # Auto-detect model files
            tokens = str(model_dir / "tokens.txt")
            encoder = self._find_model_file(model_dir, "encoder")
            decoder = self._find_model_file(model_dir, "decoder")
            joiner = self._find_model_file(model_dir, "joiner")

            if not all([encoder, decoder, joiner]):
                logger.error(
                    f"Missing model files in {model_dir}. "
                    f"Need encoder/decoder/joiner .onnx files and tokens.txt"
                )
                return

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=2,
                sample_rate=self.sample_rate,
                decoding_method="greedy_search",
            )
            logger.info(f"Sherpa ONNX ASR initialized from: {model_dir}")

        except ImportError:
            logger.error("sherpa-onnx not installed. Run: pip install sherpa-onnx")
        except Exception as e:
            logger.error(f"Failed to initialize Sherpa ONNX ASR: {e}")

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text using sherpa-onnx.

        :param audio: Audio data as numpy float32 array.
        :param sample_rate: Sample rate of the audio.
        :returns: Transcribed text.
        """
        if self._recognizer is None:
            logger.warning("[ASR] Recognizer not initialized, cannot transcribe")
            return ""

        logger.info(
            f"[ASR] Starting transcription: "
            f"audio_shape={audio.shape}, sample_rate={sample_rate}"
        )

        # Run recognition in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._recognize_sync, audio, sample_rate)
        return result

    def _recognize_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        """Synchronous recognition."""
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio.tolist())
            self._recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            logger.info(f"[ASR] Recognition result: \"{text}\" (len={len(text)})")
            return text
        except Exception as e:
            logger.error(f"[ASR] Transcription failed: {e}")
            return ""
