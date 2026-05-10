# -*- coding: utf-8 -*-
"""Silero VAD implementation."""

import numpy as np
import torch
from loguru import logger

from .vad_interface import VADInterface


class SileroVAD(VADInterface):
    """VAD implementation using Silero VAD model."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 300,
        sample_rate: int = 16000,
    ):
        """Initialize Silero VAD.

        :param threshold: Speech detection threshold (0.0-1.0).
        :param min_silence_duration_ms: Minimum silence duration to consider speech ended.
        :param speech_pad_ms: Padding around detected speech.
        :param sample_rate: Audio sample rate (must be 8000 or 16000).
        """
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the Silero VAD model."""
        try:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model.eval()
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech.

        :param audio_chunk: Audio chunk as numpy float32 array (16kHz, mono).
        :returns: True if speech is detected above threshold.
        """
        if self._model is None:
            return False

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_chunk).float()

            # Run VAD inference
            speech_prob = self._model(audio_tensor, self.sample_rate).item()
            return speech_prob >= self.threshold

        except Exception as e:
            logger.error(f"VAD inference failed: {e}")
            return False

    def reset(self):
        """Reset VAD state."""
        if self._model is not None:
            self._model.reset_states()
