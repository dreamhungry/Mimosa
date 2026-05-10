# -*- coding: utf-8 -*-
"""Abstract ASR interface."""

from abc import ABC, abstractmethod

import numpy as np


class ASRInterface(ABC):
    """Abstract interface for Automatic Speech Recognition."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        :param audio: Audio data as numpy float32 array.
        :param sample_rate: Sample rate of the audio.
        :returns: Transcribed text.
        """
        ...
