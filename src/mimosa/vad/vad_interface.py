# -*- coding: utf-8 -*-
"""Abstract VAD interface."""

from abc import ABC, abstractmethod

import numpy as np


class VADInterface(ABC):
    """Abstract interface for Voice Activity Detection."""

    @abstractmethod
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech.

        :param audio_chunk: Audio chunk as numpy float32 array.
        :returns: True if speech is detected.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset the VAD state for a new utterance."""
        ...
